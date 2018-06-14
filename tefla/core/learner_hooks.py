# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import os

import numpy as np
import six
import yaml

import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training import session_manager
from tensorflow.python.client import timeline
from tensorflow import gfile

from .encoder import Configurable, abstractstaticmethod
from ..utils import util
from . import logger as log


@six.add_metaclass(abc.ABCMeta)
class TrainingHook(tf.train.SessionRunHook, Configurable):
  """Abstract base class for training hooks."""

  def __init__(self, params, model_dir, run_config):
    tf.train.SessionRunHook.__init__(self)
    Configurable.__init__(self, params, tf.contrib.learn.ModeKeys.TRAIN, None)
    self._model_dir = model_dir
    self._run_config = run_config

  @property
  def model_dir(self):
    """Returns the directory model checkpoints are written to."""
    return os.path.abspath(self._model_dir)

  @property
  def is_chief(self):
    """Returns true if and only if the current process is the chief.

    This is used for distributed training.
    """
    return self._run_config.is_chief

  @abstractstaticmethod
  def default_params():
    raise NotImplementedError()


class MetadataCaptureHook(TrainingHook):
  """A hook to capture metadata for a single step. Useful for performance
  debugging. It performs a full trace and saves run_metadata and Chrome
  timeline information to a file.

  Args:
    step: The step number to trace. The hook is only enable for this step.
  """

  def __init__(self, params, model_dir, run_config):
    super(MetadataCaptureHook, self).__init__(params, model_dir, run_config)
    self._active = False
    self._done = False
    self._global_step = None
    self._output_dir = os.path.abspath(self.model_dir)

  @staticmethod
  def default_params():
    return {"step": 10}

  def begin(self):
    self._global_step = tf.train.get_global_step()

  def before_run(self, _run_context):
    if not self.is_chief or self._done:
      return
    if not self._active:
      return tf.train.SessionRunArgs(self._global_step)
    else:
      log.info("Performing full trace on next step.")
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      return tf.train.SessionRunArgs(self._global_step, options=run_options)

  def after_run(self, _run_context, run_values):
    if not self.is_chief or self._done:
      return

    step_done = run_values.results
    if self._active:
      log.info("Captured full trace at step %s", step_done)
      # Create output directory
      gfile.MakeDirs(self._output_dir)

      # Save run metadata
      trace_path = os.path.join(self._output_dir, "run_meta")
      with gfile.GFile(trace_path, "wb") as trace_file:
        trace_file.write(run_values.run_metadata.SerializeToString())
        log.info("Saved run_metadata to %s", trace_path)

      # Save timeline
      timeline_path = os.path.join(self._output_dir, "timeline.json")
      with gfile.GFile(timeline_path, "w") as timeline_file:
        tl_info = timeline.Timeline(run_values.run_metadata.step_stats)
        tl_chrome = tl_info.generate_chrome_trace_format(show_memory=True)
        timeline_file.write(tl_chrome)
        log.info("Saved timeline to %s", timeline_path)

      # Save tfprof op log
      tf.profiler.write_op_log(
          graph=tf.get_default_graph(), log_dir=self._output_dir, run_meta=run_values.run_metadata)
      log.info("Saved op log to %s", self._output_dir)
      self._active = False
      self._done = True

    self._active = (step_done >= self.params["step"])


class TrainSampleHook(TrainingHook):
  """Occasionally samples predictions from the training run and prints them.

  Args:
    every_n_secs: Sample predictions every N seconds.
      If set, `every_n_steps` must be None.
    every_n_steps: Sample predictions every N steps.
      If set, `every_n_secs` must be None.
    sample_dir: Optional, a directory to write samples to.
    delimiter: Join tokens on this delimiter. Defaults to space.
  """

  def __init__(self, params, model_dir, run_config):
    super(TrainSampleHook, self).__init__(params, model_dir, run_config)
    self._sample_dir = os.path.join(self.model_dir, "samples")
    self._timer = SecondOrStepTimer(
        every_secs=self.params["every_n_secs"], every_steps=self.params["every_n_steps"])
    self._pred_dict = {}
    self._should_trigger = False
    self._iter_count = 0
    self._global_step = None
    self._source_delimiter = self.params["source_delimiter"]
    self._target_delimiter = self.params["target_delimiter"]

  @staticmethod
  def default_params():
    return {
        "every_n_secs": None,
        "every_n_steps": 1000,
        "source_delimiter": " ",
        "target_delimiter": " "
    }

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()
    self._pred_dict = util.get_dict_from_collection("predictions")
    # Create the sample directory
    if self._sample_dir is not None:
      gfile.MakeDirs(self._sample_dir)

  def before_run(self, _run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      fetches = {
          "predicted_tokens": self._pred_dict["predicted_tokens"],
          "target_words": self._pred_dict["labels.target_tokens"],
          "target_len": self._pred_dict["labels.target_len"]
      }
      return tf.train.SessionRunArgs([fetches, self._global_step])
    return tf.train.SessionRunArgs([{}, self._global_step])

  def after_run(self, _run_context, run_values):
    result_dict, step = run_values.results
    self._iter_count = step

    if not self._should_trigger:
      return None

    # Convert dict of lists to list of dicts
    result_dicts = [dict(zip(result_dict, t)) for t in zip(*result_dict.values())]

    # Print results
    result_str = ""
    result_str += "Prediction followed by Target @ Step {}\n".format(step)
    result_str += ("=" * 100) + "\n"
    for result in result_dicts:
      target_len = result["target_len"]
      predicted_slice = result["predicted_tokens"][:target_len - 1]
      target_slice = result["target_words"][1:target_len]
      result_str += self._target_delimiter.encode("utf-8").join(predicted_slice).decode(
          "utf-8") + "\n"
      result_str += self._target_delimiter.encode("utf-8").join(target_slice).decode("utf-8") \
          + "\n\n"
    result_str += ("=" * 100) + "\n\n"
    log.info(result_str)
    if self._sample_dir:
      filepath = os.path.join(self._sample_dir, "samples_{:06d}.txt".format(step))
      with gfile.GFile(filepath, "w") as file:
        file.write(result_str)
    self._timer.update_last_triggered_step(self._iter_count - 1)


class PrintModelAnalysisHook(TrainingHook):
  """Writes the parameters of the model to a file and stdout."""

  def __init__(self, params, model_dir, run_config):
    super(PrintModelAnalysisHook, self).__init__(params, model_dir, run_config)
    self._filename = os.path.join(self.model_dir, "model_analysis.txt")

  @staticmethod
  def default_params():
    return {}

  def begin(self):
    if self.is_chief:
      opts = (tf.profiler.ProfileOptionBuilder().with_max_depth(10)
              .select(['accelerator_micros']).with_file_output(self._filename).build())

      tf.profiler.profile(tf.get_default_graph(), options=opts)
    with gfile.GFile(self._filename) as file:
      log.info(file.read())


class VariableRestoreHook(TrainingHook):
  """A hooks that restored variables from a given checkpoints.

  Args:
    prefix: Variables matching this prefix are restored.
    checkpoint_path: Path to the checkpoint to restore variables from.
  """

  def __init__(self, params, model_dir, run_config):
    super(VariableRestoreHook, self).__init__(params, model_dir, run_config)
    self._saver = None

  @staticmethod
  def default_params():
    return {"prefix": "", "checkpoint_path": ""}

  def begin(self):
    variables = tf.contrib.framework.get_variables(scope=self.params["prefix"])

    def varname_in_checkpoint(name):
      """Removes the prefix from the variable name."""
      prefix_parts = self.params["prefix"].split("/")
      checkpoint_prefix = "/".join(prefix_parts[:-1])
      return name.replace(checkpoint_prefix + "/", "")

    target_names = [varname_in_checkpoint(_.op.name) for _ in variables]
    restore_map = {k: v for k, v in zip(target_names, variables)}

    log.info("Restoring variables: \n%s", yaml.dump({k: v.op.name for k, v in restore_map.items()}))

    self._saver = tf.train.Saver(restore_map)

  def after_create_session(self, session, coord):
    self._saver.restore(session, self.params["checkpoint_path"])
    log.info("Successfully restored all variables")


class DelayStartHook(TrainingHook, tf.train.GlobalStepWaiterHook):
  """Delays the start of the current worker process until global step.

  K * task_id is reached. K is a parameter.
  """

  def __init__(self, params, model_dir, run_config):
    TrainingHook.__init__(self, params, model_dir, run_config)
    self._task_id = self._run_config.task_id
    self._delay_k = self.params["delay_k"]
    self._wait_until_step = int(self._delay_k * self._task_id)
    tf.train.GlobalStepWaiterHook.__init__(self, self._wait_until_step)

  @staticmethod
  def default_params():
    return {"delay_k": 500}


class SyncReplicasOptimizerHook(TrainingHook):
  """A SessionRunHook handles ops related to SyncReplicasOptimizer."""

  def __init__(self, params, model_dir, run_config):
    super(SyncReplicasOptimizerHook, self).__init__(params, model_dir, run_config)
    self._sync_optimizer = None
    self._num_tokens = -1

    self._local_init_op = None
    self._ready_for_local_init_op = None
    self._q_runner = None
    self._init_tokens_op = None

  @staticmethod
  def default_params():
    return {}

  def begin(self):
    if self._sync_optimizer is None:
      return

    if self._sync_optimizer._gradients_applied is False:
      raise ValueError("SyncReplicasOptimizer.apply_gradient should be called before using "
                       "the hook.")
    if self.is_chief:
      self._local_init_op = self._sync_optimizer.chief_init_op
      self._ready_for_local_init_op = (self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = self._sync_optimizer.get_chief_queue_runner()
      self._init_tokens_op = self._sync_optimizer.get_init_tokens_op(self._num_tokens)
    else:
      self._local_init_op = self._sync_optimizer.local_step_init_op
      self._ready_for_local_init_op = (self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = None
      self._init_tokens_op = None

  def after_create_session(self, session, coord):
    """Runs SyncReplicasOptimizer initialization ops."""

    if not self._sync_optimizer:
      return

    log.info("Found SyncReplicasOptimizer. Initializing.")

    local_init_success, msg = session_manager._ready(
        self._ready_for_local_init_op, session,
        "Model is not ready for SyncReplicasOptimizer local init.")
    if not local_init_success:
      raise RuntimeError("Init operations did not make model ready for SyncReplicasOptimizer "
                         "local_init. Init op: %s, error: %s" % (self._local_init_op.name, msg))
    session.run(self._local_init_op)
    if self._init_tokens_op is not None:
      session.run(self._init_tokens_op)
    if self._q_runner is not None:
      self._q_runner.create_threads(session, coord=coord, daemon=True, start=True)
