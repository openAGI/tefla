# -*- coding: utf-8 -*-
"""Tests for SessionRunHooks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile
import shutil
import time

import tensorflow as tf
from tensorflow.python.training import monitored_session
from tensorflow import gfile

from tefla.core import learner_hooks
from tefla.utils.util import add_dict_to_collection


class TestPrintModelAnalysisHook(tf.test.TestCase):
  """Tests the `PrintModelAnalysisHook` hook"""

  def test_begin(self):
    model_dir = tempfile.mkdtemp()
    outfile = tempfile.NamedTemporaryFile()
    tf.get_variable("weigths", [128, 128])
    hook = learner_hooks.PrintModelAnalysisHook(
        params={}, model_dir=model_dir, run_config=tf.contrib.learn.RunConfig())
    hook.begin()

    with gfile.GFile(os.path.join(model_dir, "model_analysis.txt")) as file:
      file_contents = file.read().strip()

    outfile.close()


class TestTrainSampleHook(tf.test.TestCase):
  """Tests `TrainSampleHook` class.
    """

  def setUp(self):
    super(TestTrainSampleHook, self).setUp()
    self.model_dir = tempfile.mkdtemp()
    self.sample_dir = os.path.join(self.model_dir, "samples")

    # The hook expects these collections to be in the graph
    pred_dict = {}
    pred_dict["predicted_tokens"] = tf.constant([["Hello", "World", "笑w"]])
    pred_dict["labels.target_tokens"] = tf.constant([["Hello", "World", "笑w"]])
    pred_dict["labels.target_len"] = tf.constant(2),
    add_dict_to_collection(pred_dict, "predictions")

  def tearDown(self):
    super(TestTrainSampleHook, self).tearDown()
    shutil.rmtree(self.model_dir)

  def test_sampling(self):
    hook = learner_hooks.TrainSampleHook(
        params={"every_n_steps": 10},
        model_dir=self.model_dir,
        run_config=tf.contrib.learn.RunConfig())

    global_step = tf.contrib.framework.get_or_create_global_step()
    no_op = tf.no_op()
    hook.begin()
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())

      mon_sess = monitored_session._HookedSession(sess, [hook])
      # Should trigger for step 0
      sess.run(tf.assign(global_step, 0))
      mon_sess.run(no_op)

      outfile = os.path.join(self.sample_dir, "samples_000000.txt")
      with open(outfile, "rb") as readfile:
        self.assertIn("Prediction followed by Target @ Step 0", readfile.read().decode("utf-8"))

      # Should not trigger for step 9
      sess.run(tf.assign(global_step, 9))
      mon_sess.run(no_op)
      outfile = os.path.join(self.sample_dir, "samples_000009.txt")
      self.assertFalse(os.path.exists(outfile))

      # Should trigger for step 10
      sess.run(tf.assign(global_step, 10))
      mon_sess.run(no_op)
      outfile = os.path.join(self.sample_dir, "samples_000010.txt")
      with open(outfile, "rb") as readfile:
        self.assertIn("Prediction followed by Target @ Step 10", readfile.read().decode("utf-8"))


class TestMetadataCaptureHook(tf.test.TestCase):
  """Test for the MetadataCaptureHook"""

  def setUp(self):
    super(TestMetadataCaptureHook, self).setUp()
    self.model_dir = tempfile.mkdtemp()

  def tearDown(self):
    super(TestMetadataCaptureHook, self).tearDown()
    shutil.rmtree(self.model_dir)

  def test_capture(self):
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Some test computation
    some_weights = tf.get_variable("weigths", [2, 128])
    computation = tf.nn.softmax(some_weights)

    hook = learner_hooks.MetadataCaptureHook(
        params={"step": 5}, model_dir=self.model_dir, run_config=tf.contrib.learn.RunConfig())
    hook.begin()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      # Should not trigger for step 0
      sess.run(tf.assign(global_step, 0))
      mon_sess.run(computation)
      self.assertEqual(gfile.ListDirectory(self.model_dir), [])
      # Should trigger *after* step 5
      sess.run(tf.assign(global_step, 5))
      mon_sess.run(computation)
      self.assertEqual(gfile.ListDirectory(self.model_dir), [])
      mon_sess.run(computation)
      self.assertEqual(
          set(gfile.ListDirectory(self.model_dir)), set(["run_meta", "tfprof_log",
                                                         "timeline.json"]))


if __name__ == "__main__":
  tf.test.main()
