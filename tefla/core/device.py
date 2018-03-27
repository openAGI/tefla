"""Device placement and data parallelism."""
from __future__ import absolute_import, division, print_function
from six.moves import xrange
from tensor2tensor.utils import expert_utils as eu
import tensorflow as tf


def _ps_replicas(ps_replicas, worker_replicas, worker_id, all_workers=False):
  if all_workers:
    return list(range(ps_replicas))
  # Worker K will be using replicas {0,...n-1} + K*n if we have n replicas.
  num_replicas = ps_replicas // worker_replicas
  return [d + worker_id * num_replicas for d in xrange(num_replicas)]


def _gpu_order(num_gpus, gpu_order):
  if gpu_order:
    ret = [int(s) for s in gpu_order.split(" ")]
    if len(ret) == num_gpus:
      return ret
  return list(range(num_gpus))


def _ps_gpus(ps_replicas, worker_replicas, worker_id, num_gpus, ps_gpu=None, all_workers=False):
  ps_gpus = []
  for d in _ps_replicas(ps_replicas, worker_replicas, worker_id, all_workers=all_workers):
    ps_gpus.extend([(d, gpu) for gpu in _gpu_order(num_gpus, ps_gpu)])
  return ps_gpus


def ps_devices(ps_replicas,
               worker_replicas,
               ps_job,
               num_gpus,
               worker_gpu,
               ps_gpu=None,
               all_workers=False):
  """List of ps devices (where to put the experts).

  Args:
    all_workers: whether the list is for all async workers or just this one.

  Returns:
    a list of device names
  """
  if ps_replicas > 0:
    if ps_gpu > 0:
      return [
          ps_job + "/task:%d/GPU:%d" % (d, gpu) for (d, gpu) in _ps_gpus(
              ps_replicas, worker_replicas, ps_job, num_gpus, ps_gpu, all_workers=all_workers)
      ]
    else:
      return [
          ps_job + "/task:%d" % d for d in _ps_replicas(
              ps_replicas, worker_replicas, ps_job, num_gpus, all_workers=all_workers)
      ]
  else:
    if worker_gpu > 0:
      return ["gpu:%d" % d for d in _gpu_order(num_gpus, worker_gpu)]
    else:
      return [""]


def data_parallelism(ps_replicas,
                     worker_replicas,
                     worker_gpu,
                     locally_shard_to_cpu,
                     sync=False,
                     all_workers=False):
  """Over which devices do we split each training batch.

  In old-fashioned async mode, we split the batch over all GPUs on the
  current worker.

  In sync mode, we split the batch over all the parameter server GPUs.

  This function returns an expert_utils.Parallelism object, which can be used
  to build the model.  It is configured in a way that any variables created
  by `tf.get_variable` will be assigned to the parameter servers and shared
  between datashards.

  Args:
    all_workers: whether the devices are all async workers or just this one.

  Returns:
    a expert_utils.Parallelism.
  """

  def _replica_device_setter(ps_gpu, ps_job, worker_device):
    if ps_replicas == 0:
      return worker_device
    return tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_tasks=ps_replicas,
        ps_device=ps_job + "/GPU:0" if ps_gpu > 0 else ps_job)

  if schedule == "train_and_evaluate":
    assert not sync
    datashard_devices = ["gpu:%d" % d for d in _gpu_order(worker_gpu)]
    if locally_shard_to_cpu or worker_gpu < 1:
      datashard_devices += ["cpu:0"]
    caching_devices = None
  elif sync:
    assert ps_replicas > 0
    datashard_devices = [_replica_device_setter(d) for d in ps_devices(all_workers=all_workers)]
    if ps_gpu > 0 and ps_replicas > 1:
      caching_devices = [
          ps_job + "/task:%d/cpu:0" % d for (d, _) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      caching_devices = None
  else:
    # old fashioned async - compute on worker
    if FLAGS.worker_gpu > 1:
      datashard_devices = [
          _replica_device_setter(FLAGS.worker_job + "/GPU:%d" % d)
          for d in _gpu_order(FLAGS.worker_gpu)
      ]
      caching_devices = [FLAGS.worker_job + "/GPU:0"] * FLAGS.worker_gpu
    else:
      datashard_devices = [_replica_device_setter(FLAGS.worker_job)]
      caching_devices = None
  tf.logging.info("datashard_devices: %s", datashard_devices)
  tf.logging.info("caching_devices: %s", caching_devices)
  return eu.Parallelism(
      datashard_devices,
      reuse=True,
      caching_devices=caching_devices,
      daisy_chain_variables=FLAGS.daisy_chain_variables)
