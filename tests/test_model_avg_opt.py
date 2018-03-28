"""Tests for ModelAverageOptimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import portpicker
import tensorflow as tf

from tefla.core.optimizer import ModelAverageOptimizer, ModelAverageCustomGetter, GLOBAL_VARIABLE_NAME


def create_local_cluster(num_workers, num_ps, protocol="grpc"):
  """Create local GRPC servers and return them."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  cs = tf.train.ClusterSpec(cluster_dict)

  workers = [
      tf.train.Server(cs, job_name="worker", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_workers)
  ]
  ps_servers = [
      tf.train.Server(cs, job_name="ps", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_ps)
  ]

  return cluster_dict, workers, ps_servers


# Creates the workers and return their sessions, graphs, train_ops.
# Cheif worker will update at last
def _get_workers(num_workers, steps, workers):
  sessions = []
  graphs = []
  train_ops = []
  for worker_id in range(num_workers):
    graph = tf.Graph()
    is_chief = (worker_id == 0)
    with graph.as_default():
      worker_device = "/job:worker/task:%d/cpu:0" % (worker_id)
      ma_coustom = ModelAverageCustomGetter(worker_device=worker_device)
      with tf.variable_scope(
          '', custom_getter=ma_coustom), tf.device(
              tf.train.replica_device_setter(
                  worker_device=worker_device, ps_device="/job:ps/task:0/cpu:0", ps_tasks=1)):

        global_step = tf.Variable(0, name='global_step', trainable=False)
        var_0 = tf.get_variable(initializer=0.0, name="v0")
        var_1 = tf.get_variable(initializer=1.0, name="v1")

      with tf.device("/job:worker/task:" + str(worker_id)):
        if worker_id == 0:
          grads_0 = tf.constant(-1.0)
          grads_1 = tf.constant(-1.0)
        else:
          grads_0 = tf.constant(-2.0)
          grads_1 = tf.constant(-2.0)
        sgd_opt = tf.train.GradientDescentOptimizer(1.0)
        opt = ModelAverageOptimizer(
            opt=sgd_opt,
            num_worker=num_workers,
            ma_custom_getter=ma_coustom,
            is_chief=is_chief,
            interval_steps=steps)
        train_op = [opt.apply_gradients([[grads_0, var_0], [grads_1, var_1]], global_step)]
      easgd_hook = opt.make_session_run_hook()
      # Creates MonitoredSession
      sess = tf.train.MonitoredTrainingSession(workers[worker_id].target, hooks=[easgd_hook])

    sessions.append(sess)
    graphs.append(graph)
    train_ops.append(train_op)
  return sessions, graphs, train_ops


class ModelAverageOptimizerTest(tf.test.TestCase):

  def _run(self, train_op, sess):
    sess.run(train_op)

  def test1Workers2Period(self):
    num_workers = 2
    steps = 2
    num_ps = 1
    cluster, workers, _ = create_local_cluster(num_workers=num_workers, num_ps=num_ps)

    sessions, graphs, train_ops = _get_workers(num_workers, steps, workers)

    var_0 = graphs[0].get_tensor_by_name('v0:0')
    var_1 = graphs[0].get_tensor_by_name('v1:0')
    global_step = tf.train.get_global_step(graphs[0])
    global_var_0 = graphs[0].get_tensor_by_name(GLOBAL_VARIABLE_NAME + "/v0:0")
    global_var_1 = graphs[0].get_tensor_by_name(GLOBAL_VARIABLE_NAME + "/v1:0")

    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(global_var_0))
    self.assertAllEqual(1.0, sessions[0].run(global_var_1))
    self.assertAllEqual(0, sessions[0].run(global_step))

    sessions[0].run(train_ops[0])
    sessions[1].run(train_ops[1])

    self.assertAllEqual(1.0, sessions[0].run(var_0))
    self.assertAllEqual(2.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(global_var_0))
    self.assertAllEqual(1.0, sessions[0].run(global_var_1))
    self.assertAllEqual(0, sessions[0].run(global_step))

    # iteration 2, global varibale update
    thread_0 = self.checkedThread(target=self._run, args=(train_ops[0], sessions[0]))
    thread_1 = self.checkedThread(target=self._run, args=(train_ops[1], sessions[1]))
    thread_0.start()
    thread_1.start()
    thread_0.join()
    thread_1.join()

    self.assertAllEqual(3.0, sessions[0].run(var_0))
    self.assertAllEqual(4.0, sessions[0].run(var_1))
    self.assertAllEqual(3.0, sessions[0].run(global_var_0))
    self.assertAllEqual(4.0, sessions[0].run(global_var_1))
    self.assertAllEqual(1, sessions[0].run(global_step))

    # iteration 3
    sessions[0].run(train_ops[0])

    self.assertAllEqual(4.0, sessions[0].run(var_0))
    self.assertAllEqual(5.0, sessions[0].run(var_1))
    self.assertAllEqual(3.0, sessions[0].run(global_var_0))
    self.assertAllEqual(4.0, sessions[0].run(global_var_1))
    self.assertAllEqual(1, sessions[0].run(global_step))

  def testPS2TasksWithClusterSpecClass(self):
    cluster_spec = tf.train.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    worker_device = "/job:worker/task:0"
    ma_coustom = ModelAverageCustomGetter(worker_device=worker_device)
    from tensorflow.python.training import device_setter
    with tf.device(
            tf.train.replica_device_setter(cluster=cluster_spec,
                                           worker_device=worker_device,
                                           ps_device="/job:ps")), \
            tf.variable_scope('', custom_getter=ma_coustom):
      v = tf.get_variable(initializer=[1, 2], name="v")
      w = tf.get_variable(initializer=[2, 1], name='w')
      v_g, w_g = ma_coustom._local_2_global[v], ma_coustom._local_2_global[w]
      self.assertDeviceEqual("/job:worker/task:0", v.device)
      self.assertDeviceEqual("job:ps/task:0", v_g.device)
      self.assertDeviceEqual("/job:worker/task:0", w.device)
      self.assertDeviceEqual("job:ps/task:1", w_g.device)


if __name__ == '__main__':
  tf.test.main()
