"""Tests for tf.contrib.training.device_setter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from tefla.core import device_setter as device_setter_lib

_CLUSTER_SPEC = tf.train.ClusterSpec({
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
})

MockOperation = collections.namedtuple("MockOperation", "name")


class RandomStrategyTest(tf.test.TestCase):

  def testBasic(self):
    ps_strategy = device_setter_lib.RandomStrategy(2, seed=0)
    with tf.device(tf.train.replica_device_setter(cluster=_CLUSTER_SPEC, ps_strategy=ps_strategy)):
      u = tf.Variable(tf.zeros([2, 2]))
      v = tf.Variable(tf.zeros([2, 1]))
      w = tf.Variable(tf.zeros([2, 2]))
      x = tf.Variable(tf.zeros([1, 3]))
      a = v + w
      # Randomly distributed with seed 0.
      self.assertDeviceEqual("/job:ps/task:1", u.device)
      self.assertDeviceEqual("/job:ps/task:1", u.initializer.device)
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", x.device)
      self.assertDeviceEqual("/job:ps/task:1", x.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  def testHandlesUnicode(self):
    op = MockOperation(u"A unicode \u018e string \xf1")
    ps_strategy = device_setter_lib.RandomStrategy(2, seed=0)
    ps_task = ps_strategy(op)
    self.assertEqual(ps_task, 1)


class GreedyLoadBalancingStrategyTest(tf.test.TestCase):

  def testUniformLoadEqualsRoundRobin(self):

    def _load_fn(unused_op):
      return 1

    with tf.device(
        tf.train.replica_device_setter(
            cluster=_CLUSTER_SPEC,
            ps_strategy=device_setter_lib.GreedyLoadBalancingStrategy(2, _load_fn))):
      u = tf.Variable(tf.zeros([2, 2]))
      v = tf.Variable(tf.zeros([2, 1]))
      w = tf.Variable(tf.zeros([2, 2]))
      x = tf.Variable(tf.zeros([1, 3]))
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", u.device)
      self.assertDeviceEqual("/job:ps/task:0", u.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", v.device)
      self.assertDeviceEqual("/job:ps/task:1", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:0", w.device)
      self.assertDeviceEqual("/job:ps/task:0", w.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", x.device)
      self.assertDeviceEqual("/job:ps/task:1", x.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  def testByteSizeLoadFn(self):
    with tf.device(
        tf.train.replica_device_setter(
            cluster=_CLUSTER_SPEC,
            ps_strategy=device_setter_lib.GreedyLoadBalancingStrategy(
                2, device_setter_lib.byte_size_load_fn))):
      u = tf.Variable(tf.zeros([2, 2]))
      v = tf.Variable(tf.zeros([2, 1]))
      w = tf.Variable(tf.zeros([2, 2]))
      x = tf.Variable(tf.zeros([1, 3]))
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", u.device)
      self.assertDeviceEqual("/job:ps/task:0", u.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", v.device)
      self.assertDeviceEqual("/job:ps/task:1", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:ps/task:0", x.device)
      self.assertDeviceEqual("/job:ps/task:0", x.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  def testByteSizeLoadFnWithScalar(self):
    with tf.device(
        tf.train.replica_device_setter(
            cluster=_CLUSTER_SPEC,
            ps_strategy=device_setter_lib.GreedyLoadBalancingStrategy(
                2, device_setter_lib.byte_size_load_fn))):
      # Note: we must test the load function as part of the device function
      # instead of passing u.op to the function directly, because the only
      # time that the output Tensor has unknown shape for scalars is during
      # Variable construction.
      u = tf.Variable(0)
      self.assertDeviceEqual("/job:ps/task:0", u.device)
      self.assertDeviceEqual("/job:ps/task:0", u.initializer.device)


if __name__ == "__main__":
  tf.test.main()
