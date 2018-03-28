"""Tests for snt.clip_gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
from tefla.core.special_fn import clip_gradient, scale_gradient


class ClipGradientTest(tf.test.TestCase):

  def testOpClip(self):
    x = tf.placeholder(tf.float32, shape=[2, 1])
    y = clip_gradient(x, 2, 3)
    z = tf.reduce_sum(y * y)
    dzdy = tf.gradients(z, y)[0]
    dzdx = tf.gradients(z, x)[0]

    x_np = np.array([[0.5], [2]])
    with self.test_session() as sess:
      y_np, dzdy_np, dzdx_np = sess.run([y, dzdy, dzdx], feed_dict={x: x_np})

      self.assertAllEqual(y_np, x_np)
      # We do not expect the gradients with respect to the output to be clipped.
      self.assertAllEqual(dzdy_np, np.array([[1], [4]]))
      # We expect the gradients with respect to the input to be clipped [2, 3].
      self.assertAllEqual(dzdx_np, np.array([[2], [3]]))

  def testShape(self):
    x = tf.placeholder(tf.float32, [None, 10, 13])
    y = clip_gradient(x, 0, 1)
    z = tf.reduce_sum(y * y)
    dzdx = tf.gradients(z, x)[0]

    self.assertAllEqual(y.get_shape().as_list(), [None, 10, 13])
    self.assertAllEqual(dzdx.get_shape().as_list(), [None, 10, 13])


class ScaleGradientTest(tf.test.TestCase):

  def testTwoOps(self):
    """Tests that the op can be instantiated twice with appropriate results.
        Implementations with inappropriate global registration of gradients will
        fail this test.
        """

    x = tf.placeholder(tf.float32, [1])
    y = x * x
    y = scale_gradient(y, 0.1)
    y = scale_gradient(y, 0.1)
    dydx = tf.gradients([y], [x])[0]

    with self.test_session() as sess:
      dydx_, y_ = sess.run([dydx, y], feed_dict={x: [3.0]})

    self.assertAlmostEqual(dydx_[0], 2 * 0.1**2 * 3.0, places=6)
    self.assertAlmostEqual(y_[0], 3.0**2, places=6)

  def testShape(self):
    x = tf.placeholder(tf.float32, [None, 10, 13])
    y = scale_gradient(x, 0.1)
    shape = tuple(y.get_shape().as_list())
    self.assertEqual(shape, (None, 10, 13))


if __name__ == "__main__":
  tf.test.main()
