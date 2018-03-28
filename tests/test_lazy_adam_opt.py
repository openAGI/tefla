"""Tests for LazyAdamOptimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tefla.core.optimizer import LazyAdamOptimizer


def adam_update_numpy(param, g_t, t, m, v, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class LazyAdamOptimizerTest(tf.test.TestCase):

  def testSparse(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = tf.Variable(var0_np)
        var1 = tf.Variable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = tf.IndexedSlices(
            tf.constant(grads0_np), tf.constant(grads0_np_indices), tf.constant([2]))
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = tf.IndexedSlices(
            tf.constant(grads1_np), tf.constant(grads1_np_indices), tf.constant([2]))
        opt = LazyAdamOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSparseDevicePlacement(self):
    for index_dtype in [tf.int32, tf.int64]:
      with self.test_session(force_gpu=tf.test.is_gpu_available()):
        # If a GPU is available, tests that all optimizer tf can be placed on
        # it (i.e. they have GPU kernels).
        var = tf.Variable([[1.0], [2.0]])
        indices = tf.constant([0, 1], dtype=index_dtype)
        gathered_sum = tf.reduce_sum(tf.gather(var, indices))
        optimizer = LazyAdamOptimizer(3.0)
        minimize_op = optimizer.minimize(gathered_sum)
        tf.global_variables_initializer().run()
        minimize_op.run()

  def testSparseRepeatedIndices(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        repeated_index_update_var = tf.Variable([[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = tf.Variable([[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = tf.IndexedSlices(
            tf.constant([0.1, 0.1], shape=[2, 1], dtype=dtype), tf.constant([1, 1]),
            tf.constant([2, 1]))
        grad_aggregated = tf.IndexedSlices(
            tf.constant([0.2], shape=[1, 1], dtype=dtype), tf.constant([1]), tf.constant([2, 1]))
        repeated_update_opt = LazyAdamOptimizer()
        repeated_update = repeated_update_opt.apply_gradients([(grad_repeated_index,
                                                                repeated_index_update_var)])
        aggregated_update_opt = LazyAdamOptimizer()
        aggregated_update = aggregated_update_opt.apply_gradients([(grad_aggregated,
                                                                    aggregated_update_var)])
        tf.global_variables_initializer().run()
        self.assertAllClose(aggregated_update_var.eval(), repeated_index_update_var.eval())
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(aggregated_update_var.eval(), repeated_index_update_var.eval())


if __name__ == "__main__":
  tf.test.main()
