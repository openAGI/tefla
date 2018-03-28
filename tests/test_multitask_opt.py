"""Tests for MultitaskOptimizerWrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from tefla.core.optimizer import MultitaskOptimizer, clip_gradients_by_global_norm


class MultitaskOptimizerTest(tf.test.TestCase):
  """Tests for the multitask optimizer wrapper.
    """

  def testWrapper(self):
    with self.test_session():
      var0 = tf.Variable([1.0, 2.0], dtype=tf.float32)
      var1 = tf.Variable([3.0, 4.0], dtype=tf.float32)
      grads0 = tf.constant([0.1, 0.1], dtype=tf.float32)
      grads1 = tf.constant([0.01, 0.01], dtype=tf.float32)
      grads_allzero = tf.constant([0.0, 0.0], dtype=tf.float32)
      mom_opt_impl = tf.train.MomentumOptimizer(learning_rate=2.0, momentum=0.9)
      mom_opt = MultitaskOptimizer(mom_opt_impl)
      mom_update = mom_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      mom_update_partial = mom_opt.apply_gradients(zip([grads_allzero, grads1], [var0, var1]))
      mom_update_no_action = mom_opt.apply_gradients(
          zip([grads_allzero, grads_allzero], [var0, var1]))
      self.evaluate(tf.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))

      self.assertEqual(["momentum"], mom_opt.get_slot_names())
      slot0 = mom_opt.get_slot(var0, "momentum")
      self.assertEquals(slot0.get_shape(), var0.get_shape())
      slot1 = mom_opt.get_slot(var1, "momentum")
      self.assertEquals(slot1.get_shape(), var1.get_shape())

      # Step 1: normal momentum update.
      self.evaluate(mom_update)
      # Check that the momentum accumulators have been updated.
      self.assertAllCloseAccordingToType(np.array([0.1, 0.1]), self.evaluate(slot0))
      self.assertAllCloseAccordingToType(np.array([0.01, 0.01]), self.evaluate(slot1))
      # Check that the parameters have been updated.
      self.assertAllCloseAccordingToType(
          np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]), self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]), self.evaluate(var1))

      # Step 2: momentum update that changes only slot1 but not slot0.
      self.evaluate(mom_update_partial)
      # Check that only the relevant momentum accumulator has been updated.
      self.assertAllCloseAccordingToType(np.array([0.1, 0.1]), self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]), self.evaluate(slot1))

      # Step 3: momentum update that does not change anything.
      self.evaluate(mom_update_no_action)
      # Check that the momentum accumulators have *NOT* been updated.
      self.assertAllCloseAccordingToType(np.array([0.1, 0.1]), self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]), self.evaluate(slot1))

  def testGradientClipping(self):
    with self.test_session():
      var0 = tf.Variable([1.0, 2.0], dtype=tf.float32)
      var1 = tf.Variable([3.0, 4.0], dtype=tf.float32)
      var2 = tf.Variable([3.0, 4.0], dtype=tf.float32)
      var3 = tf.Variable([3.0, 4.0], dtype=tf.float32)
      grads0 = tf.constant([10.0, 15.0], dtype=tf.float32)
      grads1 = tf.constant([0.0, 5.0], dtype=tf.float32)
      grads2 = tf.constant([0.0, 0.0], dtype=tf.float32)
      grads3 = None
      varlist = [var0, var1, var2, var3]
      gradients = [grads0, grads1, grads2, grads3]
      clipped_gradvars, global_norm = (clip_gradients_by_global_norm(
          six.moves.zip(gradients, varlist), clip_norm=1.0))
      clipped_grads = list(six.moves.zip(*clipped_gradvars))[0]
      reference_global_norm = np.sqrt(np.sum(np.square([10.0, 15.0, 0.0, 5.0])))
      self.assertAllCloseAccordingToType(self.evaluate(global_norm), reference_global_norm)
      self.assertAllCloseAccordingToType(self.evaluate(clipped_grads[2]), np.array([0., 0.]))
      self.assertEqual(clipped_grads[3], None)


if __name__ == "__main__":
  tf.test.main()
