"""Tests for common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.core import diet


class DietVarTest(tf.test.TestCase):

  def testDiet(self):

    params = diet.diet_adam_optimizer_params()

    @diet.fn_with_diet_vars(params)
    def model_fn(x):
      y = tf.layers.dense(x, 10, use_bias=False)
      return y

    @diet.fn_with_diet_vars(params)
    def model_fn2(x):
      y = tf.layers.dense(x, 10, use_bias=False)
      return y

    x = tf.random_uniform((10, 10))
    y = model_fn(x) + 10.
    y = model_fn2(y) + 10.
    grads = tf.gradients(y, [x])
    with tf.control_dependencies(grads):
      incr_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)

    train_op = tf.group(incr_step, *grads)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      orig_vals = sess.run(tf.global_variables())
      for _ in range(10):
        sess.run(train_op)
      new_vals = sess.run(tf.global_variables())

      different = []
      for old, new in zip(orig_vals, new_vals):
        try:
          self.assertAllClose(old, new)
        except AssertionError:
          different.append(True)
      self.assertEqual(len(different), len(tf.global_variables()))


if __name__ == "__main__":
  tf.test.main()
