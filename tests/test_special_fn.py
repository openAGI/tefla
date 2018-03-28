from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from tefla.core.special_fn import fn_with_custom_grad, conv2d_gru, conv2d_lstm, multiscale_conv2d_sum, conv1d_memory_efficient, clip_variables
import tensorflow as tf


class FnWithCustomGradTest(tf.test.TestCase):

  def testCorrectness(self):

    w = tf.random_uniform([6, 10])

    def fn(a, b, c):
      return tf.layers.dense(
          a, 10, use_bias=False,
          kernel_initializer=lambda shape, dtype, partition_info: w) + tf.matmul(b, c)

    def grad_fn(inputs, variables, outputs, grad_outputs):
      outputs = outputs[0]
      grad_outputs = grad_outputs[0]
      grad_inputs = tf.gradients(outputs, inputs, grad_ys=grad_outputs)
      grad_vars = tf.gradients(outputs, variables, grad_ys=grad_outputs)
      return grad_inputs, grad_vars

    custom_fn = fn_with_custom_grad(grad_fn)(fn)

    a = tf.random_uniform([11, 6])
    b = tf.random_uniform([11, 7])
    c = tf.random_uniform([7, 10])

    out = fn(a, b, c)
    custom_out = custom_fn(a, b, c)
    self.assertEqual(out.get_shape().as_list(), custom_out.get_shape().as_list())

    loss = tf.reduce_mean(out)
    custom_loss = tf.reduce_mean(custom_out)

    grads = tf.gradients(loss, [a, b, c] + [tf.trainable_variables()[0]])
    custom_grads = tf.gradients(custom_loss, [a, b, c] + [tf.trainable_variables()[1]])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out_val, custom_out_val, grads_val, custom_grads_val = sess.run(
          [out, custom_out, grads, custom_grads])
      self.assertAllClose(out_val, custom_out_val)
      for g1, g2 in zip(grads_val, custom_grads_val):
        self.assertAllClose(g1, g2)

  def testCustomGrad(self):

    def fn(a, b, c):
      return tf.layers.dense(a, 10, use_bias=False) + tf.matmul(b, c)

    def grad_fn(inputs, variables, unused_outputs, unused_grad_outputs):
      grad_inputs = [tf.ones_like(t) * (i + 1.) for i, t in enumerate(inputs)]
      grad_vars = [tf.ones_like(t) * (i + len(inputs) + 1.) for i, t in enumerate(variables)]
      return grad_inputs, grad_vars

    a = tf.random_uniform([11, 6])
    b = tf.random_uniform([11, 7])
    c = tf.random_uniform([7, 10])
    w = tf.random_uniform([6, 10])
    out = fn_with_custom_grad(grad_fn)(fn)(a, b, c)
    loss = tf.reduce_mean(out)
    grads = tf.gradients(loss, [a, b, c, tf.trainable_variables()[0]])
    expected_grads = [tf.ones_like(t) * (i + 1.) for i, t in enumerate([a, b, c, w])]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      g_val, eg_val = sess.run([grads, expected_grads])
      for g1, g2 in zip(g_val, eg_val):
        self.assertAllClose(g1, g2)

  def testConvGRU(self):
    x = tf.convert_to_tensor(np.random.rand(5, 7, 3, 11), dtype=tf.float32)
    with self.test_session() as session:
      y = conv2d_gru(x, 11, False, None, filter_size=(1, 3))
      z = conv2d_gru(x, 11, False, None, filter_size=(1, 3), padding="LEFT", name='left_conv_gru')
      session.run(tf.global_variables_initializer())
      res1 = session.run(y)
      res2 = session.run(z)
    self.assertEqual(res1.shape, (5, 7, 3, 11))
    self.assertEqual(res2.shape, (5, 7, 3, 11))

  def testConvLSTM(self):
    x = tf.convert_to_tensor(np.random.rand(5, 7, 11, 13), dtype=tf.float32)
    with self.test_session() as session:
      y = conv2d_gru(x, 13, False, None, filter_size=(1, 3))
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 11, 13))

  def testMultiscaleConvSum(self):
    x = tf.convert_to_tensor(np.random.rand(5, 9, 1, 11), dtype=tf.float32)
    with self.test_session() as session:
      y = multiscale_conv2d_sum(
          x, 13, False, None, [((1, 1), (5, 5)), ((2, 2), (3, 3))], "AVG", padding="SAME")
      session.run(tf.global_variables_initializer())
      res = session.run(y)
      self.assertEqual(res.shape, (5, 9, 1, 13))

  def testConv1dMemoryEfficient(self):
    batch = 3
    length = 23
    io_size = 16
    filter_size = 7
    x = np.random.rand(batch, length, io_size)
    dy = np.random.rand(batch, length, io_size)
    with self.test_session() as session:
      x = tf.to_float(x)
      dy = tf.to_float(dy)
      f1 = tf.get_variable("f1", [1, io_size, filter_size])
      f2 = tf.get_variable("f2", [1, filter_size, io_size])
      norm_scale, norm_bias = layer_norm_vars(io_size)
      y = conv1d_memory_efficient(
          x, filter_size, True, None, forget=False, test_vars=(f1, f2, norm_scale, norm_bias))
      y_forget = conv1d_memory_efficient(
          x, filter_size, True, None, forget=True, test_vars=(f1, f2, norm_scale, norm_bias))
      dx, df1, df2, dnorm_scale, dnorm_bias = tf.gradients(
          ys=[y], xs=[x, f1, f2, norm_scale, norm_bias], grad_ys=[dy])
      dx_f, df1_f, df2_f, dnorm_scale_f, dnorm_bias_f = tf.gradients(
          ys=[y_forget], xs=[x, f1, f2, norm_scale, norm_bias], grad_ys=[dy])
      session.run(tf.global_variables_initializer())
      (y, y_forget, dx, df1, df2, dnorm_scale, dnorm_bias, dx_f, df1_f, df2_f, dnorm_scale_f,
       dnorm_bias_f) = session.run([
           y, y_forget, dx, df1, df2, dnorm_scale, dnorm_bias, dx_f, df1_f, df2_f, dnorm_scale_f,
           dnorm_bias_f
       ])
    self.assertAllClose(y, y_forget)
    self.assertAllClose(df2, df2_f)
    self.assertAllClose(df1, df1_f)
    self.assertAllClose(dnorm_scale, dnorm_scale_f)
    self.assertAllClose(dnorm_bias, dnorm_bias_f)
    self.assertAllClose(dx, dx_f)


class ClipWeightsTest(tf.test.TestCase):
  """Tests for `discriminator_weight_clip`."""

  def setUp(self):
    self.variables = [tf.Variable(2.0)]

  def _test_weight_clipping_helper(self):
    loss = self.variables[0]
    opt = tf.train.GradientDescentOptimizer(1.0)
    opt_clip = clip_variables(opt, self.variables, 0.1)

    train_op1 = opt.minimize(loss, var_list=self.variables)
    train_op2 = opt_clip.minimize(loss, var_list=self.variables)

    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(2.0, self.variables[0].eval())
      sess.run(train_op1)
      self.assertLess(0.1, self.variables[0].eval())

    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(2.0, self.variables[0].eval())
      sess.run(train_op2)
      self.assertNear(0.1, self.variables[0].eval(), 1e-7)

  def test_weight_clipping_argsonly(self):
    self._test_weight_clipping_helper()

  def _test_incorrect_weight_clip_value_helper(self):
    opt = tf.train.GradientDescentOptimizer(1.0)

    with self.assertRaisesRegexp(ValueError, 'must be positive'):
      clip_variables(opt, self.variables, weight_clip=-1)

  def test_incorrect_weight_clip_value_argsonly(self):
    self._test_incorrect_weight_clip_value_helper()


def layer_norm_vars(filters):
  """Create Variables for layer norm."""
  scale = tf.get_variable("layer_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable("layer_norm_bias", [filters], initializer=tf.zeros_initializer())
  return scale, bias


if __name__ == "__main__":
  tf.test.main()
