from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from tefla.core.special_fn import fn_with_custom_grad, conv2d_gru, conv2d_lstm
import tensorflow as tf


class FnWithCustomGradTest(tf.test.TestCase):

    def testCorrectness(self):

        w = tf.random_uniform([6, 10])

        def fn(a, b, c):
            return tf.layers.dense(
                a,
                10,
                use_bias=False,
                kernel_initializer=lambda shape, dtype, partition_info: w
            ) + tf.matmul(b, c)

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
        self.assertEqual(out.get_shape().as_list(),
                         custom_out.get_shape().as_list())

        loss = tf.reduce_mean(out)
        custom_loss = tf.reduce_mean(custom_out)

        grads = tf.gradients(loss, [a, b, c] + [tf.trainable_variables()[0]])
        custom_grads = tf.gradients(custom_loss,
                                    [a, b, c] + [tf.trainable_variables()[1]])

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
            grad_inputs = [tf.ones_like(t) * (i + 1.)
                           for i, t in enumerate(inputs)]
            grad_vars = [
                tf.ones_like(t) * (i + len(inputs) + 1.)
                for i, t in enumerate(variables)
            ]
            return grad_inputs, grad_vars

        a = tf.random_uniform([11, 6])
        b = tf.random_uniform([11, 7])
        c = tf.random_uniform([7, 10])
        w = tf.random_uniform([6, 10])
        out = fn_with_custom_grad(grad_fn)(fn)(a, b, c)
        loss = tf.reduce_mean(out)
        grads = tf.gradients(loss, [a, b, c, tf.trainable_variables()[0]])
        expected_grads = [
            tf.ones_like(t) * (i + 1.) for i, t in enumerate([a, b, c, w])
        ]
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            g_val, eg_val = sess.run([grads, expected_grads])
            for g1, g2 in zip(g_val, eg_val):
                self.assertAllClose(g1, g2)

    def testConvGRU(self):
        x = tf.convert_to_tensor(
            np.random.rand(5, 7, 3, 11), dtype=tf.float32)
        with self.test_session() as session:
            y = conv2d_gru(
                x, 11, False, None, filter_size=(1, 3))
            z = conv2d_gru(
                x, 11, False, None, filter_size=(1, 3), padding="LEFT", name='left_conv_gru')
            session.run(tf.global_variables_initializer())
            res1 = session.run(y)
            res2 = session.run(z)
        self.assertEqual(res1.shape, (5, 7, 3, 11))
        self.assertEqual(res2.shape, (5, 7, 3, 11))

    def testConvLSTM(self):
        x = tf.convert_to_tensor(
            np.random.rand(5, 7, 11, 13), dtype=tf.float32)
        with self.test_session() as session:
            y = conv2d_gru(
                x, 13, False, None, filter_size=(1, 3))
            session.run(tf.global_variables_initializer())
            res = session.run(y)
        self.assertEqual(res.shape, (5, 7, 11, 13))


if __name__ == "__main__":
    tf.test.main()
