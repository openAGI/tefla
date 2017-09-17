from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import random


import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.framework import random_seed

from tefla.core import rnn_cell


class RNN_CellTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.set_random_seed(1)

    def testAttentionCellFailures(self):
        with self.assertRaisesRegexp(TypeError, "The parameter cell is not RNNCell."):
            rnn_cell.AttentionCell(None, 0, None)

        num_units = 8
        with tf.Graph().as_default():
            lstm_cell = rnn_cell.LSTMCell(
                num_units, None)
            with self.assertRaisesRegexp(ValueError, "attn_length should be greater than zero, got 0"):
                rnn_cell.AttentionCell(
                    lstm_cell, 0, None)
            with self.assertRaisesRegexp(ValueError, "attn_length should be greater than zero, got -1"):
                rnn_cell.AttentionCell(
                    lstm_cell, -1, True)

    def testAttentionCellZeros(self):
        num_units = 8
        attn_length = 16
        batch_size = 3
        input_size = 4
        with tf.Graph().as_default():
            with self.test_session() as sess:
                with tf.variable_scope("state_is_tuple"):
                    lstm_cell = rnn_cell.LSTMCell(
                        num_units, None)
                    cell = rnn_cell.AttentionCell(
                        lstm_cell, attn_length, None)
                    zeros = tf.zeros(
                        [batch_size, num_units], dtype=np.float32)
                    attn_state_zeros = tf.zeros(
                        [batch_size, attn_length * num_units], dtype=np.float32)
                    zero_state = ((zeros, zeros), zeros, attn_state_zeros)
                    inputs = tf.zeros(
                        [batch_size, input_size], dtype=tf.float32)
                    output, state = cell(inputs, zero_state)
                    self.assertEquals(output.get_shape(), [
                                      batch_size, num_units])
                    self.assertEquals(len(state), 3)
                    self.assertEquals(len(state[0]), 2)
                    self.assertEquals(state[0][0].get_shape(),
                                      [batch_size, num_units])
                    self.assertEquals(state[0][1].get_shape(),
                                      [batch_size, num_units])
                    self.assertEquals(state[1].get_shape(), [
                                      batch_size, num_units])
                    self.assertEquals(state[2].get_shape(),
                                      [batch_size, attn_length * num_units])
                    tensors = [output] + list(state)
                    zero_result = sum(
                        [tf.reduce_sum(tf.abs(x)) for x in tensors])
                    sess.run(tf.global_variables_initializer())
                    self.assertTrue(sess.run(zero_result) < 1e-6)

    def testAttentionCellValues(self):
        num_units = 8
        attn_length = 16
        batch_size = 3
        with tf.Graph().as_default():
            with self.test_session() as sess:
                with tf.variable_scope("state_is_tuple"):
                    lstm_cell = rnn_cell.LSTMCell(
                        num_units, None)
                    cell = rnn_cell.AttentionCell(
                        lstm_cell, attn_length, None)
                    zeros = tf.constant(
                        0.1 * np.ones(
                            [batch_size, num_units], dtype=np.float32),
                        dtype=tf.float32)
                    attn_state_zeros = tf.constant(
                        0.1 * np.ones(
                            [batch_size, attn_length * num_units], dtype=np.float32),
                        dtype=tf.float32)
                    zero_state = ((zeros, zeros), zeros, attn_state_zeros)
                    inputs = tf.constant(
                        np.array(
                            [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
                            dtype=np.float32),
                        dtype=tf.float32)
                    output, state = cell(inputs, zero_state)
                    concat_state = tf.concat(
                        [state[0][0], state[0][1], state[1], state[2]], 1)
                    sess.run(tf.global_variables_initializer())
                    output, state = sess.run([output, concat_state])
                    for i in range(1, batch_size):
                        self.assertTrue(
                            float(np.linalg.norm((output[0, :] - output[i, :]))) > 1e-6)
                        self.assertTrue(
                            float(np.linalg.norm((state[0, :] - state[i, :]))) > 1e-6)

    def testMultiRNNCellWithStateTuple(self):
        with self.test_session() as sess:
            with tf.variable_scope(
                    "root", initializer=tf.constant_initializer(0.5)):
                x = tf.zeros([1, 2])
                m_bad = tf.zeros([1, 4])
                m_good = (tf.zeros([1, 2]), tf.zeros([1, 2]))

                # Test incorrectness of state
                with self.assertRaisesRegexp(ValueError, "Expected state .* a tuple"):
                    rnn_cell.MultiRNNCell(
                        [rnn_cell.GRUCell(2, None)
                         for _ in range(2)], state_is_tuple=True)(x, m_bad)

                _, ml = rnn_cell.MultiRNNCell(
                    [rnn_cell.GRUCell(2, None)
                     for _ in range(2)], state_is_tuple=True)(x, m_good)

                sess.run([tf.global_variables_initializer()])
                res = sess.run(ml, {
                    x.name: np.array([[1., 1.]]),
                    m_good[0].name: np.array([[0.1, 0.1]]),
                    m_good[1].name: np.array([[0.1, 0.1]])
                })

                # The numbers in results were not calculated, this is just a
                # smoke test.
                self.assertAllClose(res[0], [[-0.024449, -0.201085]])
                self.assertAllClose(res[1], [[-0.072805,  0.185238]])

    def testHighwayWrapper(self):
        with self.test_session() as sess:
            with tf.variable_scope(
                    "base_cell", initializer=tf.constant_initializer(0.5)):
                x = tf.zeros([1, 3])
                m = tf.zeros([1, 3])
                base_cell = rnn_cell.GRUCell(
                    3, None, w_init=tf.constant_initializer(0.5))
                g, m_new = base_cell(x, m)
            with tf.variable_scope(
                    "hw_cell", initializer=tf.constant_initializer(0.5)):
                hw_cell = rnn_cell.HighwayCell(
                    rnn_cell.GRUCell(3, None, w_init=tf.constant_initializer(0.5)), None, carry_bias_init=-100.0)
                g_res, m_new_res = hw_cell(x, m)
                sess.run([tf.global_variables_initializer()])
                res = sess.run([g, g_res, m_new, m_new_res], {
                    x: np.array([[1., 1., 1.]]),
                    m: np.array([[0.1, 0.1, 0.1]])
                })
            # As carry_bias_init is very negative, the carry gate is 'open' and the
            # transform gate is 'closed'. This means the output equals the input.
            self.assertAllClose(res[1], res[0])
            # States are left untouched
            self.assertAllClose(res[2], res[3])

    def testNASCell(self):
        num_units = 6
        batch_size = 3
        expected_output = np.array([[0.576751, 0.576751, 0.576751, 0.576751,
                                     0.576751, 0.576751],
                                    [0.618936, 0.618936, 0.618936, 0.618936,
                                     0.618936, 0.618936],
                                    [0.627393, 0.627393, 0.627393, 0.627393,
                                     0.627393, 0.627393]])
        expected_state = np.array([[0.71579772, 0.71579772, 0.71579772, 0.71579772,
                                    0.71579772, 0.71579772, 0.57675087, 0.57675087,
                                    0.57675087, 0.57675087, 0.57675087, 0.57675087],
                                   [0.78041625, 0.78041625, 0.78041625, 0.78041625,
                                    0.78041625, 0.78041625, 0.6189357, 0.6189357,
                                    0.61893570, 0.6189357, 0.6189357, 0.6189357],
                                   [0.79457647, 0.79457647, 0.79457647, 0.79457647,
                                    0.79457653, 0.79457653, 0.62739348, 0.62739348,
                                    0.62739348, 0.62739348, 0.62739348, 0.62739348]
                                   ])
        with self.test_session() as sess:
            with tf.variable_scope(
                "nas_test",
                    initializer=tf.constant_initializer(0.5)):
                cell = rnn_cell.NASCell(
                    num_units, None, w_init=tf.constant_initializer(0.5))
                inputs = tf.constant(
                    np.array([[1., 1., 1., 1.],
                              [2., 2., 2., 2.],
                              [3., 3., 3., 3.]],
                             dtype=np.float32),
                    dtype=tf.float32)
                state_value = tf.constant(
                    0.1 * np.ones(
                        (batch_size, num_units), dtype=np.float32),
                    dtype=tf.float32)
                init_state = rnn_cell.core_rnn_cell.LSTMStateTuple(
                    state_value, state_value)
                output, state = cell(inputs, init_state)
                sess.run([tf.global_variables_initializer()])
                res = sess.run([output, state])

                # This is a smoke test: Only making sure expected values not change.
                self.assertEqual(len(res), 2)
                self.assertAllClose(res[0], expected_output)
                # There should be 2 states in the tuple.
                self.assertEqual(len(res[1]), 2)
                # Checking the shape of each state to be batch_size * num_units
                new_c, new_h = res[1]
                self.assertEqual(new_c.shape[0], batch_size)
                self.assertEqual(new_c.shape[1], num_units)
                self.assertEqual(new_h.shape[0], batch_size)
                self.assertEqual(new_h.shape[1], num_units)
                self.assertAllClose(np.concatenate(
                    res[1], axis=1), expected_state)

    def testNASCellProj(self):
        num_units = 6
        batch_size = 3
        num_proj = 5
        expected_output = np.array([[1.697418, 1.697418, 1.697418, 1.697418,
                                     1.697418],
                                    [1.840037, 1.840037, 1.840037, 1.840037,
                                     1.840037],
                                    [1.873985, 1.873985, 1.873985, 1.873985,
                                     1.873985]])
        expected_state = np.array([[0.69855207, 0.69855207, 0.69855207, 0.69855207,
                                    0.69855207, 0.69855207, 1.69741797, 1.69741797,
                                    1.69741797, 1.69741797, 1.69741797],
                                   [0.77073824, 0.77073824, 0.77073824, 0.77073824,
                                    0.77073824, 0.77073824, 1.84003687, 1.84003687,
                                    1.84003687, 1.84003687, 1.84003687],
                                   [0.78973997, 0.78973997, 0.78973997, 0.78973997,
                                    0.78973997, 0.78973997, 1.87398517, 1.87398517,
                                    1.87398517, 1.87398517, 1.87398517]])
        with self.test_session() as sess:
            with tf.variable_scope(
                "nas_proj_test",
                    initializer=tf.constant_initializer(0.5)):
                cell = rnn_cell.NASCell(
                    num_units, None, w_init=tf.constant_initializer(0.5), num_proj=num_proj)
                inputs = tf.constant(
                    np.array([[1., 1., 1., 1.],
                              [2., 2., 2., 2.],
                              [3., 3., 3., 3.]],
                             dtype=np.float32),
                    dtype=tf.float32)
                state_value_c = tf.constant(
                    0.1 * np.ones(
                        (batch_size, num_units), dtype=np.float32),
                    dtype=tf.float32)
                state_value_h = tf.constant(
                    0.1 * np.ones(
                        (batch_size, num_proj), dtype=np.float32),
                    dtype=tf.float32)
                init_state = rnn_cell.core_rnn_cell.LSTMStateTuple(
                    state_value_c, state_value_h)
                output, state = cell(inputs, init_state)
                sess.run([tf.global_variables_initializer()])
                res = sess.run([output, state])

                # This is a smoke test: Only making sure expected values not change.
                self.assertEqual(len(res), 2)
                self.assertAllClose(res[0], expected_output)
                # There should be 2 states in the tuple.
                self.assertEqual(len(res[1]), 2)
                # Checking the shape of each state to be batch_size * num_units
                new_c, new_h = res[1]
                self.assertEqual(new_c.shape[0], batch_size)
                self.assertEqual(new_c.shape[1], num_units)
                self.assertEqual(new_h.shape[0], batch_size)
                self.assertEqual(new_h.shape[1], num_proj)
                self.assertAllClose(np.concatenate(
                    res[1], axis=1), expected_state)

    def testConv1DLSTMCell(self):
        with self.test_session() as sess:
            shape = [2, 1]
            filter_size = [3]
            num_features = 1
            batch_size = 2
            expected_state_c = np.array(
                [[[1.80168676], [1.80168676]], [[2.91189098], [2.91189098]]],
                dtype=np.float32)
            expected_state_h = np.array(
                [[[0.83409756], [0.83409756]], [[0.94695842], [0.94695842]]],
                dtype=np.float32)
            with tf.variable_scope(
                    "root", initializer=tf.constant_initializer(1.0 / 2.0)):
                x = tf.placeholder(tf.float32, [None, None, 1])
                cell = rnn_cell.Conv1DLSTMCell(input_shape=shape,
                                               kernel_shape=filter_size,
                                               output_channels=num_features, reuse=None, w_init=tf.constant_initializer(1.0 / 2.0))
                hidden = cell.zero_state(tf.shape(x)[0], tf.float32)
                output, state = cell(x, hidden)

                sess.run([tf.global_variables_initializer()])
                res = sess.run([output, state], {
                    hidden[0].name:
                        np.array([[[1.], [1.]],
                                  [[2.], [2.]]]),
                    x.name:
                        np.array([[[1.], [1.]],
                                  [[2.], [2.]]]),
                })
                # This is a smoke test, making sure expected values are unchanged.
                self.assertEqual(len(res), 2)
                self.assertAllClose(res[0], res[1].h)
                self.assertAllClose(res[1].c, expected_state_c)
                self.assertAllClose(res[1].h, expected_state_h)


if __name__ == '__main__':
    tf.test.main()
