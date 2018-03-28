from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from tefla.core import encoder


class UnidirectionalRNNEncoderTest(tf.test.TestCase):
  """
    Tests the UnidirectionalRNNEncoder class.
    """

  def setUp(self):
    super(UnidirectionalRNNEncoderTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.mode = tf.contrib.learn.ModeKeys.TRAIN
    self.params = encoder.UnidirectionalRNNEncoder.default_params()
    self.params["rnn_cell"]["cell_params"]["num_units"] = 32

  def test_encode(self):
    inputs = tf.random_normal([self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = encoder.UnidirectionalRNNEncoder(self.params, self.mode)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    self.assertAllEqual(encoder_output_.outputs.shape, [self.batch_size, self.sequence_length, 32])
    self.assertIsInstance(encoder_output_.final_state, tf.contrib.rnn.LSTMStateTuple)
    self.assertAllEqual(encoder_output_.final_state.h.shape, [self.batch_size, 32])
    self.assertAllEqual(encoder_output_.final_state.c.shape, [self.batch_size, 32])


class BidirectionalRNNEncoderTest(tf.test.TestCase):
  """
    Tests the BidirectionalRNNEncoder class.
    """

  def setUp(self):
    super(BidirectionalRNNEncoderTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.params = encoder.BidirectionalRNNEncoder.default_params()
    self.params["rnn_cell"]["cell_params"]["num_units"] = 32
    self.mode = tf.contrib.learn.ModeKeys.TRAIN

  def test_encode(self):
    inputs = tf.random_normal([self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = encoder.BidirectionalRNNEncoder(self.params, self.mode)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    self.assertAllEqual(encoder_output_.outputs.shape,
                        [self.batch_size, self.sequence_length, 32 * 2])

    self.assertIsInstance(encoder_output_.final_state[0], tf.contrib.rnn.LSTMStateTuple)
    self.assertIsInstance(encoder_output_.final_state[1], tf.contrib.rnn.LSTMStateTuple)
    self.assertAllEqual(encoder_output_.final_state[0].h.shape, [self.batch_size, 32])
    self.assertAllEqual(encoder_output_.final_state[0].c.shape, [self.batch_size, 32])
    self.assertAllEqual(encoder_output_.final_state[1].h.shape, [self.batch_size, 32])
    self.assertAllEqual(encoder_output_.final_state[1].c.shape, [self.batch_size, 32])


class StackBidirectionalRNNEncoderTest(tf.test.TestCase):
  """
    Tests the StackBidirectionalRNNEncoder class.
    """

  def setUp(self):
    super(StackBidirectionalRNNEncoderTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.mode = tf.contrib.learn.ModeKeys.TRAIN

  def _test_encode_with_params(self, params):
    """Tests the StackBidirectionalRNNEncoder with a specific cell"""
    inputs = tf.random_normal([self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = encoder.StackBidirectionalRNNEncoder(params, self.mode)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    output_size = encode_fn.params["rnn_cell"]["cell_params"]["num_units"]

    self.assertAllEqual(encoder_output_.outputs.shape,
                        [self.batch_size, self.sequence_length, output_size * 2])

    return encoder_output_

  def test_encode_with_single_cell(self):
    encoder_output_ = self._test_encode_with_params({
        "rnn_cell": {
            "num_layers": 1,
            "cell_params": {
                "num_units": 32
            }
        }
    })

    self.assertIsInstance(encoder_output_.final_state[0][0], tf.contrib.rnn.LSTMStateTuple)
    self.assertIsInstance(encoder_output_.final_state[1][0], tf.contrib.rnn.LSTMStateTuple)
    self.assertAllEqual(encoder_output_.final_state[0][0].h.shape, [self.batch_size, 32])
    self.assertAllEqual(encoder_output_.final_state[0][0].c.shape, [self.batch_size, 32])
    self.assertAllEqual(encoder_output_.final_state[1][0].h.shape, [self.batch_size, 32])
    self.assertAllEqual(encoder_output_.final_state[1][0].c.shape, [self.batch_size, 32])

  def test_encode_with_multi_cell(self):
    encoder_output_ = self._test_encode_with_params({
        "rnn_cell": {
            "num_layers": 4,
            "cell_params": {
                "num_units": 32
            }
        }
    })

    for layer_idx in range(4):
      self.assertIsInstance(encoder_output_.final_state[0][layer_idx],
                            tf.contrib.rnn.LSTMStateTuple)
      self.assertIsInstance(encoder_output_.final_state[1][layer_idx],
                            tf.contrib.rnn.LSTMStateTuple)
      self.assertAllEqual(encoder_output_.final_state[0][layer_idx].h.shape, [self.batch_size, 32])
      self.assertAllEqual(encoder_output_.final_state[0][layer_idx].c.shape, [self.batch_size, 32])
      self.assertAllEqual(encoder_output_.final_state[1][layer_idx].h.shape, [self.batch_size, 32])
      self.assertAllEqual(encoder_output_.final_state[1][layer_idx].c.shape, [self.batch_size, 32])


class PoolingEncoderTest(tf.test.TestCase):
  """
    Tests the PoolingEncoder class.
    """

  def setUp(self):
    super(PoolingEncoderTest, self).setUp()
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.mode = tf.contrib.learn.ModeKeys.TRAIN

  def _test_with_params(self, params):
    """Tests the encoder with a given parameter configuration"""
    inputs = tf.random_normal([self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = encoder.PoolingEncoder(params, self.mode, None)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    self.assertAllEqual(encoder_output_.outputs.shape,
                        [self.batch_size, self.sequence_length, self.input_depth])
    self.assertAllEqual(encoder_output_.attention_values.shape,
                        [self.batch_size, self.sequence_length, self.input_depth])
    self.assertAllEqual(encoder_output_.final_state.shape, [self.batch_size, self.input_depth])

  def test_encode_with_pos(self):
    self._test_with_params({
        "position_embeddings.enable": True,
        "position_embeddings.num_positions": self.sequence_length
    })

  def test_encode_without_pos(self):
    self._test_with_params({
        "position_embeddings.enable": False,
        "position_embeddings.num_positions": 0
    })


class ConvEncoderTest(tf.test.TestCase):
  """
    Tests the ConvEncoder class.
    """

  def setUp(self):
    super(ConvEncoderTest, self).setUp()
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.mode = tf.contrib.learn.ModeKeys.TRAIN

  def _test_with_params(self, params):
    """Tests the encoder with a given parameter configuration"""
    inputs = tf.random_normal([self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = encoder.ConvEncoder(params, self.mode, None)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    att_value_units = encode_fn.params["attention_cnn.units"]
    output_units = encode_fn.params["output_cnn.units"]

    np.testing.assert_array_equal(encoder_output_.outputs.shape,
                                  [self.batch_size, self.sequence_length, att_value_units])
    np.testing.assert_array_equal(encoder_output_.attention_values.shape,
                                  [self.batch_size, self.sequence_length, output_units])
    np.testing.assert_array_equal(encoder_output_.final_state.shape,
                                  [self.batch_size, output_units])

  def test_encode_with_pos(self):
    self._test_with_params({
        "position_embeddings.enable": True,
        "position_embeddings.num_positions": self.sequence_length,
        "attention_cnn.units": 5,
        "output_cnn.units": 6
    })


if __name__ == "__main__":
  tf.test.main()
