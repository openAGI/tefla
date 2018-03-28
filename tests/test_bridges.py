"""
Tests for Encoder-Decoder bridges.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest

from tefla.core.rnn_cell import MultiRNNCell, GRUCell, LSTMCell
from tefla.core.encoder import EncoderOutput
from tefla.core.bridges import ZeroBridge, InitialStateBridge
from tefla.core.bridges import PassThroughBridge

DecoderOutput = namedtuple("DecoderOutput", ["predicted_ids"])


class BridgeTest(tf.test.TestCase):
  """Abstract class for bridge tests"""

  def setUp(self):
    super(BridgeTest, self).setUp()
    self.batch_size = 4
    self.encoder_cell = MultiRNNCell([GRUCell(4, None), GRUCell(8, None)])
    self.decoder_cell = MultiRNNCell([LSTMCell(16, None), GRUCell(8, None)])
    final_encoder_state = nest.map_structure(
        lambda x: tf.convert_to_tensor(value=np.random.randn(self.batch_size, x), dtype=tf.float32),
        self.encoder_cell.state_size)
    self.encoder_outputs = EncoderOutput(
        outputs=tf.convert_to_tensor(
            value=np.random.randn(self.batch_size, 10, 16), dtype=tf.float32),
        attention_values=tf.convert_to_tensor(
            value=np.random.randn(self.batch_size, 10, 16), dtype=tf.float32),
        attention_values_length=np.full([self.batch_size], 10),
        final_state=final_encoder_state)

  def _create_bridge(self):
    """Creates the bridge class to be tests. Must be implemented by
        child classes"""
    raise NotImplementedError()

  def _assert_correct_outputs(self):
    """Asserts bridge outputs are correct. Must be implemented by
        child classes"""
    raise NotImplementedError()

  def _run(self, scope=None, **kwargs):
    """Runs the bridge with the given arguments
        """

    with tf.variable_scope(scope or "bridge"):
      bridge = self._create_bridge(**kwargs)
      initial_state = bridge()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      initial_state_ = sess.run(initial_state)

    return initial_state_


class TestZeroBridge(BridgeTest):
  """Tests for the ZeroBridge class"""

  def _create_bridge(self, **kwargs):
    return ZeroBridge(
        encoder_outputs=self.encoder_outputs,
        decoder_state_size=self.decoder_cell.state_size,
        params=kwargs,
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        reuse=None)

  def _assert_correct_outputs(self, initial_state_):
    initial_state_flat_ = nest.flatten(initial_state_)
    for element in initial_state_flat_:
      self.assertAllEqual(element, np.zeros_like(element))

  def test_zero_bridge(self):
    self._assert_correct_outputs(self._run())


class TestPassThroughBridge(BridgeTest):
  """Tests for the ZeroBridge class"""

  def _create_bridge(self, **kwargs):
    return PassThroughBridge(
        encoder_outputs=self.encoder_outputs,
        decoder_state_size=self.decoder_cell.state_size,
        params=kwargs,
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        reuse=None)

  def _assert_correct_outputs(self, initial_state_):
    nest.assert_same_structure(initial_state_, self.decoder_cell.state_size)
    nest.assert_same_structure(initial_state_, self.encoder_outputs.final_state)

    encoder_state_flat = nest.flatten(self.encoder_outputs.final_state)
    with self.test_session() as sess:
      encoder_state_flat_ = sess.run(encoder_state_flat)

    initial_state_flat_ = nest.flatten(initial_state_)
    for e_dec, e_enc in zip(initial_state_flat_, encoder_state_flat_):
      self.assertAllEqual(e_dec, e_enc)

  def test_passthrough_bridge(self):
    self.decoder_cell = self.encoder_cell
    self._assert_correct_outputs(self._run())


class TestInitialStateBridge(BridgeTest):
  """Tests for the InitialStateBridge class"""

  def _create_bridge(self, **kwargs):
    return InitialStateBridge(
        encoder_outputs=self.encoder_outputs,
        decoder_state_size=self.decoder_cell.state_size,
        params=kwargs,
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        reuse=None)

  def _assert_correct_outputs(self, initial_state_):
    nest.assert_same_structure(initial_state_, self.decoder_cell.state_size)

  def test_with_final_state(self):
    self._assert_correct_outputs(self._run(bridge_input="final_state"))

  def test_with_outputs(self):
    self._assert_correct_outputs(self._run(bridge_input="outputs"))

  def test_with_activation_fn(self):
    self._assert_correct_outputs(self._run(bridge_input="final_state", activation_fn="tanh"))


if __name__ == "__main__":
  tf.test.main()
