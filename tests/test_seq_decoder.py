from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from tefla.core.decoder import AttentionLayerDot, AttentionLayerBahdanau


class AttentionLayerTest(tf.test.TestCase):
  """
    Tests the AttentionLayer module.
    """

  def setUp(self):
    super(AttentionLayerTest, self).setUp()
    self.batch_size = 8
    self.attention_dim = 128
    self.input_dim = 16
    self.seq_len = 10
    self.state_dim = 32

  def _create_layer(self):
    """Creates the attention layer. Should be implemented by child classes"""
    raise NotImplementedError

  def _test_layer(self):
    """Tests Attention layer with a  given score type"""
    inputs_pl = tf.placeholder(tf.float32, (None, None, self.input_dim))
    inputs_length_pl = tf.placeholder(tf.int32, [None])
    state_pl = tf.placeholder(tf.float32, (None, self.state_dim))
    attention_fn = self._create_layer()
    scores, context = attention_fn(
        query=state_pl, keys=inputs_pl, values=inputs_pl, values_length=inputs_length_pl)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {}
      feed_dict[inputs_pl] = np.random.randn(self.batch_size, self.seq_len, self.input_dim)
      feed_dict[state_pl] = np.random.randn(self.batch_size, self.state_dim)
      feed_dict[inputs_length_pl] = np.arange(self.batch_size) + 1
      scores_, context_ = sess.run([scores, context], feed_dict)

    self.assertAllEqual(scores_.shape, [self.batch_size, self.seq_len])
    self.assertAllEqual(context_.shape, [self.batch_size, self.input_dim])

    for idx, batch in enumerate(scores_, 1):
      # All scores that are padded should be zero
      np.testing.assert_array_equal(batch[idx:], np.zeros_like(batch[idx:]))

    # Scores should sum to 1
    scores_sum = np.sum(scores_, axis=1)
    self.assertNDArrayNear(scores_sum, np.ones([self.batch_size]), 0.00001)


class AttentionLayerDotTest(AttentionLayerTest):
  """Tests the AttentionLayerDot class"""

  def _create_layer(self):
    return AttentionLayerDot(
        params={"num_units": self.attention_dim}, mode=tf.contrib.learn.ModeKeys.TRAIN)

  def test_layer(self):
    self._test_layer()


class AttentionLayerBahdanauTest(AttentionLayerTest):
  """Tests the AttentionLayerBahdanau class"""

  def _create_layer(self):
    return AttentionLayerBahdanau(
        params={"num_units": self.attention_dim}, mode=tf.contrib.learn.ModeKeys.TRAIN)

  def test_layer(self):
    self._test_layer()


if __name__ == "__main__":
  tf.test.main()
