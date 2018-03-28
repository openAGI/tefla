"""
Tests for Beam Search and related functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from tefla.core import beam_search


class TestGatherTree(tf.test.TestCase):
  """Tests the gather_tree function"""

  def test_gather_tree(self):
    predicted_ids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    parent_ids = np.array([[0, 0, 0], [0, 1, 1], [2, 1, 2]])
    expected_result = np.array([[2, 2, 2], [6, 5, 6], [7, 8, 9]])

    res = beam_search.gather_tree(
        tf.convert_to_tensor(predicted_ids), tf.convert_to_tensor(parent_ids))
    with self.test_session() as sess:
      res_ = sess.run(res)

    self.assertNDArrayNear(expected_result, res_, 0.0001)


class TestLengthNorm(tf.test.TestCase):
  """Tests the length normalization score"""

  def test_length_norm(self):
    lengths_ = np.array([[1, 2, 3], [3, 3, 3]])
    penalty_factor_ = 0.6
    length_pen = beam_search.length_penalty(
        sequence_lengths=tf.convert_to_tensor(lengths_), penalty_factor=penalty_factor_)

    with self.test_session() as sess:
      length_pen_ = sess.run(length_pen)

    self.assertNear(length_pen_[0, 0], 1.0, 0.00001)
    self.assertNear(length_pen_[0, 1], 1.0969027, 0.00001)
    self.assertNear(length_pen_[0, 2], 1.1884017, 0.00001)


class TestBeamStep(tf.test.TestCase):
  """Tests a single step of beam search
    """

  def setUp(self):
    super(TestBeamStep, self).setUp()
    self.state_size = 10
    config = beam_search.BeamSearchConfig(
        beam_width=3,
        vocab_size=5,
        eos_token=0,
        length_penalty_weight=0.6,
        choose_successors_fn=beam_search.choose_top_k)
    self.config = config

  def test_step(self):
    beam_state = beam_search.BeamSearchState(
        log_probs=tf.nn.log_softmax(tf.ones(self.config.beam_width)),
        lengths=tf.constant(2, shape=[self.config.beam_width], dtype=tf.int32),
        finished=tf.zeros([self.config.beam_width], dtype=tf.bool))

    logits_ = np.full([self.config.beam_width, self.config.vocab_size], 0.0001)
    logits_[0, 2] = 1.9
    logits_[0, 3] = 2.1
    logits_[1, 3] = 3.1
    logits_[1, 4] = 0.9
    logits = tf.convert_to_tensor(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search.beam_search_step(
        time_=2, logits=logits, beam_state=beam_state, config=self.config)

    with self.test_session() as sess:
      outputs_, next_state_, state_, log_probs_ = sess.run(
          [outputs, next_beam_state, beam_state, log_probs])

    self.assertNDArrayNear(outputs_.predicted_ids, [3, 3, 2], 0.00001)
    self.assertNDArrayNear(outputs_.beam_parent_ids, [1, 0, 0], 0.00001)
    self.assertNDArrayNear(next_state_.lengths, [3, 3, 3], 0.00001)
    self.assertAllEqual(next_state_.finished, [False, False, False])

    expected_log_probs = state_.log_probs[[1, 0, 0]]
    expected_log_probs[0] += log_probs_[1, 3]
    expected_log_probs[1] += log_probs_[0, 3]
    expected_log_probs[2] += log_probs_[0, 2]
    self.assertNDArrayNear(next_state_.log_probs, expected_log_probs, 0.00001)

  def test_step_with_eos(self):
    beam_state = beam_search.BeamSearchState(
        log_probs=tf.nn.log_softmax(tf.ones(self.config.beam_width)),
        lengths=tf.convert_to_tensor([2, 1, 2], dtype=tf.int32),
        finished=tf.constant([False, True, False], dtype=tf.bool))

    logits_ = np.full([self.config.beam_width, self.config.vocab_size], 0.0001)
    logits_[0, 2] = 1.1
    logits_[1, 2] = 1.0
    logits_[2, 2] = 1.0
    logits = tf.convert_to_tensor(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search.beam_search_step(
        time_=2, logits=logits, beam_state=beam_state, config=self.config)

    with self.test_session() as sess:
      outputs_, next_state_, state_, log_probs_ = sess.run(
          [outputs, next_beam_state, beam_state, log_probs])

    self.assertNDArrayNear(outputs_.predicted_ids, [0, 2, 2], 0.00001)
    self.assertNDArrayNear(outputs_.beam_parent_ids, [1, 0, 2], 0.00001)
    self.assertNDArrayNear(next_state_.lengths, [1, 3, 3], 0.00001)
    self.assertAllEqual(next_state_.finished, [True, False, False])

    expected_log_probs = state_.log_probs[outputs_.beam_parent_ids]
    expected_log_probs[1] += log_probs_[0, 2]
    expected_log_probs[2] += log_probs_[2, 2]
    self.assertNDArrayNear(next_state_.log_probs, expected_log_probs, 0.00001)

  def test_step_with_new_eos(self):
    beam_state = beam_search.BeamSearchState(
        log_probs=tf.nn.log_softmax(tf.ones(self.config.beam_width)),
        lengths=tf.constant(2, shape=[self.config.beam_width], dtype=tf.int32),
        finished=tf.zeros([self.config.beam_width], dtype=tf.bool))

    logits_ = np.full([self.config.beam_width, self.config.vocab_size], 0.0001)
    logits_[0, 0] = 1.9
    logits_[0, 3] = 2.1
    logits_[1, 3] = 3.1
    logits_[1, 4] = 0.9
    logits = tf.convert_to_tensor(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search.beam_search_step(
        time_=2, logits=logits, beam_state=beam_state, config=self.config)

    with self.test_session() as sess:
      outputs_, next_state_, state_, log_probs_ = sess.run(
          [outputs, next_beam_state, beam_state, log_probs])

    self.assertNDArrayNear(outputs_.predicted_ids, [3, 3, 0], 0.00001)
    self.assertNDArrayNear(outputs_.beam_parent_ids, [1, 0, 0], 0.00001)
    self.assertNDArrayNear(next_state_.lengths, [3, 3, 2], 0.00001)
    self.assertAllEqual(next_state_.finished, [False, False, True])


class TestEosMasking(tf.test.TestCase):
  """Tests EOS masking used in beam search
    """

  def test_eos_masking(self):
    probs = tf.constant([[-.2, -.2, -.2, -.2, -.2], [-.3, -.3, -.3, 3, 0], [5, 6, 0, 0, 0]])
    eos_token = 0
    previously_finished = tf.constant([0, 1, 0], dtype=tf.float32)
    masked = beam_search.mask_probs(probs, eos_token, previously_finished)

    with self.test_session() as sess:
      probs = sess.run(probs)
      masked = sess.run(masked)

      self.assertNDArrayNear(probs[0], masked[0], 0.00001)
      self.assertNDArrayNear(probs[2], masked[2], 0.00001)
      self.assertEqual(masked[1][0], 0)
      self.assertNear(masked[1][1], np.finfo('float32').min, 0.00001)
      self.assertNear(masked[1][2], np.finfo('float32').min, 0.00001)
      self.assertNear(masked[1][3], np.finfo('float32').min, 0.00001)
      self.assertNear(masked[1][4], np.finfo('float32').min, 0.00001)


if __name__ == "__main__":
  tf.test.main()
