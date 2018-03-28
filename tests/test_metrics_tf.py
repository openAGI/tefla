# -*- coding: utf-8 -*-
"""Tests for Metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from tefla.core.metrics import BleuMetricSpec, RougeMetricSpec, moses_multi_bleu, rouge


class TestMosesBleu(tf.test.TestCase):
  """Tests using the Moses multi-bleu script to calcualte BLEU score
    """

  def _test_multi_bleu(self, hypotheses, references, lowercase, expected_bleu):
    """Runs a multi-bleu test."""
    result = moses_multi_bleu(hypotheses=hypotheses, references=references, lowercase=lowercase)
    self.assertNDArrayNear(result, expected_bleu, 0.01)

  def test_multi_bleu(self):
    self._test_multi_bleu(
        hypotheses=np.array(
            ["The brown fox jumps over the dog 笑", "The brown fox jumps over the dog 2 笑"]),
        references=np.array([
            "The quick brown fox jumps over the lazy dog 笑",
            "The quick brown fox jumps over the lazy dog 笑"
        ]),
        lowercase=False,
        expected_bleu=46.51)

  def test_empty(self):
    self._test_multi_bleu(
        hypotheses=np.array([]), references=np.array([]), lowercase=False, expected_bleu=0.00)

  def test_multi_bleu_lowercase(self):
    self._test_multi_bleu(
        hypotheses=np.array(
            ["The brown fox jumps over The Dog 笑", "The brown fox jumps over The Dog 2 笑"]),
        references=np.array([
            "The quick brown fox jumps over the lazy dog 笑",
            "The quick brown fox jumps over the lazy dog 笑"
        ]),
        lowercase=True,
        expected_bleu=46.51)


class TestTextMetricSpec(tf.test.TestCase):
  """Abstract class for testing TextMetricSpecs
    based on hypotheses and references"""

  def _test_metric_spec(self, metric_spec, hyps, refs, expected_scores):
    """Tests a MetricSpec"""
    predictions = {"predicted_tokens": tf.placeholder(dtype=tf.string)}
    labels = {"target_tokens": tf.placeholder(dtype=tf.string)}

    value, update_op = metric_spec.create_metric_ops(None, labels, predictions)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      scores = []
      for hyp, ref in zip(hyps, refs):
        hyp = hyp.split(" ")
        ref = ref.split(" ")
        sess.run(update_op, {
            predictions["predicted_tokens"]: [hyp],
            labels["target_tokens"]: [ref]
        })
        scores.append(sess.run(value))

      for score, expected in zip(scores, expected_scores):
        self.assertNDArrayNear(score, expected, 0.01)
        self.assertNDArrayNear(score, expected, 0.01)


class TestBleuMetricSpec(TestTextMetricSpec):
  """Tests the `BleuMetricSpec`"""

  def test_bleu(self):
    metric_spec = BleuMetricSpec({})
    return self._test_metric_spec(
        metric_spec=metric_spec,
        hyps=["A B C D E F", "A B C D E F"],
        refs=["A B C D E F", "A B A D E F"],
        expected_scores=[100.0, 69.19])


class TestRougeMetricSpec(TestTextMetricSpec):
  """Tests the `RougeMetricSpec`"""

  def test_rouge_1_f_score(self):
    metric_spec = RougeMetricSpec({"rouge_type": "rouge_1/f_score"})
    self._test_metric_spec(
        metric_spec=metric_spec,
        hyps=["A B C D E F", "A B C D E F"],
        refs=["A B C D E F", "A B A D E F"],
        expected_scores=[1.0, 0.954])

    self._test_metric_spec(metric_spec=metric_spec, hyps=[], refs=[], expected_scores=[0.0])

    self._test_metric_spec(metric_spec=metric_spec, hyps=["A"], refs=["B"], expected_scores=[0.0])

  def test_rouge_2_f_score(self):
    metric_spec = RougeMetricSpec({"rouge_type": "rouge_2/f_score"})
    self._test_metric_spec(
        metric_spec=metric_spec,
        hyps=["A B C D E F", "A B C D E F"],
        refs=["A B C D E F", "A B A D E F"],
        expected_scores=[1.0, 0.8])

    self._test_metric_spec(metric_spec=metric_spec, hyps=[], refs=[], expected_scores=[0.0])

    self._test_metric_spec(metric_spec=metric_spec, hyps=["A"], refs=["B"], expected_scores=[0.0])

  def test_rouge_l_f_score(self):
    metric_spec = RougeMetricSpec({"rouge_type": "rouge_l/f_score"})

    self._test_metric_spec(
        metric_spec=metric_spec,
        hyps=["A B C D E F", "A B C D E F"],
        refs=["A B C D E F", "A B A D E F"],
        expected_scores=[1.0, 0.916])

    self._test_metric_spec(metric_spec=metric_spec, hyps=[], refs=[], expected_scores=[0.0])

    self._test_metric_spec(metric_spec=metric_spec, hyps=["A"], refs=["B"], expected_scores=[0.0])


class TestRougeMetric(tf.test.TestCase):
  """Tests the RougeMetric"""

  def test_rouge(self):
    hypotheses = np.array(
        ["The brown fox jumps over the dog 笑", "The brown fox jumps over the dog 2 笑"])
    references = np.array([
        "The quick brown fox jumps over the lazy dog 笑",
        "The quick brown fox jumps over the lazy dog 笑"
    ])
    output = rouge(hypotheses, references)
    self.assertNDArrayNear(output["rouge_1/f_score"], 0.865, 0.01)
    self.assertNDArrayNear(output["rouge_2/f_score"], 0.548, 0.01)
    self.assertNDArrayNear(output["rouge_l/f_score"], 0.852, 0.01)


if __name__ == "__main__":
  tf.test.main()
