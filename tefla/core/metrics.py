# -*- coding: utf-8 -*-
# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six
import tensorflow as tf
import numpy as np
import os
import re
import subprocess
import tempfile
import itertools

from six.moves import urllib
from pydoc import locate
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from tensorflow.contrib import metrics
from tensorflow.contrib.learn import MetricSpec
from ..convert import convert
from ..convert_labels import convert_labels
from .encoder import Configurable
from ..utils import postproc


@six.add_metaclass(abc.ABCMeta)
class Metric(object):

  def __init__(self, name=None):
    self.name = name

  @abc.abstractmethod
  def metric(self, predictions, targets, **kwargs):
    raise NotImplementedError


class MetricMixin(object):

  def __init__(self, name='metric_hist_confusion'):
    self.name = name
    super(MetricMixin, self).__init__(name)

  def confusion_matrix(self, rater_a, rater_b, min_rating=None, max_rating=None):
    """Returns the confusion matrix between rater's ratings."""
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
      min_rating = min(rater_a + rater_b)
    if max_rating is None:
      max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
      conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

  def histogram(self, ratings, min_rating=None, max_rating=None):
    """Returns the counts of each type of rating that a rater made."""
    if min_rating is None:
      min_rating = min(ratings)
    if max_rating is None:
      max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
      hist_ratings[r - min_rating] += 1
    return hist_ratings


class Top_k(Metric):
  """Class to compute Top_k accuracy metric for predictions and labels."""

  def __init__(self, name='Top_K'):
    super(Top_k, self).__init__(name)
    self.name = name

  def metric(self, predictions, targets, top_k=1):
    """Computes top k metric.

    Args:
        predictions: 2D tensor/array, predictions of the network
        targets: 2D tensor/array, ground truth labels of the network
        top_k: int, returns the top_k accuracy; {1,2,3 max_classes}

    Returns:
        top_k accuracy
    """
    return self._top_k_op(predictions, targets, top_k=top_k)

  def _top_k_op(self, predictions, targets, top_k=1, dtype=tf.int32):
    with tf.name_scope('Top_' + str(top_k)):
      targets = tf.cast(targets, dtype)
      correct_pred = tf.nn.in_top_k(predictions, tf.argmax(targets, 1), top_k)
      acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc


class IOU(Metric, MetricMixin):
  """Class to compute IOU metric for predictions and labels."""

  def __init__(self, name='IOU'):
    super(IOU, self).__init__(name)
    self.name = name

  def metric(self, predictions, targets, top_k=1):
    """Computes top k metric.

    Args:
        predictions: 2D tensor/array, predictions of the network
        targets: 2D tensor/array, ground truth labels of the network
        top_k: int, returns the top_k accuracy; {1,2,3 max_classes}

    Returns:
        top_k accuracy
    """
    return self._iou_op(predictions, targets)

  def _iou_op(self, predictions, targets, top_k=1, dtype=tf.int32):
    with tf.name_scope('iou'):
      targets = tf.cast(targets, dtype)
      conf_mat = self.confusion_matrix(predictions, targets)
      t = []
      k = len(conf_mat[0])
      for i in range(k):
        t.append(sum([conf_mat[i][j] for j in range(k)]))
      return (1.0 / k) * sum([
          float(conf_mat[i][i]) / (t[i] - conf_mat[i][i] + sum([conf_mat[j][i] for j in range(k)]))
          for i in range(k)
      ])


class IOUSeg(object):

  def __init__(self, name='IOU'):
    self.name = name
    super(IOUSeg, self).__init__()

  def meaniou(self, predictor, predict_dir, image_size):
    segparams = util.SegParams()
    classes = segparams.feature_classes().values()
    num_classes = len(classes) + 1
    hist = np.zeros((num_classes, num_classes))
    image_names = [
        filename.strip() for filename in os.listdir(predict_dir) if filename.endswith('.jpg')
    ]
    for image_filename in image_names:
      final_prediction_map = predictor.predict(os.path.join(predict_dir, image_filename))
      final_prediction_map = final_prediction_map.transpose(0, 2, 1).squeeze()
      gt_name = os.path.join(predict_dir, image_filename[:-4] + '_final_mask' + '.png')
      gt = convert(gt_name, image_size)
      gt = np.asarray(gt)
      gt = convert_labels(gt, image_size, image_size)
      hist += compute_hist(gt, final_prediction_map, num_classes=num_classes)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    meaniou = np.nanmean(iou)

    return meaniou

  def per_class_iou(self, predictor, predict_dir, image_size):
    image_names = [
        filename.strip() for filename in os.listdir(predict_dir) if filename.endswith('.jpg')
    ]
    per_class_iou_hist = defaultdict(np.ndarray)
    per_class_iou_dict = defaultdict(float)
    segparams = util.SegParams()
    classes = segparams.feature_classes().values()
    num_classes = len(classes) + 1
    for class_id, class_name in enumerate(classes, 1):
      per_class_iou_hist[class_name] = np.zeros((num_classes, num_classes))

    for image_filename in image_names:
      final_prediction_map = predictor.predict(os.path.join(predict_dir, image_filename))
      final_prediction_map = final_prediction_map.transpose(0, 2, 1).squeeze()
      gt_name = os.path.join(predict_dir, image_filename[:-4] + '_final_mask' + '.png')
      gt = convert(gt_name, image_size)
      gt = np.asarray(gt)
      gt = convert_labels(gt, image_size, image_size)
      for class_id, class_name in enumerate(classes, 1):
        per_class_iou_hist[class_name] += compute_hist(
            np.asarray(gt == class_id, dtype=np.int32),
            np.asarray(final_prediction_map == class_id, dtype=np.int32),
            num_classes=num_classes)

    for class_id, class_name in enumerate(classes, 1):
      per_class_iou_dict[class_name] = np.nanmean(
          np.diag(per_class_iou_hist[class_name]) /
          (per_class_iou_hist[class_name].sum(1) + per_class_iou_hist[class_name].sum(0) -
           np.diag(per_class_iou_hist[class_name])))
    return per_class_iou_dict


class Kappa(Metric, MetricMixin):

  def __init__(self, name='kappa'):
    super(Kappa, self).__init__(name)

  def metric(self, predictions, targets, num_classes):
    """Computes Kappa metric.

    Args:
        predictions: 2D tensor/array, predictions of the network
        targets: 2D tensor/array, ground truth labels of the network

    Returns:
        Kappa score
    """
    targets = np.array(targets)
    predictions = np.array(predictions)
    if len(targets.shape) > 1 and targets.shape[1] > 1:
      targets = targets.dot(range(targets.shape[1]))
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
      predictions = np.argmax(predictions, axis=1)
    try:
      return self._quadratic_weighted_kappa(predictions, targets, max_rating=num_classes - 1)
    except IndexError:
      return np.nan

  def _quadratic_weighted_kappa(self, rater_a, rater_b, min_rating=0, max_rating=4):
    """Calculates the quadratic weighted kappa quadratic_weighted_kappa
    calculates the quadratic weighted kappa value, which is a measure of inter-
    rater agreement between two raters that provide discrete numeric ratings.
    Potential values range from -1 (representing complete disagreement) to 1
    (representing complete agreement).  A kappa value of 0 is expected if all
    agreement is due to chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.clip(rater_a, min_rating, max_rating)
    rater_b = np.clip(rater_b, min_rating, max_rating)

    rater_a = np.round(rater_a).astype(int).ravel()
    rater_a[~np.isfinite(rater_a)] = 0
    rater_b = np.round(rater_b).astype(int).ravel()
    rater_b[~np.isfinite(rater_b)] = 0

    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
      min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
      max_rating = max(max(rater_a), max(rater_b))
    conf_mat = self.confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = self.histogram(rater_a, min_rating, max_rating)
    hist_rater_b = self.histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
      for j in range(num_ratings):
        expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
        d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
        numerator += d * conf_mat[i][j] / num_scored_items
        denominator += d * expected_count / num_scored_items
    try:
      return 1.0 - numerator / denominator
    except ZeroDivisionError:
      return 0.0001


class KappaV2(Metric, MetricMixin):

  def __init__(self, name='kappa', num_classes=10, batch_size=32):
    self.num_classes = num_classes
    self.batch_size = batch_size
    super(KappaV2, self).__init__(name)

  def metric(self, predictions, targets, num_classes=None, batch_size=None, **kwargs):
    """Computes Kappa metric.

    Args:
        predictions: 2D tensor/array, predictions of the network
        targets: 2D tensor/array, ground truth labels of the network
        num_classes: int, num_classes of the network
        batch_size: batch_size for predictions of the network

    Returns:
        Kappa score
    """
    if num_classes is None:
      num_classes = self.num_classes
    if batch_size is None:
      batch_size = self.batch_size
    targets = tf.convert_to_tensor(targets)
    predictions = tf.convert_to_tensor(predictions)
    if targets.get_shape().ndims == 1:
      targets = tf.one_hot(targets, num_classes, on_value=1, off_value=0)
    if predictions.get_shape().ndims == 1:
      predictions = tf.one_hot(predictions, num_classes, on_value=1, off_value=0)
    return self._kappa_loss(
        predictions, targets, batch_size=batch_size, num_ratings=num_classes, **kwargs)

  def _kappa_loss(self,
                  predictions,
                  labels,
                  y_pow=1,
                  eps=1e-15,
                  num_ratings=5,
                  batch_size=32,
                  name='kappa'):
    with tf.name_scope(name):
      labels = tf.to_float(labels)
      predictions = tf.to_float(predictions)
      repeat_op = tf.to_float(
          tf.tile(tf.reshape(tf.range(0, num_ratings), [num_ratings, 1]), [1, num_ratings]))
      repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
      weights = repeat_op_sq / tf.to_float((num_ratings - 1)**2)

      pred_ = predictions**y_pow
      try:
        pred_norm = pred_ / \
            (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
      except Exception as e:
        print(e.message)
        pred_norm = pred_ / \
            (eps + tf.reshape(tf.reduce_sum(pred_, 1),
                              [batch_size, 1]))

      hist_rater_a = tf.reduce_sum(pred_norm, 0)
      hist_rater_b = tf.reduce_sum(labels, 0)

      conf_mat = tf.matmul(tf.transpose(pred_norm), labels)

      nom = tf.reduce_sum(weights * conf_mat)
      denom = tf.reduce_sum(weights * tf.matmul(
          tf.reshape(hist_rater_a, [num_ratings, 1]), tf.reshape(hist_rater_b, [1, num_ratings])) /
          tf.to_float(batch_size))

      try:
        return (1 - nom / denom)
      except Exception as e:
        print(e.message)
        return (1 - nom / (denom + eps))


class Auroc(Metric):

  def __init__(self, name='auroc'):
    super(Auroc, self).__init__(name)
    self.name = name

  def metric(self, predictions, targets, num_classes=5):
    """Computes auroc metric.

    Args:
        predictions: 2D tensor/array, predictions of the network
        targets: 2D tensor/array, ground truth labels of the network
        num_classes: int, num_classes of the network

    Returns:
        auroc score
    """
    if targets.ndim == 2:
      targets = np.argmax(targets, axis=1)
    if predictions.ndim == 1:
      predictions = one_hot(predictions, m=num_classes)
    return self._auroc(predictions, targets)

  def _auroc(self, y_pred, y_true):
    try:
      return roc_auc_score(y_true, y_pred[:, 1])
    except ValueError as e:
      print(e)
      return accuracy_score(y_true, np.argmax(y_pred, axis=1))


class F1score(Metric):

  def __init__(self, name='auroc'):
    super(F1score, self).__init__(name)
    self.name = name

  def metric(self, predictions, targets, num_classes=5):
    """Computes F1 metric.

    Args:
        predictions: 2D tensor/array, predictions of the network
        targets: 2D tensor/array, ground truth labels of the network
        num_classes: int, num_classes of the network

    Returns:
        F1 score
    """
    if targets.ndim == 2:
      targets = np.argmax(targets, axis=1)
    if predictions.ndim == 1:
      predictions = one_hot(predictions, m=num_classes)
    return self._f1_score(predictions, targets)

  def _f1_score(self, y_pred, y_true):
    y_pred_2 = np.argmax(y_pred, axis=1)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred_2)
    return accuracy_score(y_true, y_pred_2) if 0 in f1 else np.mean(f1)


def accuracy_op(predictions, targets, num_classes=5):
  """Computes accuracy metric.

  Args:
      predictions: 2D tensor/array, predictions of the network
      targets: 2D tensor/array, ground truth labels of the network
      num_classes: int, num_classes of the network

  Returns:
      accuracy
  """
  with tf.name_scope('Accuracy'):
    if targets.ndim == 2:
      targets = np.argmax(targets, axis=1)
    if predictions.ndim == 1:
      predictions = one_hot(predictions, m=num_classes)
    acc = accuracy_score(targets, np.argmax(predictions, axis=1))
  return acc


def one_hot(vec, m=None):
  """Retruns one hot vector.

  Args:
      vec: a vector
      m: num_classes
  """
  if m is None:
    m = int(np.max(vec)) + 1
  return np.eye(m)[vec].astype('int32')


def fast_hist(a, b, n):
  k = (a >= 0) & (a < n)
  return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(preds, gt, num_classes=15):
  hist = np.zeros((num_classes, num_classes))
  hist += fast_hist(np.reshape(gt, (-1)), np.reshape(preds, (-1)), num_classes)
  return hist


def dice_coef(y_true, y_pred):
  """Compute dice coef.

  Args:
      y_true: a 2-D `array`, ground truth label
      y_pred: q 2-D `array`, prediction

  Returns:
      a `float`, dice value
  """
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  intersection = np.sum(y_true_f * y_pred_f)
  return (2. * intersection + 100) / (np.sum(y_true_f) + np.sum(y_pred_f) + 100)


def char_accuracy(predictions, targets, rej_char, streaming=False):
  """Computes character level accuracy. Both predictions and targets should
  have the same shape.

  [batch_size x seq_length].

  Args:
      predictions: predicted characters ids.
      targets: ground truth character ids.
      rej_char: the character id used to mark an empty element (end of sequence).
      streaming: if True, uses the streaming mean from the slim.metric module.

  Returns:
      a update_ops for execution and value tensor whose value on evaluation
          returns the total character accuracy.
  """
  with tf.variable_scope('CharAccuracy'):
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())

    targets = tf.to_int32(targets)
    const_rej_char = tf.constant(rej_char, shape=targets.get_shape())
    weights = tf.to_float(tf.not_equal(targets, const_rej_char))
    correct_chars = tf.to_float(tf.equal(predictions, targets))
    accuracy_per_example = tf.div(
        tf.reduce_sum(tf.multiply(correct_chars, weights), 1), tf.reduce_sum(weights, 1))
    if streaming:
      return tf.contrib.metrics.streaming_mean(accuracy_per_example)
    else:
      return tf.reduce_mean(accuracy_per_example)


def sequence_accuracy(predictions, targets, rej_char, streaming=False):
  """Computes sequence level accuracy. Both input tensors should have the same
  shape: [batch_size x seq_length].

  Args:
      predictions: predicted character classes.
      targets: ground truth character classes.
      rej_char: the character id used to mark empty element (end of sequence).
      streaming: if True, uses the streaming mean from the slim.metric module.

  Returns:
      a update_ops for execution and value tensor whose value on evaluation
          returns the total sequence accuracy.
  """

  with tf.variable_scope('SequenceAccuracy'):
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())

    targets = tf.to_int32(targets)
    const_rej_char = tf.constant(rej_char, shape=targets.get_shape(), dtype=tf.int32)
    include_mask = tf.not_equal(targets, const_rej_char)
    include_predictions = tf.to_int32(
        tf.where(include_mask, predictions,
                 tf.zeros_like(predictions) + rej_char))
    correct_chars = tf.to_float(tf.equal(include_predictions, targets))
    correct_chars_counts = tf.cast(
        tf.reduce_sum(correct_chars, reduction_indices=[1]), dtype=tf.int32)
    target_length = targets.get_shape().dims[1].value
    target_chars_counts = tf.constant(target_length, shape=correct_chars_counts.get_shape())
    accuracy_per_example = tf.to_float(tf.equal(correct_chars_counts, target_chars_counts))
    if streaming:
      return tf.contrib.metrics.streaming_mean(accuracy_per_example)
    else:
      return tf.reduce_mean(accuracy_per_example)


def accumulate_strings(values, name="strings"):
  """Accumulates strings into a vector.

  Args:
    values: A 1-d string tensor that contains values to add to the accumulator.

  Returns:
    A tuple (value_tensor, update_op).
  """
  tf.assert_type(values, tf.string)
  strings = tf.Variable(
      name=name,
      initial_value=[],
      dtype=tf.string,
      trainable=False,
      collections=[],
      validate_shape=True)
  value_tensor = tf.identity(strings)
  update_op = tf.assign(ref=strings, value=tf.concat([strings, values], 0), validate_shape=False)
  return value_tensor, update_op


@six.add_metaclass(abc.ABCMeta)
class TextMetricSpec(Configurable, MetricSpec):
  """Abstract class for text-based metrics calculated based on hypotheses and
  references. Subclasses must implement `metric_fn`.

  Args:
    name: A name for the metric
    separator: A separator used to join predicted tokens. Default to space.
    eos_token: A string token used to find the end of a sequence. Hypotheses
      and references will be slcied until this token is found.
  """

  def __init__(self, params, name):
    # We don't call the super constructor on purpose
    """Initializer."""
    Configurable.__init__(self, params, tf.contrib.learn.ModeKeys.EVAL)
    self._name = name
    self._eos_token = self.params["eos_token"]
    self._sos_token = self.params["sos_token"]
    self._separator = self.params["separator"]
    self._postproc_fn = None
    if self.params["postproc_fn"]:
      self._postproc_fn = locate(self.params["postproc_fn"])
      if self._postproc_fn is None:
        raise ValueError("postproc_fn not found: {}".format(self.params["postproc_fn"]))

  @property
  def name(self):
    """Name of the metric."""
    return self._name

  @staticmethod
  def default_params():
    return {
        "sos_token": "SEQUENCE_START",
        "eos_token": "SEQUENCE_END",
        "separator": " ",
        "postproc_fn": "",
    }

  def create_metric_ops(self, _inputs, labels, predictions):
    """Creates (value, update_op) tensors."""
    with tf.variable_scope(self._name):

      # Join tokens into single strings
      predictions_flat = tf.reduce_join(
          predictions["predicted_tokens"], 1, separator=self._separator)
      labels_flat = tf.reduce_join(labels["target_tokens"], 1, separator=self._separator)

      sources_value, sources_update = accumulate_strings(values=predictions_flat, name="sources")
      targets_value, targets_update = accumulate_strings(values=labels_flat, name="targets")

      metric_value = tf.py_func(
          func=self._py_func, inp=[sources_value, targets_value], Tout=tf.float32, name="value")

    with tf.control_dependencies([sources_update, targets_update]):
      update_op = tf.identity(metric_value, name="update_op")

    return metric_value, update_op

  def _py_func(self, hypotheses, references):
    """Wrapper function that converts tensors to unicode and slices them until
    the EOS token is found."""
    # Deal with byte chars
    if hypotheses.dtype.kind == np.dtype("U"):
      hypotheses = np.char.encode(hypotheses, "utf-8")
    if references.dtype.kind == np.dtype("U"):
      references = np.char.encode(references, "utf-8")

    # Convert back to unicode object
    hypotheses = [_.decode("utf-8") for _ in hypotheses]
    references = [_.decode("utf-8") for _ in references]

    # Slice all hypotheses and references up to SOS -> EOS
    sliced_hypotheses = [
        postproc.slice_text(_, self._eos_token, self._sos_token) for _ in hypotheses
    ]
    sliced_references = [
        postproc.slice_text(_, self._eos_token, self._sos_token) for _ in references
    ]

    # Apply postprocessing function
    if self._postproc_fn:
      sliced_hypotheses = [self._postproc_fn(_) for _ in sliced_hypotheses]
      sliced_references = [self._postproc_fn(_) for _ in sliced_references]

    return self.metric_fn(sliced_hypotheses, sliced_references)

  def metric_fn(self, hypotheses, references):
    """Calculates the value of the metric.

    Args:
      hypotheses: A python list of strings, each corresponding to a
        single hypothesis/example.
      references: A python list of strings, each corresponds to a single
        reference. Must have the same number of elements of `hypotheses`.

    Returns:
      A float value.
    """
    raise NotImplementedError()


class BleuMetricSpec(TextMetricSpec):
  """Calculates BLEU score using the Moses multi-bleu.perl script."""

  def __init__(self, params):
    super(BleuMetricSpec, self).__init__(params, "bleu")

  def metric_fn(self, hypotheses, references):
    return moses_multi_bleu(hypotheses, references, lowercase=False)


class RougeMetricSpec(TextMetricSpec):
  """Calculates BLEU score using the Moses multi-bleu.perl script."""

  def __init__(self, params, **kwargs):
    if not params["rouge_type"]:
      raise ValueError("You must provide a rouge_type for ROUGE")
    super(RougeMetricSpec, self).__init__(params, params["rouge_type"], **kwargs)
    self._rouge_type = self.params["rouge_type"]

  @staticmethod
  def default_params():
    params = TextMetricSpec.default_params()
    params.update({
        "rouge_type": "",
    })
    return params

  def metric_fn(self, hypotheses, references):
    if not hypotheses or not references:
      return np.float32(0.0)
    return np.float32(rouge(hypotheses, references)[self._rouge_type])


class LogPerplexityMetricSpec(MetricSpec, Configurable):
  """A MetricSpec to calculate straming log perplexity."""

  def __init__(self, params):
    """Initializer."""
    # We don't call the super constructor on purpose
    Configurable.__init__(self, params, tf.contrib.learn.ModeKeys.EVAL)

  @staticmethod
  def default_params():
    return {}

  @property
  def name(self):
    """Name of the metric."""
    return "log_perplexity"

  def create_metric_ops(self, _inputs, labels, predictions):
    """Creates the metric op."""
    loss_mask = tf.sequence_mask(
        lengths=tf.to_int32(labels["target_len"] - 1),
        maxlen=tf.to_int32(tf.shape(predictions["losses"])[1]))
    return metrics.streaming_mean(predictions["losses"], loss_mask)


def moses_multi_bleu(hypotheses, references, lowercase=False):
  """Calculate the bleu score for hypotheses and references using the MOSES
  ulti-bleu.perl script.

  Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script

  Returns:
    The BLEU score as a float32 value.
  """

  if np.size(hypotheses) == 0:
    return np.float32(0.0)

  # Get MOSES multi-bleu script
  try:
    multi_bleu_path, _ = urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
        "master/scripts/generic/multi-bleu.perl")
    os.chmod(multi_bleu_path, 0o755)
  except:
    tf.logging.info("Unable to fetch multi-bleu.perl script, using local.")
    metrics_dir = os.path.dirname(os.path.realpath(__file__))
    bin_dir = os.path.abspath(os.path.join(metrics_dir, "..", "..", "tools"))
    multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")

  # Dump hypotheses and references to tempfiles
  hypothesis_file = tempfile.NamedTemporaryFile()
  hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
  hypothesis_file.write(b"\n")
  hypothesis_file.flush()
  reference_file = tempfile.NamedTemporaryFile()
  reference_file.write("\n".join(references).encode("utf-8"))
  reference_file.write(b"\n")
  reference_file.flush()

  # Calculate BLEU using multi-bleu script
  with open(hypothesis_file.name, "r") as read_pred:
    bleu_cmd = [multi_bleu_path]
    if lowercase:
      bleu_cmd += ["-lc"]
    bleu_cmd += [reference_file.name]
    try:
      bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
      bleu_out = bleu_out.decode("utf-8")
      bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
      bleu_score = float(bleu_score)
    except subprocess.CalledProcessError as error:
      if error.output is not None:
        tf.logging.warning("multi-bleu.perl script returned non-zero exit code")
        tf.logging.warning(error.output)
      bleu_score = np.float32(0.0)

  # Close temp files
  hypothesis_file.close()
  reference_file.close()

  return np.float32(bleu_score)


def _get_ngrams(n, text):
  """Calcualtes n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result."""
  return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences."""
  assert len(sentences) > 0
  assert n > 0

  words = _split_into_words(sentences)
  return _get_ngrams(n, words)


def _len_lcs(x, y):
  """
    Returns the length of the Longest Common Subsequence between sequences x
    and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns
      integer: Length of LCS between x and y
    """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _recon_lcs(x, y):
  """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns:
      sequence: LCS of x and y
    """
  i, j = len(x), len(y)
  table = _lcs(x, y)

  def _recon(i, j):
    """private recon calculation."""
    if i == 0 or j == 0:
      return []
    elif x[i - 1] == y[j - 1]:
      return _recon(i - 1, j - 1) + [(x[i - 1], i)]
    elif table[i - 1, j] > table[i, j - 1]:
      return _recon(i - 1, j)
    else:
      return _recon(i, j - 1)

  recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
  return recon_tuple


def rouge_n(evaluated_sentences, reference_sentences, n=2):
  """
    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.

    Returns:
      A tuple (f1, precision, recall) for ROUGE-N

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  reference_count = len(reference_ngrams)
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

  # return overlapping_count / reference_count
  return f1_score, precision, recall


def _f_p_r_lcs(llcs, m, n):
  """
    Computes the LCS-based F-measure score
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Args:
      llcs: Length of LCS
      m: number of words in reference summary
      n: number of words in candidate summary

    Returns:
      Float. LCS-based F-measure score
    """
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs, p_lcs, r_lcs


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
  """Computes ROUGE-L (sentence level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/ rouge-
  working-note-v1.3.1.pdf.

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set

  Returns:
    A float: F_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")
  reference_words = _split_into_words(reference_sentences)
  evaluated_words = _split_into_words(evaluated_sentences)
  m = len(reference_words)
  n = len(evaluated_words)
  lcs = _len_lcs(evaluated_words, reference_words)
  return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
  """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C. For example
    if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
    c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
    “w1 w2” and the longest common subsequence of r_i and c2 is “w1 w3 w5”. The
    union longest common subsequence of r_i, c1, and c2 is “w1 w2 w3 w5” and
    LCS_u(r_i, C) = 4/5.

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      float: LCS_u(r_i, C)

    ValueError:
      Raises exception if a param has len <= 0
    """
  if len(evaluated_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  lcs_union = set()
  reference_words = _split_into_words([reference_sentence])
  combined_lcs_length = 0
  for eval_s in evaluated_sentences:
    evaluated_words = _split_into_words([eval_s])
    lcs = set(_recon_lcs(reference_words, evaluated_words))
    combined_lcs_length += len(lcs)
    lcs_union = lcs_union.union(lcs)

  union_lcs_count = len(lcs_union)
  union_lcs_value = union_lcs_count / combined_lcs_length
  return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
  """Computes ROUGE-L (summary level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/ rouge-
  working-note-v1.3.1.pdf.

  Calculated according to:
  R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
  P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  SUM(i,u) = SUM from i through u
  u = number of sentences in reference summary
  C = Candidate summary made up of v sentences
  m = number of words in reference summary
  n = number of words in candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries

  Returns:
    A float: F_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  # total number of words in reference sentences
  m = len(_split_into_words(reference_sentences))

  # total number of words in evaluated sentences
  n = len(_split_into_words(evaluated_sentences))

  union_lcs_sum_across_all_references = 0
  for ref_s in reference_sentences:
    union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences, ref_s)
  return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


def rouge(hypotheses, references):
  """Calculates average rouge scores for a list of hypotheses and
  references."""

  # Filter out hyps that are of 0 length
  # hyps_and_refs = zip(hypotheses, references)
  # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
  # hypotheses, references = zip(*hyps_and_refs)

  # Calculate ROUGE-1 F1, precision, recall scores
  rouge_1 = [rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)]
  rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

  # Calculate ROUGE-2 F1, precision, recall scores
  rouge_2 = [rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)]
  rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

  # Calculate ROUGE-L F1, precision, recall scores
  rouge_l = [rouge_l_sentence_level([hyp], [ref]) for hyp, ref in zip(hypotheses, references)]
  rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

  return {
      "rouge_1/f_score": rouge_1_f,
      "rouge_1/r_score": rouge_1_r,
      "rouge_1/p_score": rouge_1_p,
      "rouge_2/f_score": rouge_2_f,
      "rouge_2/r_score": rouge_2_r,
      "rouge_2/p_score": rouge_2_p,
      "rouge_l/f_score": rouge_l_f,
      "rouge_l/r_score": rouge_l_r,
      "rouge_l/p_score": rouge_l_p,
  }
