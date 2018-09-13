# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import numpy as np
import tensorflow as tf
import numbers
from functools import partial
from ..utils import util
from .layers import flatten, fully_connected as fc, relu
from .layers import gradient_reverse
from ..utils import losses_utils
log_loss = tf.losses.log_loss


def log_loss_custom(predictions, labels, eps=1e-7, name='log'):
  """Define a log loss.

  Args:
      predictions: 2D tensor or array, [batch_size, num_classes] predictions of the network .
      labels: 2D or array tensor, [batch_size, num_classes]  ground truth labels or target labels.
      eps: a constant to set upper or lower limit for labels, smoothening factor
      name: Optional scope/name for op_scope.

  Returns:
      A tensor with the log loss.
  """
  with tf.name_scope(name):
    predictions = tf.to_float(predictions)
    labels = tf.to_float(labels)
    predictions = tf.clip_by_value(predictions, eps, 1 - eps)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    loss = -tf.reduce_mean(labels * tf.log(predictions))
    return loss


def log_loss_tf(predictions, labels, eps=1e-7, weights=1.0, name='log_loss'):
  """Define a log loss.

  Args:
      predictions: 2D tensor or array, [batch_size, num_classes] predictions of the network .
      labels: 2D or array tensor, [batch_size, num_classes]  ground truth labels or target labels.
      eps: a constant to set upper or lower limit for labels, smoothening factor
      name: Optional scope/name for op_scope.

  Returns:
      A tensor with the log loss.
  """
  with tf.name_scope(name):
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    predictions = tf.to_float(predictions)
    labels = tf.to_float(labels)
    losses = -tf.multiply(labels, tf.log(predictions + eps)) - tf.multiply(
        (1 - labels), tf.log(1 - predictions + eps))
  return tf.losses.compute_weighted_loss(losses, weights)


def kappa_loss(predictions, labels, y_pow=1, eps=1e-15, num_ratings=5, batch_size=32, name='kappa'):
  """Define a kappa loss, Its a continuous differentiable approximation of
  discrete kappa loss.

  Args:
      predictions: 2D tensor or array, [batch_size, num_classes] predictions of the network .
      labels: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
      y_pow: int, to whcih the labels should be raised; useful if model diverge. e.g. y_pow=2
      num_ratings: numbers of rater to used, typically num_classes of the model
      batch_size: batch_size of the training or validation ops
      eps: a float, prevents divide by zero
      name: Optional scope/name for op_scope.

  Returns:
      A tensor with the kappa loss.
  """
  with tf.name_scope(name):
    labels = tf.to_float(labels)
    repeat_op = tf.to_float(
        tf.tile(tf.reshape(tf.range(0, num_ratings), [num_ratings, 1]), [1, num_ratings]))
    repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
    weights = repeat_op_sq / tf.to_float((num_ratings - 1)**2)

    pred_ = predictions**y_pow
    try:
      pred_norm = pred_ / \
          (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
    except Exception:
      pred_norm = pred_ / \
          (eps + tf.reshape(tf.reduce_sum(pred_, 1), [batch_size, 1]))

    hist_rater_a = tf.reduce_sum(pred_norm, 0)
    hist_rater_b = tf.reduce_sum(labels, 0)

    conf_mat = tf.matmul(tf.transpose(pred_norm), labels)

    nom = tf.reduce_sum(weights * conf_mat)
    denom = tf.reduce_sum(weights * tf.matmul(
        tf.reshape(hist_rater_a, [num_ratings, 1]), tf.reshape(hist_rater_b, [1, num_ratings])) /
                          tf.to_float(batch_size))

    try:
      return -(1 - nom / denom)
    except Exception:
      return -(1 - nom / (denom + eps))


def kappa_log_loss(predictions,
                   labels,
                   label_smoothing=0.0,
                   y_pow=1,
                   batch_size=32,
                   log_scale=0.5,
                   num_classes=5,
                   log_offset=0.50,
                   name='kappa_log'):
  """Define a joint kappa and log loss, Kappa is a continuous differentiable
  approximation of discrete kappa loss.

  Args:
      predictions: 2D tensor or array, [batch_size, num_classes] predictions of the network .
      labels: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
      label_smoothing: a float, used to smooth the labels for better generalization
                       if greater than 0 then smooth the labels.
      y_pow: int, to whcih the labels should be raised; useful if model diverge. e.g. y_pow=2
      num_ratings: numbers of rater to used, typically num_classes of the model
      batch_size: batch_size of the training or validation ops
      log_scale: a float, used to multiply the clipped log loss, e.g: 0.5
      log_offset:a float minimum log loss offset to substract from original log loss; e.g. 0.50
      name: Optional scope/name for op_scope.

  Returns:
      A tensor with the kappa log loss.
  """
  with tf.name_scope(name):
    num_classes = labels.get_shape()[-1].value
    labels = tf.cast(labels, predictions.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      labels = labels * smooth_positives + smooth_negatives
    log_loss_res = log_loss(predictions, labels)
    kappa_loss_res = kappa_loss(
        predictions, labels, y_pow=y_pow, num_ratings=num_classes, batch_size=batch_size)
    return kappa_loss_res + log_scale * (log_loss_res - log_offset)


def kappa_log_loss_clipped(predictions,
                           labels,
                           label_smoothing=0.0,
                           y_pow=1,
                           batch_size=32,
                           log_scale=0.5,
                           log_cutoff=0.80,
                           num_classes=5,
                           name='kappa_log_clipped'):
  """Define a joint kappa and log loss; log loss is clipped by a defined min
  value; Kappa is a continuous differentiable approximation of discrete kappa
  loss.

  Args:
      predictions: 2D tensor or array, [batch_size, num_classes] predictions of the network .
      labels: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
      label_smoothing: a float, used to smooth the labels for better generalization
                       if greater than 0 then smooth the labels.
      y_pow: int, to whcih the labels should be raised; useful if model diverge. e.g. y_pow=2
      num_ratings: numbers of rater to used, typically num_classes of the model
      batch_size: batch_size of the training or validation ops
      log_scale: a float, used to multiply the clipped log loss, e.g: 0.5
      log_cutoff:a float, minimum log loss value; e.g. 0.50
      name: Optional scope/name for op_scope.

  Returns:
      A tensor with the clipped kappa log loss.
  """
  with tf.name_scope(name):
    num_classes = labels.get_shape()[-1].value
    labels = tf.cast(labels, predictions.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      labels = labels * smooth_positives + smooth_negatives
    log_loss_res = log_loss_tf(predictions, labels)
    kappa_loss_res = kappa_loss(
        predictions, labels, y_pow=y_pow, num_ratings=num_classes, batch_size=batch_size)
    return kappa_loss_res + log_scale * tf.clip_by_value(log_loss_res, log_cutoff, 10**3)


def cross_entropy_loss(logits, labels, label_smoothing=0.0, weight=1.0, name='cross_entropy_loss'):
  """Define a cross entropy loss with label smoothing.

  Args:
      predictions: 2D tensor or array, [batch_size, num_classes] predictions of the network .
      labels: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
      label_smoothing: a float, used to smooth the labels for better generalization
                      if greater than 0 then smooth the labels.
      weight: scale the loss by this factor.
      name: Optional scope/name for op_scope.

  Returns:
      A tensor with the cross entropy loss.
  """
  logits.get_shape().assert_is_compatible_with(labels.get_shape())
  with tf.name_scope(name):
    num_classes = labels.get_shape()[-1].value
    labels = tf.cast(labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      labels = labels * smooth_positives + smooth_negatives
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    weight = tf.convert_to_tensor(weight, dtype=logits.dtype.base_dtype, name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
    return loss


def l1_l2_regularizer(var, weight_l1=1.0, weight_l2=1.0, name='l1_l2_regularizer'):
  """Define a L2Loss, useful for regularize, i.e. weight decay.

  Args:
      var: tensor to regularize.
      weight_l1: an optional weight to modulate the l1 loss.
      weight_l2: an optional weight to modulate the l2 loss.
      name: Optional scope/name for op_scope.

  Returns:
      the l1+L2 loss op.
  """
  with tf.name_scope(name):
    weight_l1_t = tf.convert_to_tensor(weight_l1, dtype=var.dtype.base_dtype, name='weight_l1')
    weight_l2_t = tf.convert_to_tensor(weight_l2, dtype=var.dtype.base_dtype, name='weight_l2')
    reg_l1 = tf.multiply(weight_l1_t, tf.reduce_sum(tf.abs(var)), name='value_l1')
    reg_l2 = tf.multiply(weight_l2_t, tf.nn.l2_loss(var), name='value_l2')
    return tf.add(reg_l1, reg_l2, name='value')


def l1_regularizer(scale, name='l1_regularizer'):
  """Returns a function that can be used to apply L1 regularization to weights.
  L1 regularization encourages sparsity.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    name: An optional name/scope name.

  Returns:
    A function with signature `l1(weights)` that apply L1 regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' % scale)
    if scale == 0.:
      return lambda _: None

  def l1(weights, name='l1_regularizer'):
    """Applies L1 regularization to weights."""
    with tf.name_scope(name):
      my_scale = tf.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
      return tf.multiply(my_scale, tf.reduce_sum(tf.abs(weights)), name=name)

  return l1


def l2_regularizer(scale, name='l2_regularizer'):
  """Returns a function that can be used to apply L2 regularization to weights.
  Small values of L2 can help prevent overfitting the training data.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    name: An optional name/scope name.

  Returns:
    A function with signature `l2(weights)` that applies L2 regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' % scale)
    if scale == 0.:
      return lambda _: None

  def l2(weights, name='l2_regularizer'):
    """Applies l2 regularization to weights."""
    with tf.name_scope(name):
      my_scale = tf.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
      return tf.multiply(my_scale, nn.l2_loss(weights), name=name)

  return l2


def discretized_mix_logistic_loss(inputs,
                                  predictions,
                                  sum_all=True,
                                  name='disretized_mix_logistic_loss'):
  """log-likelihood for mixture of discretized logistics, assumes the data has
  been rescaled to.

      [-1,1] interval

  Args:
      predictions: 4D tensor or array, [batch_size, width, height, out_channels]
          predictions of the network .
      inputs: 4D tensor or array, [batch_size, width, height, num_classes]
          ground truth labels or target labels.
      name: Optional scope/name for op_scope.

  Returns:
      A tensor with the discretized mix logistic loss.
  """
  with tf.name_scope(name):
    inputs_shape = list(map(int, inputs.get_shape()))
    predictions_shape = list(map(int, predictions.get_shape()))
    nr_mix = int(predictions_shape[-1] / 10)
    logit_probs = predictions[:, :, :, :nr_mix]
    predictions = tf.reshape(predictions[:, :, :, nr_mix:], inputs_shape + [nr_mix * 3])
    means = predictions[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(predictions[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = tf.nn.tanh(predictions[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    inputs = tf.reshape(inputs, inputs_shape + [1]) + tf.zeros(inputs_shape + [nr_mix])
    m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * inputs[:, :, :, 0, :],
                    [inputs_shape[0], inputs_shape[1], inputs_shape[2], 1, nr_mix])
    m3 = tf.reshape(
        means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * inputs[:, :, :, 0, :] +
        coeffs[:, :, :, 2, :] * inputs[:, :, :, 1, :],
        [inputs_shape[0], inputs_shape[1], inputs_shape[2], 1, nr_mix])
    means = tf.concat([
        tf.reshape(means[:, :, :, 0, :],
                   [inputs_shape[0], inputs_shape[1], inputs_shape[2], 1, nr_mix]), m2, m3
    ],
                      axis=3)
    centered_inputs = inputs - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_inputs + 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_inputs - 1. / 255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_inputs
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)
    log_probs = tf.select(
        inputs < -0.999, log_cdf_plus,
        tf.select(
            inputs > 0.999, log_one_minus_cdf_min,
            tf.select(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)),
                      log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs, 3) + \
        log_prob_from_logits(logit_probs)
    if sum_all:
      return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
      return -tf.reduce_sum(log_sum_exp(log_probs), [1, 2])


def mse_loss(pred, labels):
  try:
    batch_size = tf.cast(pred.shape[0], tf.float32)
  except Exception as e:
    print('Pred is a tf tensor %s' % str(e.message))
    batch_size = tf.cast(tf.shape(pred)[0], tf.float32)
  loss_val = tf.sqrt(2 * tf.nn.l2_loss(pred - labels)) / batch_size
  return loss_val


def pullaway_loss(embeddings, name='pullaway_loss'):
  """Pull Away loss calculation.

  Args:
      embeddings: The embeddings to be orthogonalized for varied faces.
         Shape [batch_size, embeddings_dim]

  Return: pull away term loss
  """
  with tf.name_scope(name):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
    batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
    pt_loss = (tf.reduce_sum(similarity) - batch_size) / \
        (batch_size * (batch_size - 1))
    return pt_loss


def log_sum_exp(x):
  """numerically stable log_sum_exp implementation that prevents overflow."""
  axis = len(x.get_shape()) - 1
  m = tf.reduce_max(x, axis)
  m2 = tf.reduce_max(x, axis, keep_dims=True)
  return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
  """numerically stable log_softmax implementation that prevents overflow."""
  axis = len(x.get_shape()) - 1
  m = tf.reduce_max(x, axis, keep_dims=True)
  return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))


def segment_loss(logits, labels, num_classes, head=None):
  """Calculate the loss from the logits and the labels.

  Args:
    logits: tensor, float - [batch_size * width * height, num_classes].
        Use vgg_fcn.up as logits.
    labels: Labels tensor, int32 - [batch_size * width * height, num_classes].
        The ground truth of your data.
    head: numpy array - [num_classes]
        Weighting the loss of each class
        Optional: Prioritize some classes

  Returns:
    loss: Loss tensor of type float.
  """
  with tf.name_scope('segment_loss'):
    # logits = tf.reshape(logits, (-1, num_classes))
    epsilon = tf.constant(value=1e-7)
    labels = tf.to_float(labels)
    # labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

    softmax = tf.nn.softmax(logits) + epsilon

    if head is not None:
      cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax), head), axis=[1])
    else:
      cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), axis=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return cross_entropy_mean


def triplet_loss(anchor, positive, negative, alpha=0.2, name='triplet_loss'):
  """Calculate the triplet loss according to the FaceNet paper.

  Args:
    anchor: 2-D `tensor` [batch_size, embedding_size], the embeddings for the anchor images.
    positive: 2-D `tensor` [batch_size, embedding_size], the embeddings for the positive images.
    negative: 2-D `tensor` [batch_size, embedding_size], the embeddings for the negative images.
    alpha: positive to negative triplet distance margin

  Returns:
    the triplet loss.
  """
  with tf.name_scope(name):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
  return loss


def decov_loss(xs, name='decov_loss'):
  """Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf 'Reducing
  Overfitting In Deep Networks by Decorrelating Representation'.

  Args:
      xs: 4-D `tensor` [batch_size, height, width, channels], input

  Returns:
      a `float` decov loss
  """
  with tf.name_scope(name):
    x = tf.reshape(xs, [int(xs.get_shape()[0]), -1])
    m = tf.reduce_mean(x, 0, True)
    z = tf.expand_dims(x - m, 2)
    corr = tf.reduce_mean(tf.matmul(z, tf.transpose(z, perm=[0, 2, 1])), 0)
    corr_frob_sqr = tf.reduce_sum(tf.square(corr))
    corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
    loss = 0.5 * (corr_frob_sqr - corr_diag_sqr)
    return loss


def center_loss(features, label, alpha, num_classes, name='center_loss'):
  """Center loss based on the paper "A Discriminative Feature Learning Approach
  for Deep Face Recognition" (http://ydwen.github.io/papers/WenECCV16.pdf)

  Args:
      features: 2-D `tensor` [batch_size, feature_length], input features
      label: 1-D `tensor` [batch_size], input label
      alpha: center loss parameter
      num_classes: a `int` numof classes for training

  Returns:
      a `float`, center loss
  """
  with tf.variable_scope(name):
    num_features = features.get_shape()[1]
    centers = tf.get_variable(
        'centers', [num_classes, num_features],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0),
        trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alpha) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.nn.l2_loss(features - centers_batch)
    return loss, centers


def correlation_loss(source_samples, target_samples, weight, name='corr_loss'):
  """Adds a similarity loss term, the correlation between two representations.

  Args:
      source_samples: a tensor of shape [num_samples, num_features]
      target_samples: a tensor of shape [num_samples, num_features]
      weight: a scalar weight for the loss.
      scope: optional name scope for summary tags.

  Returns:
      a scalar tensor representing the correlation loss value.
  """
  with tf.name_scope(name):
    source_samples -= tf.reduce_mean(source_samples, 0)
    target_samples -= tf.reduce_mean(target_samples, 0)
    source_samples = tf.nn.l2_normalize(source_samples, 1)
    target_samples = tf.nn.l2_normalize(target_samples, 1)
    source_cov = tf.matmul(tf.transpose(source_samples), source_samples)
    target_cov = tf.matmul(tf.transpose(target_samples), target_samples)
    corr_loss = tf.reduce_mean(tf.square(source_cov - target_cov)) * weight

  assert_op = tf.Assert(tf.is_finite(corr_loss), [corr_loss])
  with tf.control_dependencies([assert_op]):
    tag = 'Correlation Loss'
    barrier = tf.no_op(tag)

  return corr_loss


def maximum_mean_discrepancy(x,
                             y,
                             kernel=util.gaussian_kernel_matrix,
                             name='maximum_mean_discrepancy'):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.

    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.

    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.

    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
  with tf.name_scope(name):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples, weight, name='mmd_loss'):
  """Adds a similarity loss term, the MMD between two representations.

  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the MMD loss value.
  """
  with tf.name_scope(name):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(util.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight
  assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  with tf.control_dependencies([assert_op]):
    tag = 'MMD_Loss'
    barrier = tf.no_op(tag)
  return loss_value


def dann_loss(source_samples, target_samples, weight, name='dann_loss'):
  """Adds the domain adversarial (DANN) loss.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.variable_scope(name):
    batch_size = tf.shape(source_samples)[0]
    samples = tf.concat(values=[source_samples, target_samples], axis=0)
    samples = flatten(samples)

    domain_selection_mask = tf.concat(
        values=[tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)

    grl = gradient_reverse(samples)
    grl = tf.reshape(grl, (-1, samples.get_shape().as_list()[1]))

    grl = fc(grl, 100, True, None, activation=relu, name='fc1')
    logits = fc(grl, 1, True, None, activation=None, name='fc2')

    domain_predictions = tf.sigmoid(logits)

  domain_loss = tf.losses.log_loss(domain_selection_mask, domain_predictions, weights=weight)

  domain_accuracy = util.accuracy_tf(domain_selection_mask, tf.round(domain_predictions))

  assert_op = tf.Assert(tf.is_finite(domain_loss), [domain_loss])
  with tf.control_dependencies([assert_op]):
    tag_loss = 'losses/domain_loss'
    barrier = tf.no_op(tag_loss)

  return domain_loss


def difference_loss(private_samples, shared_samples, weight=1.0, name='difference_loss'):
  """Adds the difference loss between the private and shared representations.

  Args:
    private_samples: a tensor of shape [num_samples, num_features].
    shared_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the incoherence loss.
    name: the name of the tf summary.
  """
  with tf.name_scope(name):
    private_samples -= tf.reduce_mean(private_samples, 0)
    shared_samples -= tf.reduce_mean(shared_samples, 0)

    private_samples = tf.nn.l2_normalize(private_samples, 1)
    shared_samples = tf.nn.l2_normalize(shared_samples, 1)

    correlation_matrix = tf.matmul(private_samples, shared_samples, transpose_a=True)

    cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
    cost = tf.where(cost > 0, cost, 0, name='value')

  assert_op = tf.Assert(tf.is_finite(cost), [cost])
  with tf.control_dependencies([assert_op]):
    barrier = tf.no_op(name)
  return cost


def log_quaternion_loss_batch(predictions, labels, name='log_quaternion_batch_loss'):
  """A helper function to compute the error between quaternions.

  Args:
    predictions: A Tensor of size [batch_size, 4].
    labels: A Tensor of size [batch_size, 4].
    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.

  Returns:
    A Tensor of size [batch_size], denoting the error between the quaternions.
  """
  assertions = []
  assertions.append(
      tf.Assert(
          tf.reduce_all(tf.less(tf.abs(tf.reduce_sum(tf.square(predictions), [1]) - 1), 1e-4)),
          ['The l2 norm of each prediction quaternion vector should be 1.']))
  assertions.append(
      tf.Assert(
          tf.reduce_all(tf.less(tf.abs(tf.reduce_sum(tf.square(labels), [1]) - 1), 1e-4)),
          ['The l2 norm of each label quaternion vector should be 1.']))
  with tf.name_scope(name):
    with tf.control_dependencies(assertions):
      product = tf.multiply(predictions, labels)
    internal_dot_products = tf.reduce_sum(product, [1])
    logcost = tf.log(1e-4 + 1 - tf.abs(internal_dot_products))
  return logcost


def log_quaternion_loss(predictions, labels, batch_size, name='log_quaternion_loss'):
  """A helper function to compute the mean error between batches of
  quaternions.

  The caller is expected to add the loss to the graph.

  Args:
    predictions: A Tensor of size [batch_size, 4].
    labels: A Tensor of size [batch_size, 4].
    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.

  Returns:
    A Tensor of size 1, denoting the mean error between batches of quaternions.
  """
  with tf.name_scope(name):
    logcost = log_quaternion_loss_batch(predictions, labels)
    logcost = tf.reduce_sum(logcost, [0])
    logcost = tf.multiply(logcost, 1.0 / batch_size, name='log_quaternion_loss')
  return logcost


def random_perturbation_loss(embedded, length, loss_fn, perturb_norm_length=0.1):
  """Adds noise to embeddings and recomputes classification loss.

  Args:
      embedded: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim]
      length: a `int`, length of the mask
      loss_fn: a callable, that returns loss
      perturb_norm_length: a `float`, Norm length of adversarial perturbation
          to be optimized with validatio

  Returns:
      perturbation loss
  """
  noise = tf.random_normal(shape=tf.shape(embedded))
  perturb = _scale_l2(_mask_by_length(noise, length), perturb_norm_length)
  return loss_fn(embedded + perturb)


def adversarial_loss(embedded, loss, loss_fn, perturb_norm_length=0.1):
  """Adds gradient to embedding and recomputes classification loss.

  Args:
      embedded: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim]
      loss: `float`, loss
      loss_fn: a callable, that returns loss
      perturb_norm_length: a `float`, Norm length of adversarial perturbation
          to be optimized with validatio

  Returns:
      adversial loss
  """
  grad, = tf.gradients(
      loss, embedded, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
  grad = tf.stop_gradient(grad)
  perturb = _scale_l2(grad, perturb_norm_length)
  return loss_fn(embedded + perturb)


def virtual_adversarial_loss(logits,
                             embedded,
                             labels,
                             length,
                             logits_from_embedding_fn,
                             num_classes,
                             num_power_iteration=1,
                             small_constant_for_finite_diff=1e-3,
                             perturb_norm_length=0.1):
  """Virtual adversarial loss. Computes virtual adversarial perturbation by
  finite difference method and power iteration, adds it to the embedding, and
  computes the KL divergence between the new logits and the original logits.

  Args:
      logits: 2-D float `Tensor`, [num_timesteps*batch_size, m], where m=1 if
          num_classes=2, otherwise m=num_classes.
      embedded: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim].
      labels: 1-D `Tensor`, input labels
      length: a `int`, input length
      logits_from_embedding_fn: callable that takes embeddings and returns
          classifier logits.
      num_classes: num_classes for training
      vocab_size: a `int`, vocabular size of the problem
      num_power_iteration: a `int`, the number of power iteration
      small_constant_for_finite_diff: a `float`, Small constant for finite difference method
      perturb_norm_length: a `float`, Norm length of adversarial perturbation
          to be optimized with validatio

  Returns:
      a `float` `scalar`, KL divergence.
  """
  logits = tf.stop_gradient(logits)
  weights = _end_of_seq_mask(labels, vocab_size)

  d = _mask_by_length(tf.random_normal(shape=tf.shape(embedded)), length)

  for _ in range(num_power_iteration):
    d = _scale_l2(d, small_constant_for_finite_diff)
    d_logits = logits_from_embedding_fn(embedded + d)
    kl = _kl_divergence_with_logits(logits, d_logits, weights, num_classes)
    d, = tf.gradients(kl, d, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    d = tf.stop_gradient(d)

  perturb = _scale_l2(_mask_by_length(d, length), perturb_norm_length)
  vadv_logits = logits_from_embedding_fn(embedded + perturb)
  return _kl_divergence_with_logits(logits, vadv_logits, weights, num_classes)


def random_perturbation_loss_brnn(embedded, length, loss_fn, perturb_norm_length=0.1):
  """Adds noise to embeddings and recomputes classification loss fir
  bidirectional rnn models.

  Args:
      embedded: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim]
      length: a `int`, length of the mask
      loss_fn: a callable, that returns loss
      perturb_norm_length: a `float`, Norm length of adversarial perturbation to
          be optimized with validatio

  Returns:
      perturbation loss
  """
  noise = [tf.random_normal(shape=tf.shape(emb)) for emb in embedded]
  masked = [_mask_by_length(n, length) for n in noise]
  scaled = [_scale_l2(m, perturb_norm_length) for m in masked]
  return loss_fn([e + s for (e, s) in zip(embedded, scaled)])


def adversarial_loss_brnn(embedded, loss, loss_fn, perurb_norm_length=0.1):
  """Adds gradient to embeddings and recomputes classification loss for
  bidirectional rnn models.

  Args:
      embedded: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim]
      loss: `float`, loss
      loss_fn: a callable, that returns loss
      perturb_norm_length: a `float`, Norm length of adversarial perturbation
          to be optimized with validatio

  Returns:
      adversial loss
  """
  grads = tf.gradients(
      loss, embedded, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
  adv_exs = [
      emb + _scale_l2(tf.stop_gradient(g), perturb_norm_length) for emb, g in zip(embedded, grads)
  ]
  return loss_fn(adv_exs)


def virtual_adversarial_loss_brnn(logits,
                                  embedded,
                                  labels,
                                  length,
                                  logits_from_embedding_fn,
                                  vocab_size,
                                  num_classes,
                                  num_power_iteration=1,
                                  small_constant_for_finite_diff=1e-3,
                                  perturb_norm_length=0.1):
  """Virtual adversarial loss for bidirectional models Computes virtual
  adversarial perturbation by finite difference method and power iteration,
  adds it to the embedding, and computes the KL divergence between the new
  logits and the original logits.

  Args:
      logits: 2-D float `Tensor`, [num_timesteps*batch_size, m], where m=1 if
          num_classes=2, otherwise m=num_classes.
      embedded: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim].
      labels: 1-D `Tensor`, input labels
      length: a `int`, input length
      logits_from_embedding_fn: callable that takes embeddings and returns
          classifier logits.
      num_classes: num_classes for training
      vocab_size: a `int`, vocabular size of the problem
      num_power_iteration: a `int`, the number of power iteration
      small_constant_for_finite_diff: a `float`, Small constant for finite difference method
      perturb_norm_length: a `float`, Norm length of adversarial perturbation
          to be optimized with validatio

  Returns:
      a `float` `scalar`, KL divergence.
  """
  logits = tf.stop_gradient(logits)
  weights = _end_of_seq_mask(labels, vocab_size)

  perturbs = [_mask_by_length(tf.random_normal(shape=tf.shape(emb)), length) for emb in embedded]
  for _ in range(num_power_iteration):
    perturbs = [_scale_l2(d, small_constant_for_finite_diff) for d in perturbs]
    d_logits = logits_from_embedding_fn([emb + d for (emb, d) in zip(embedded, perturbs)])
    kl = _kl_divergence_with_logits(logits, d_logits, weights, num_classes)
  perturbs = tf.gradients(
      kl, perturbs, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
  perturbs = [tf.stop_gradient(d) for d in perturbs]

  perturbs = [_scale_l2(_mask_by_length(d, length), perturb_norm_length) for d in perturbs]
  vadv_logits = logits_from_embedding_fn([emb + d for (emb, d) in zip(embedded, perturbs)])
  return _kl_divergence_with_logits(logits, vadv_logits, weights, num_classes)


def _mask_by_length(t, length):
  maxlen = t.get_shape().as_list()[1]
  mask = tf.sequence_mask(length, maxlen=maxlen)
  mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
  return t * mask


def _scale_l2(x, norm_length):
  alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
  l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
  x_unit = x / l2_norm
  return norm_length * x_unit


def _end_of_seq_mask(tokens, vocab_size):
  """Generate a mask for the EOS token (1.0 on EOS, 0.0 otherwise).

  Args:
      tokens: 1-D integer `Tensor` [num_timesteps*batch_size]. Each element is an
          id from the vocab.
      vocab_size: a `int`, vocabular size of the problem

  Returns:
      Float 1-D `Tensor` same shape as tokens, whose values are 1.0 on the end of
          sequence and 0.0 on the others.
  """
  eos_id = vocab_size - 1
  return tf.cast(tf.equal(tokens, eos_id), tf.float32)


def _kl_divergence_with_logits(q_logits, p_logits, weights, num_classes):
  """Returns weighted KL divergence between distributions q and p.

  Args:
      q_logits: logits for 1st argument of KL divergence shape
            [num_timesteps * batch_size, num_classes] if num_classes > 2, and
            [num_timesteps * batch_size] if num_classes == 2.
      p_logits: logits for 2nd argument of KL divergence with same shape q_logits.
      weights: 1-D `float` tensor with shape [num_timesteps * batch_size].
           Elements should be 1.0 only on end of sequences
      num_classes: a `int`, number of training classes

  Returns:
      a `float` `scalar`, KL divergence.
  """
  if num_classes == 2:
    q = tf.nn.sigmoid(q_logits)
    p = tf.nn.sigmoid(p_logits)
    kl = (-tf.nn.sigmoid_cross_entropy_with_logits(logits=q_logits, labels=q) +
          f.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=q))

  else:
    q = tf.nn.softmax(q_logits)
    p = tf.nn.softmax(p_logits)
    kl = tf.reduce_sum(q * (tf.log(q) - tf.log(p)), 1)

  num_labels = tf.reduce_sum(weights)
  num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)

  kl.get_shape().assert_has_rank(2)
  weights.get_shape().assert_has_rank(1)
  loss = tf.identity(tf.reduce_sum(tf.expand_dims(weights, -1) * kl) / num_labels, name='kl')
  return loss


def cross_entropy_sequence_loss(logits, targets, sequence_length):
  """Calculates the per-example cross-entropy loss for a sequence of logits and
  masks out all losses passed the sequence length.

  Args:
      logits: Logits of shape `[T, B, vocab_size]`
      targets: Target classes of shape `[T, B]`
      sequence_length: An int32 tensor of shape `[B]` corresponding
         to the length of each input

  Returns:
      A tensor of shape [T, B] that contains the loss per example, per time step.
  """
  with tf.name_scope("cross_entropy_sequence_loss"):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss_mask = tf.sequence_mask(tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
    losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])

  return losses


def dice_loss(predictions, targets, weights=1., name='dice_loss'):
  with tf.name_scope(name):
    # predictions = tf.to_float(predictions)
    targets = tf.to_float(targets)
    intersection = 2 * tf.reduce_sum(predictions * targets) + weights
    union = weights + tf.reduce_sum(predictions) + tf.reduce_sum(targets)
    loss = -(intersection / (union))
  return loss


def precision_recall_auc_loss(labels,
                              logits,
                              precision_range=(0.0, 1.0),
                              num_anchors=20,
                              weights=1.0,
                              dual_rate_factor=0.1,
                              label_priors=None,
                              surrogate_type='xent',
                              lambdas_initializer=tf.constant_initializer(1.0),
                              reuse=None,
                              variables_collections=None,
                              trainable=True,
                              scope=None):
  """Computes precision-recall AUC loss.
  The loss is based on a sum of losses for recall at a range of
  precision values (anchor points). This sum is a Riemann sum that
  approximates the area under the precision-recall curve.
  The per-example `weights` argument changes not only the coefficients of
  individual training examples, but how the examples are counted toward the
  constraint. If `label_priors` is given, it MUST take `weights` into account.
  That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    precision_range: A length-two tuple, the range of precision values over
      which to compute AUC. The entries must be nonnegative, increasing, and
      less than or equal to 1.0.
    num_anchors: The number of grid points used to approximate the Riemann sum.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
    lambdas_initializer: An initializer for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.
  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [1, num_labels, num_anchors] consisting of the
        Lagrange multipliers.
      biases: A Tensor of shape [1, num_labels, num_anchors] consisting of the
        learned bias term for each.
      label_priors: A Tensor of shape [1, num_labels, 1] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.
  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  with tf.variable_scope(scope, 'precision_recall_auc', [labels, logits, label_priors], reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(labels, logits, weights)
    num_labels = losses_utils.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    dual_rate_factor = losses_utils.convert_and_cast(dual_rate_factor, 'dual_rate_factor',
                                                     logits.dtype)

    # Create Tensor of anchor points and distance between anchors.
    precision_values, delta = _range_to_anchors_and_delta(precision_range, num_anchors, logits.dtype)
    # Create lambdas with shape [1, num_labels, num_anchors].
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[1, num_labels, num_anchors],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Create biases with shape [1, num_labels, num_anchors].
    biases = tf.contrib.framework.model_variable(
        name='biases',
        shape=[1, num_labels, num_anchors],
        dtype=logits.dtype,
        initializer=tf.zeros_initializer(),
        collections=variables_collections,
        trainable=trainable)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(label_priors, labels, weights, variables_collections)
    label_priors = tf.reshape(label_priors, [1, num_labels, 1])

    # Expand logits, labels, and weights to shape [batch_size, num_labels, 1].
    logits = tf.expand_dims(logits, 2)
    labels = tf.expand_dims(labels, 2)
    weights = tf.expand_dims(weights, 2)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    loss = weights * losses_utils.weighted_surrogate_loss(
        labels,
        logits + biases,
        surrogate_type=surrogate_type,
        positive_weights=1.0 + lambdas * (1.0 - precision_values),
        negative_weights=lambdas * precision_values)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * (1.0 - precision_values) * label_priors * maybe_log2
    per_anchor_loss = loss - lambda_term
    per_label_loss = delta * tf.reduce_sum(per_anchor_loss, 2)
    # Normalize the AUC such that a perfect score function will have AUC 1.0.
    # Because precision_range is discretized into num_anchors + 1 intervals
    # but only num_anchors terms are included in the Riemann sum, the
    # effective length of the integration interval is `delta` less than the
    # length of precision_range.
    scaled_loss = tf.div(
        per_label_loss, precision_range[1] - precision_range[0] - delta, name='AUC_Normalize')
    scaled_loss = tf.reshape(scaled_loss, original_shape)

    other_outputs = {
        'lambdas':
        lambdas_variable,
        'biases':
        biases,
        'label_priors':
        label_priors,
        'true_positives_lower_bound':
        true_positives_lower_bound(labels, logits, weights, surrogate_type),
        'false_positives_upper_bound':
        false_positives_upper_bound(labels, logits, weights, surrogate_type)
    }

    return scaled_loss, other_outputs


def roc_auc_loss(labels, logits, weights=1.0, surrogate_type='xent', scope=None):
  """Computes ROC AUC loss.
  The area under the ROC curve is the probability p that a randomly chosen
  positive example will be scored higher than a randomly chosen negative
  example. This loss approximates 1-p by using a surrogate (either hinge loss or
  cross entropy) for the indicator function. Specifically, the loss is:
    sum_i sum_j w_i*w_j*loss(logit_i - logit_j)
  where i ranges over the positive datapoints, j ranges over the negative
  datapoints, logit_k denotes the logit (or score) of the k-th datapoint, and
  loss is either the hinge or log loss given a positive label.
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape and dtype as `labels`.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for the indicator function.
    scope: Optional scope for `name_scope`.
  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise loss.
    other_outputs: An empty dictionary, for consistency.
  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
    # Convert inputs to tensors and standardize dtypes.
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(labels, logits, weights)

    # Create tensors of pairwise differences for logits and labels, and
    # pairwise products of weights. These have shape
    # [batch_size, batch_size, num_labels].
    logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
    labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
    weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

    signed_logits_difference = labels_difference * logits_difference
    raw_loss = losses_utils.weighted_surrogate_loss(
        labels=tf.ones_like(signed_logits_difference),
        logits=signed_logits_difference,
        surrogate_type=surrogate_type)
    weighted_loss = weights_product * raw_loss

    # Zero out entries of the loss where labels_difference zero (so loss is only
    # computed on pairs with different labels).
    loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
    loss = tf.reshape(loss, original_shape)
    return loss, {}


def recall_at_precision_loss(labels,
                             logits,
                             target_precision,
                             weights=1.0,
                             dual_rate_factor=0.1,
                             label_priors=None,
                             surrogate_type='xent',
                             lambdas_initializer=tf.constant_initializer(1.0),
                             reuse=None,
                             variables_collections=None,
                             trainable=True,
                             scope=None):
  """Computes recall at precision loss.
  The loss is based on a surrogate of the form
      wt * w(+) * loss(+) + wt * w(-) * loss(-) - c * pi,
  where:
  - w(+) =  1 + lambdas * (1 - target_precision)
  - loss(+) is the cross-entropy loss on the positive examples
  - w(-) = lambdas * target_precision
  - loss(-) is the cross-entropy loss on the negative examples
  - wt is a scalar or tensor of per-example weights
  - c = lambdas * (1 - target_precision)
  - pi is the label_priors.
  The per-example weights change not only the coefficients of individual
  training examples, but how the examples are counted toward the constraint.
  If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_precision: The precision at which to compute the loss. Can be a
      floating point value between 0 and 1 for a single precision value, or a
      `Tensor` of shape [num_labels], holding each label's target precision
      value.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
    lambdas_initializer: An initializer for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.
  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.
  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  with tf.variable_scope(scope, 'recall_at_precision', [logits, labels, label_priors], reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(labels, logits, weights)
    num_labels = losses_utils.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_precision = losses_utils.convert_and_cast(target_precision, 'target_precision',
                                                     logits.dtype)
    dual_rate_factor = losses_utils.convert_and_cast(dual_rate_factor, 'dual_rate_factor',
                                                     logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(label_priors, labels, weights, variables_collections)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * losses_utils.weighted_surrogate_loss(
        labels,
        logits,
        surrogate_type=surrogate_type,
        positive_weights=1.0 + lambdas * (1.0 - target_precision),
        negative_weights=lambdas * target_precision)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * (1.0 - target_precision) * label_priors * maybe_log2
    loss = tf.reshape(weighted_loss - lambda_term, original_shape)
    other_outputs = {
        'lambdas':
        lambdas_variable,
        'label_priors':
        label_priors,
        'true_positives_lower_bound':
        true_positives_lower_bound(labels, logits, weights, surrogate_type),
        'false_positives_upper_bound':
        false_positives_upper_bound(labels, logits, weights, surrogate_type)
    }

    return loss, other_outputs


def precision_at_recall_loss(labels,
                             logits,
                             target_recall,
                             weights=1.0,
                             dual_rate_factor=0.1,
                             label_priors=None,
                             surrogate_type='xent',
                             lambdas_initializer=tf.constant_initializer(1.0),
                             reuse=None,
                             variables_collections=None,
                             trainable=True,
                             scope=None):
  """Computes precision at recall loss.
  The loss is based on a surrogate of the form
     wt * loss(-) + lambdas * (pi * (b - 1) + wt * loss(+))
  where:
  - loss(-) is the cross-entropy loss on the negative examples
  - loss(+) is the cross-entropy loss on the positive examples
  - wt is a scalar or tensor of per-example weights
  - b is the target recall
  - pi is the label_priors.
  The per-example weights change not only the coefficients of individual
  training examples, but how the examples are counted toward the constraint.
  If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_recall: The recall at which to compute the loss. Can be a floating
      point value between 0 and 1 for a single target recall value, or a
      `Tensor` of shape [num_labels] holding each label's target recall value.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
    lambdas_initializer: An initializer for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.
  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.
  """
  with tf.variable_scope(scope, 'precision_at_recall', [logits, labels, label_priors], reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(labels, logits, weights)
    num_labels = losses_utils.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_recall = losses_utils.convert_and_cast(target_recall, 'target_recall', logits.dtype)
    dual_rate_factor = losses_utils.convert_and_cast(dual_rate_factor, 'dual_rate_factor',
                                                     logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(label_priors, labels, weights, variables_collections)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * losses_utils.weighted_surrogate_loss(
        labels, logits, surrogate_type, positive_weights=lambdas, negative_weights=1.0)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * label_priors * (target_recall - 1.0) * maybe_log2
    loss = tf.reshape(weighted_loss + lambda_term, original_shape)
    other_outputs = {
        'lambdas':
        lambdas_variable,
        'label_priors':
        label_priors,
        'true_positives_lower_bound':
        true_positives_lower_bound(labels, logits, weights, surrogate_type),
        'false_positives_upper_bound':
        false_positives_upper_bound(labels, logits, weights, surrogate_type)
    }

    return loss, other_outputs


def false_positive_rate_at_true_positive_rate_loss(labels,
                                                   logits,
                                                   target_rate,
                                                   weights=1.0,
                                                   dual_rate_factor=0.1,
                                                   label_priors=None,
                                                   surrogate_type='xent',
                                                   lambdas_initializer=tf.constant_initializer(1.0),
                                                   reuse=None,
                                                   variables_collections=None,
                                                   trainable=True,
                                                   scope=None):
  """Computes false positive rate at true positive rate loss.
  Note that `true positive rate` is a synonym for Recall, and that minimizing
  the false positive rate and maximizing precision are equivalent for a fixed
  Recall. Therefore, this function is identical to precision_at_recall_loss.
  The per-example weights change not only the coefficients of individual
  training examples, but how the examples are counted toward the constraint.
  If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_rate: The true positive rate at which to compute the loss. Can be a
      floating point value between 0 and 1 for a single true positive rate, or
      a `Tensor` of shape [num_labels] holding each label's true positive rate.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions. 'xent' will use the cross-entropy
      loss surrogate, and 'hinge' will use the hinge loss.
    lambdas_initializer: An initializer op for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.
  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.
  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  return precision_at_recall_loss(
      labels=labels,
      logits=logits,
      target_recall=target_rate,
      weights=weights,
      dual_rate_factor=dual_rate_factor,
      label_priors=label_priors,
      surrogate_type=surrogate_type,
      lambdas_initializer=lambdas_initializer,
      reuse=reuse,
      variables_collections=variables_collections,
      trainable=trainable,
      scope=scope)


def true_positive_rate_at_false_positive_rate_loss(labels,
                                                   logits,
                                                   target_rate,
                                                   weights=1.0,
                                                   dual_rate_factor=0.1,
                                                   label_priors=None,
                                                   surrogate_type='xent',
                                                   lambdas_initializer=tf.constant_initializer(1.0),
                                                   reuse=None,
                                                   variables_collections=None,
                                                   trainable=True,
                                                   scope=None):
  """Computes true positive rate at false positive rate loss.
  The loss is based on a surrogate of the form
      wt * loss(+) + lambdas * (wt * loss(-) - r * (1 - pi))
  where:
  - loss(-) is the loss on the negative examples
  - loss(+) is the loss on the positive examples
  - wt is a scalar or tensor of per-example weights
  - r is the target rate
  - pi is the label_priors.
  The per-example weights change not only the coefficients of individual
  training examples, but how the examples are counted toward the constraint.
  If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_rate: The false positive rate at which to compute the loss. Can be a
      floating point value between 0 and 1 for a single false positive rate, or
      a `Tensor` of shape [num_labels] holding each label's false positive rate.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions. 'xent' will use the cross-entropy
      loss surrogate, and 'hinge' will use the hinge loss.
    lambdas_initializer: An initializer op for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.
  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.
  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  with tf.variable_scope(scope, 'tpr_at_fpr', [labels, logits, label_priors], reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(labels, logits, weights)
    num_labels = losses_utils.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_rate = losses_utils.convert_and_cast(target_rate, 'target_rate', logits.dtype)
    dual_rate_factor = losses_utils.convert_and_cast(dual_rate_factor, 'dual_rate_factor',
                                                     logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(label_priors, labels, weights, variables_collections)

    # Loss op and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * losses_utils.weighted_surrogate_loss(
        labels,
        logits,
        surrogate_type=surrogate_type,
        positive_weights=1.0,
        negative_weights=lambdas)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * target_rate * (1.0 - label_priors) * maybe_log2
    loss = tf.reshape(weighted_loss - lambda_term, original_shape)
    other_outputs = {
        'lambdas':
        lambdas_variable,
        'label_priors':
        label_priors,
        'true_positives_lower_bound':
        true_positives_lower_bound(labels, logits, weights, surrogate_type),
        'false_positives_upper_bound':
        false_positives_upper_bound(labels, logits, weights, surrogate_type)
    }

  return loss, other_outputs


def _prepare_labels_logits_weights(labels, logits, weights):
  """Validates labels, logits, and weights.
  Converts inputs to tensors, checks shape compatibility, and casts dtype if
  necessary.
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    weights: Either `None` or a `Tensor` with shape broadcastable to `logits`.
  Returns:
    labels: Same as `labels` arg after possible conversion to tensor, cast, and
      reshape.
    logits: Same as `logits` arg after possible conversion to tensor and
      reshape.
    weights: Same as `weights` arg after possible conversion, cast, and reshape.
    original_shape: Shape of `labels` and `logits` before reshape.
  Raises:
    ValueError: If `labels` and `logits` do not have the same shape.
  """
  # Convert `labels` and `logits` to Tensors and standardize dtypes.
  logits = tf.convert_to_tensor(logits, name='logits')
  labels = losses_utils.convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
  weights = losses_utils.convert_and_cast(weights, 'weights', logits.dtype.base_dtype)

  try:
    labels.get_shape().merge_with(logits.get_shape())
  except ValueError:
    raise ValueError('logits and labels must have the same shape (%s vs %s)' % (logits.get_shape(),
                                                                                labels.get_shape()))

  original_shape = labels.get_shape().as_list()
  if labels.get_shape().ndims > 0:
    original_shape[0] = -1
  if labels.get_shape().ndims <= 1:
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.reshape(logits, [-1, 1])

  if weights.get_shape().ndims == 1:
    # Weights has shape [batch_size]. Reshape to [batch_size, 1].
    weights = tf.reshape(weights, [-1, 1])
  if weights.get_shape().ndims == 0:
    # Weights is a scalar. Change shape of weights to match logits.
    weights *= tf.ones_like(logits)

  return labels, logits, weights, original_shape


def _range_to_anchors_and_delta(precision_range, num_anchors, dtype):
  """Calculates anchor points from precision range.
  Args:
    precision_range: As required in precision_recall_auc_loss.
    num_anchors: int, number of equally spaced anchor points.
    dtype: Data type of returned tensors.
  Returns:
    precision_values: A `Tensor` of data type dtype with equally spaced values
      in the interval precision_range.
    delta: The spacing between the values in precision_values.
  Raises:
    ValueError: If precision_range is invalid.
  """
  # Validate precision_range.
  if not 0 <= precision_range[0] <= precision_range[-1] <= 1:
    raise ValueError(
        'precision values must obey 0 <= %f <= %f <= 1' % (precision_range[0], precision_range[-1]))
  if not 0 < len(precision_range) < 3:
    raise ValueError('length of precision_range (%d) must be 1 or 2' % len(precision_range))

  # Sets precision_values uniformly between min_precision and max_precision.
  values = np.linspace(start=precision_range[0], stop=precision_range[1], num=num_anchors + 2)[1:-1]
  precision_values = losses_utils.convert_and_cast(values, 'precision_values', dtype)
  delta = losses_utils.convert_and_cast(values[0] - precision_range[0], 'delta', dtype)
  # Makes precision_values [1, 1, num_anchors].
  precision_values = losses_utils.expand_outer(precision_values, 3)
  return precision_values, delta


def _create_dual_variable(name, shape, dtype, initializer, collections, trainable, dual_rate_factor):
  """Creates a new dual variable.
  Dual variables are required to be nonnegative. If trainable, their gradient
  is reversed so that they are maximized (rather than minimized) by the
  optimizer.
  Args:
    name: A string, the name for the new variable.
    shape: Shape of the new variable.
    dtype: Data type for the new variable.
    initializer: Initializer for the new variable.
    collections: List of graph collections keys. The new variable is added to
      these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
    trainable: If `True`, the default, also adds the variable to the graph
      collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
      the default list of variables to use by the `Optimizer` classes.
    dual_rate_factor: A floating point value or `Tensor`. The learning rate for
      the dual variable is scaled by this factor.
  Returns:
    dual_value: An op that computes the absolute value of the dual variable
      and reverses its gradient.
    dual_variable: The underlying variable itself.
  """
  # We disable partitioning while constructing dual variables because they will
  # be updated with assign, which is not available for partitioned variables.
  partitioner = tf.get_variable_scope().partitioner
  try:
    tf.get_variable_scope().set_partitioner(None)
    dual_variable = tf.contrib.framework.model_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        collections=collections,
        trainable=trainable)
  finally:
    tf.get_variable_scope().set_partitioner(partitioner)
  # Using the absolute value enforces nonnegativity.
  dual_value = tf.abs(dual_variable)

  if trainable:
    # To reverse the gradient on the dual variable, multiply the gradient by
    # -dual_rate_factor
    dual_value = (tf.stop_gradient(
        (1.0 + dual_rate_factor) * dual_value) - dual_rate_factor * dual_value)
  return dual_value, dual_variable


def maybe_create_label_priors(label_priors, labels, weights, variables_collections):
  """Creates moving average ops to track label priors, if necessary.
  Args:
    label_priors: As required in e.g. precision_recall_auc_loss.
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    weights: As required in e.g. precision_recall_auc_loss.
    variables_collections: Optional list of collections for the variables, if
      any must be created.
  Returns:
    label_priors: A Tensor of shape [num_labels] consisting of the
      weighted label priors, after updating with moving average ops if created.
  """
  if label_priors is not None:
    label_priors = losses_utils.convert_and_cast(
        label_priors, name='label_priors', dtype=labels.dtype.base_dtype)
    return tf.squeeze(label_priors)

  label_priors = losses_utils.build_label_priors(
      labels, weights, variables_collections=variables_collections)
  return label_priors


def true_positives_lower_bound(labels, logits, weights, surrogate_type):
  """Calculate a lower bound on the number of true positives.
  This lower bound on the number of true positives given `logits` and `labels`
  is the same one used in the global objectives loss functions.
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` of shape [batch_size, num_labels] or
      [batch_size, num_labels, num_anchors]. If the third dimension is present,
      the lower bound is computed on each slice [:, :, k] independently.
    weights: Per-example loss coefficients, with shape broadcast-compatible with
        that of `labels`.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
  Returns:
    A `Tensor` of shape [num_labels] or [num_labels, num_anchors].
  """
  maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
  maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
  if logits.get_shape().ndims == 3 and labels.get_shape().ndims < 3:
    labels = tf.expand_dims(labels, 2)
  loss_on_positives = losses_utils.weighted_surrogate_loss(
      labels, logits, surrogate_type, negative_weights=0.0) / maybe_log2
  return tf.reduce_sum(weights * (labels - loss_on_positives), 0)


def false_positives_upper_bound(labels, logits, weights, surrogate_type):
  """Calculate an upper bound on the number of false positives.
  This upper bound on the number of false positives given `logits` and `labels`
  is the same one used in the global objectives loss functions.
  Args:
    labels: A `Tensor` of shape [batch_size, num_labels]
    logits: A `Tensor` of shape [batch_size, num_labels]  or
      [batch_size, num_labels, num_anchors]. If the third dimension is present,
      the lower bound is computed on each slice [:, :, k] independently.
    weights: Per-example loss coefficients, with shape broadcast-compatible with
        that of `labels`.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
  Returns:
    A `Tensor` of shape [num_labels] or [num_labels, num_anchors].
  """
  maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
  maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
  loss_on_negatives = losses_utils.weighted_surrogate_loss(
      labels, logits, surrogate_type, positive_weights=0.0) / maybe_log2
  return tf.reduce_sum(weights * loss_on_negatives, 0)
