"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.core import initializers as initz
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import dropout, repeat, prelu, relu, global_avg_pool
from tefla.core.layers import input, conv2d, max_pool, prelu, softmax

# sizes - (width, height)
image_size = (256, 256)
crop_size = (224, 224)


def vgg_a(is_training, reuse,
          num_classes=1000,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          name='vgg_a'):
    """Oxford Net VGG 11-Layers version A Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      name: Optional name for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    logit_args = make_args(
        activation=None, w_init=initz.he_normal(scale=1), **common_args)
    pred_args = make_args(
        activation=prelu, w_init=initz.he_normal(scale=1), **common_args)
    pool_args = make_args(padding='SAME', **common_args)

    inputs = input((None, crop_size[1], crop_size[0], 3), **common_args)
    with tf.variable_scope(name, 'vgg_a', [inputs]):
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        net = repeat(inputs, 1, conv2d,
                     64, filter_size=(3, 3), name='conv1', **conv_args)
        net = max_pool(net, name='pool1', **pool_args)
        net = repeat(net, 1, conv2d, 128, filter_size=(
            3, 3), name='conv2', **conv_args)
        net = max_pool(net, name='pool2', **pool_args)
        net = repeat(net, 2, conv2d, 256, filter_size=(
            3, 3), name='conv3', **conv_args)
        net = max_pool(net, name='pool3', **pool_args)
        net = repeat(net, 2, conv2d, 512, filter_size=(
            3, 3), name='conv4', **conv_args)
        net = max_pool(net, name='pool4', **pool_args)
        net = repeat(net, 2, conv2d, 512, filter_size=(
            3, 3), name='conv5', **conv_args)
        net = max_pool(net, name='pool5', **pool_args)
        # Use conv2d instead of fully_connected layers.
        net = conv2d(net, 4096, filter_size=(7, 7), name='fc6', **conv_args)
        net = dropout(net, drop_p=1 - dropout_keep_prob, is_training=is_training,
                      name='dropout6', **common_args)
        net = conv2d(net, 4096, filter_size=(1, 1), name='fc7', **conv_args)
        net = dropout(net, drop_p=1 - dropout_keep_prob, is_training=is_training,
                      name='dropout7', **common_args)
        logits = conv2d(net, num_classes, filter_size=(1, 1),
                        activation=None,
                        name='logits', **logit_args)
        if spatial_squeeze:
            logits = tf.squeeze(logits, [1, 2], name='logits/squeezed')
        predictions = softmax(logits, name='predictions', **pred_args)
        return end_points(is_training)


def vgg_16(is_training, reuse,
           num_classes=1000,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           name='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      name: Optional name for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    logit_args = make_args(
        activation=None, w_init=initz.he_normal(scale=1), **common_args)
    pred_args = make_args(
        activation=prelu, w_init=initz.he_normal(scale=1), **common_args)
    pool_args = make_args(padding='SAME', **common_args)
    inputs = input((None, crop_size[1], crop_size[0], 3), **common_args)
    with tf.variable_scope(name, 'vgg_16', [inputs]):
        net = repeat(inputs, 2, conv2d,
                     64, filter_size=(3, 3), name='conv1', **conv_args)
        net = max_pool(net, name='pool1', **pool_args)
        net = repeat(net, 2, conv2d, 128, filter_size=(
            3, 3), name='conv2', **conv_args)
        net = max_pool(net, name='pool2', **pool_args)
        net = repeat(net, 3, conv2d, 256, filter_size=(
            3, 3), name='conv3', **conv_args)
        net = max_pool(net, name='pool3', **pool_args)
        net = repeat(net, 3, conv2d, 512, filter_size=(
            3, 3), name='conv4', **conv_args)
        net = max_pool(net, name='pool4', **pool_args)
        net = repeat(net, 3, conv2d, 512, filter_size=(
            3, 3), name='conv5', **conv_args)
        net = max_pool(net, name='pool5', **pool_args)
        # Use conv2d instead of fully_connected layers.
        net = conv2d(net, 4096, filter_size=(7, 7), name='fc6', **conv_args)
        net = dropout(net, drop_p=1 - dropout_keep_prob, is_training=is_training,
                      name='dropout6', **common_args)
        net = conv2d(net, 4096, filter_size=(1, 1), name='fc7', **conv_args)
        net = dropout(net, drop_p=1 - dropout_keep_prob, is_training=is_training,
                      name='dropout7', **common_args)
        logits = conv2d(net, num_classes, filter_size=(1, 1),
                        activation=None,
                        name='logits', **logit_args)
        # Convert end_points_collection into a end_point dict.
        if spatial_squeeze:
            logits = tf.squeeze(logits, [1, 2], name='logits/squeezed')
        predictions = softmax(logits, name='predictions', **pred_args)
        return end_points(is_training)


def vgg_19(is_training, reuse,
           num_classes=1000,
           dropout_keep_prob=0.5,
           spatial_squeeze=False,
           name='vgg_19'):
    """Oxford Net VGG 19-Layers version E Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      name: Optional name for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=None, activation=relu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    logit_args = make_args(
        activation=None, w_init=initz.he_normal(scale=1), **common_args)
    pred_args = make_args(**common_args)
    pool_args = make_args(padding='SAME', filter_size=2, **common_args)
    inputs = input((None, crop_size[1], crop_size[0], 3), **common_args)
    with tf.variable_scope(name, 'vgg_19', [inputs]):
        net = repeat(inputs, 2, conv2d,
                     64, filter_size=(3, 3), name='conv1', **conv_args)
        net = max_pool(net, name='pool1', **pool_args)
        net = repeat(net, 2, conv2d, 128, filter_size=(
            3, 3), name='conv2', **conv_args)
        net = max_pool(net, name='pool2', **pool_args)
        net = repeat(net, 4, conv2d, 256, filter_size=(
            3, 3), name='conv3', **conv_args)
        net = max_pool(net, name='pool3', **pool_args)
        net = repeat(net, 4, conv2d, 512, filter_size=(
            3, 3), name='conv4', **conv_args)
        net = max_pool(net, name='pool4', **pool_args)
        net = repeat(net, 4, conv2d, 512, filter_size=(
            3, 3), name='conv5', **conv_args)
        net = max_pool(net, name='pool5', **pool_args)
        # Use conv2d instead of fully_connected layers.
        net = conv2d(net, 4096, filter_size=(7, 7),
                     name='fc6', **conv_args)
        net = dropout(net, drop_p=1 - dropout_keep_prob,
                      name='dropout6', **common_args)
        net = conv2d(net, 4096, filter_size=(1, 1), name='fc7', **conv_args)
        net = max_pool(net, filter_size=(7, 7), stride=7)
        net = dropout(net, drop_p=1 - dropout_keep_prob,
                      name='dropout7', **common_args)
        logits = conv2d(net, num_classes, filter_size=(1, 1),
                        name='fc8', **logit_args)
        if spatial_squeeze:
            logits = tf.squeeze(logits, [1, 2], name='logits')
        # logits = global_avg_pool(logits)
        predictions = softmax(logits, name='predictions', **pred_args)
        return end_points(is_training)
