"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.core import initializers as initz
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import dropout
from tefla.core.layers import input, conv2d, max_pool, prelu, softmax

# sizes - (width, height)
image_size = (256, 256)
crop_size = (224, 224)


def model(is_training, reuse,
          num_classes=1000,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='alexnet_v2'):
    """AlexNet version 2.

    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224. To use in fully
          convolutional mode, set spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.

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
    with tf.variable_scope(name, 'alexnet_v2', [inputs]):
        net = conv2d(inputs, 64, filter_size=(11, 11), stride=(4, 4),
                     name='conv1', **conv_args)
        net = max_pool(net, stride=(2, 2), name='pool1', **pool_args)
        net = conv2d(net, 192, filter_size=(5, 5), name='conv2', **conv_args)
        net = max_pool(net, stride=(2, 2), name='pool2', **pool_args)
        net = conv2d(net, 384, name='conv3', **conv_args)
        net = conv2d(net, 384, name='conv4', **conv_args)
        net = conv2d(net, 256, name='conv5', **conv_args)
        net = max_pool(net, stride=(2, 2), name='pool5', **pool_args)

        # Use conv2d instead of fully_connected layers.
        net = conv2d(net, 4096, filter_size=(5, 5),
                     name='fc6', **conv_args)
        net = dropout(net, drop_p=1 - dropout_keep_prob,
                      name='dropout6', **common_args)
        net = conv2d(net, 4096, filter_size=(1, 1), name='fc7', **conv_args)
        net = dropout(net, drop_p=1 - dropout_keep_prob,
                      name='dropout7', **common_args)
        logits = conv2d(net, num_classes, filter_size=(1, 1),
                        activation=None,
                        name='logits', **logit_args)

        if spatial_squeeze:
            logits = tf.squeeze(logits, [1, 2], name='fc8/squeezed')
        predictions = softmax(logits, name='predictions', **pred_args)
        return end_points(is_training)
