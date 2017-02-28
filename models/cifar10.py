from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.core import initializers as initz
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import dropout, batch_norm_tf as batch_norm
from tefla.core.layers import input, conv2d, max_pool, prelu, softmax, global_avg_pool, separable_conv2d
from tefla.utils import util

# sizes - (width, height)
image_size = (32, 32)
crop_size = (28, 28)


def fire_module(inputs, squeeze_depth, expand_depth, name=None, **kwargs):
    reuse = kwargs.get('reuse')
    with tf.variable_scope(name, 'fire', [inputs], reuse=reuse):
        net = squeeze(inputs, squeeze_depth, **kwargs)
        outputs = expand(net, expand_depth, **kwargs)
        return outputs


def squeeze(inputs, num_outputs, **kwargs):
    return conv2d(inputs, num_outputs, filter_size=(1, 1), stride=(1, 1), name='squeeze', batch_norm=True, activation=prelu, **kwargs)


def expand(inputs, num_outputs, **kwargs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, filter_size=(
            1, 1), stride=(1, 1), name='1x1', **kwargs)
        e3x3 = conv2d(inputs, num_outputs,
                      filter_size=(3, 3), name='3x3', **kwargs)
    return tf.concat([e1x1, e3x3], 3)


def bottleneck_simple(inputs, squeeze_depth, expand_depth, name=None, **kwargs):
    """Bottleneck residual unit variant with BN before convolutions.
    Note: inputs depth same as outputh depth

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      squeeze_depth: The depth of the squeeze layers.
      exapnd_depth: The depth of the exapnd layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      name: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    depth_in = util.last_dimension(inputs.get_shape(), min_rank=4)
    assert depth_in == 2 * expand_depth, 'input depth should be twice of exapnd_depth'
    with tf.variable_scope(name, 'bottleneck_simple', [inputs]):
        preact = batch_norm(inputs,
                            name='preact', is_training=is_training, reuse=reuse)
        preact = prelu(preact, reuse, name='prelu_start')
        residual = squeeze(preact, squeeze_depth, **kwargs)
        residual = expand(residual, expand_depth, **kwargs)

        output = prelu(inputs + residual, reuse, name='prelu_residual')
        return output


def model(inputs, is_training, reuse, num_classes=10, dropout_keep_prob=0.5):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    conv_args_fm = make_args(w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    pool_args = make_args(padding='SAME', **common_args)
    with tf.variable_scope('squeezenet', values=[inputs]):
        net = separable_conv2d(inputs, 256, stride=(2, 2),
                               name='conv1', **conv_args)
        # net = conv2d(inputs, 96, stride=(2, 2), name='conv1', **conv_args)
        net = max_pool(net, name='maxpool1', **pool_args)
        net = fire_module(net, 16, 64, name='fire2', **conv_args_fm)
        net = bottleneck_simple(net, 16, 64, name='fire3', **conv_args_fm)
        net = batch_norm(net, activation_fn=tf.nn.relu,
                         name='fire3_bn', is_training=is_training, reuse=reuse)
        net = fire_module(net, 32, 128, name='fire4', **conv_args_fm)
        net = max_pool(net, name='maxpool4', **pool_args)
        net = bottleneck_simple(net, 32, 128, name='fire5', **conv_args_fm)
        net = batch_norm(net, activation_fn=tf.nn.relu,
                         name='fire5_bn', is_training=is_training, reuse=reuse)
        net = fire_module(net, 48, 192, name='fire6', **conv_args_fm)
        net = bottleneck_simple(net, 48, 192, name='fire7', **conv_args_fm)
        net = batch_norm(net, activation_fn=tf.nn.relu,
                         name='fire7_bn', is_training=is_training, reuse=reuse)
        net = fire_module(net, 64, 256, name='fire8', **conv_args_fm)
        net = max_pool(net,  name='maxpool8', **pool_args)
        net = dropout(net, drop_p=1 - dropout_keep_prob,
                      name='dropout6', **common_args)
        net = conv2d(net, num_classes, filter_size=(
            1, 1), name='conv10', **conv_args_fm)
        logits = global_avg_pool(net, name='logits', **pool_args)
        predictions = softmax(logits, name='predictions', **common_args)
        return end_points(is_training)


def bottleneck(inputs, depth, squeeze_depth, expand_depth, stride, rate=1, name=None, **kwargs):
    """Bottleneck residual unit variant with BN before convolutions.

    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      squeeze_depth: The depth of the squeeze layers.
      exapnd_depth: The depth of the exapnd layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      name: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    with tf.variable_scope(name, 'bottleneck_v2', [inputs]):
        depth_in = util.last_dimension(inputs.get_shape(), min_rank=4)
        preact = batch_norm(inputs, activation_fn=tf.nn.relu,
                            name='preact', is_training=is_training, reuse=reuse)
        if depth == depth_in:
            shortcut = inputs
        else:
            shortcut = conv2d(preact, depth, is_training, reuse, filter_size=(1, 1), stride=(
                stride, stride), batch_norm=None, activation=None, name='shortcut')

        residual = fire_module(preact, squeeze_depth,
                               expand_depth, name='fire2', **kwargs)
        residual = conv2d(residual, depth, is_training, reuse, filter_size=(1, 1), stride=(1, 1),
                          batch_norm=None, activation=None, name='conv3')

        output = tf.nn.relu(shortcut + residual)
        return output
