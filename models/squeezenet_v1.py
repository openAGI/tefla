from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.core import initializers as initz
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import dropout, batch_norm_tf as batch_norm
from tefla.core.layers import input, conv2d, max_pool, prelu, softmax, global_avg_pool
from tefla.uitls import util

# sizes - (width, height)
image_size = (256, 256)
crop_size = (224, 224)


def fire_module(inputs, squeeze_depth, expand_depth, name=None, **kwargs):
    reuse = kwargs.get('reuse')
    with tf.variable_scope(name, 'fire', [inputs], reuse=reuse):
        net = squeeze(inputs, squeeze_depth, **kwargs)
        outputs = expand(net, expand_depth, **kwargs)
        return outputs


def squeeze(inputs, num_outputs, **kwargs):
    return conv2d(inputs, num_outputs, filter_size=(1, 1), stride=(1, 1), name='squeeze', **kwargs)


def expand(inputs, num_outputs, **kwargs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, filter_size=(
            1, 1), stride=1, name='1x1', **kwargs)
        e3x3 = conv2d(inputs, num_outputs,
                      filter_size=(3, 3), name='3x3', **kwargs)
    return tf.concat(3, [e1x1, e3x3])


def model_v1(is_training, reuse, dropout_keep_prob=0.5):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    pred_args = make_args(
        activation=prelu, w_init=initz.he_normal(scale=1), **common_args)
    pool_args = make_args(padding='SAME', **common_args)
    inputs = input((None, crop_size[1], crop_size[0], 3), **common_args)
    with tf.variable_scope('squeezenet', values=[inputs]):
        net = conv2d(inputs, 96, stride=(2, 2), name='conv1', **conv_args)
        net = max_pool(net, name='maxpool1', **pool_args)
        net = fire_module(net, 16, 64, name='fire2', **conv_args)
        net = fire_module(net, 16, 64, name='fire3', **conv_args)
        net = fire_module(net, 32, 128, name='fire4', **conv_args)
        net = max_pool(net, name='maxpool4', **pool_args)
        net = fire_module(net, 32, 128, name='fire5', **conv_args)
        net = fire_module(net, 48, 192, name='fire6', **conv_args)
        net = fire_module(net, 48, 192, name='fire7', **conv_args)
        net = fire_module(net, 64, 256, name='fire8', **conv_args)
        net = max_pool(net,  name='maxpool8', **pool_args)
        net = fire_module(net, 64, 256, name='fire9', **conv_args)
        # Reversed avg and conv layers per 'Network in Network'
        net = dropout(net, drop_p=1 - dropout_keep_prob,
                      scope='dropout6', **common_args)
        net = conv2d(net, 10, filter_size=(1, 1), name='conv10', **conv_args)
        logits = global_avg_pool(net, name='logits', **pool_args)
        predictions = softmax(logits, name='predictions', **pred_args)
        return end_points(is_training)


def bottleneck_simple(inputs, squeeze_depth, expand_depth, name=None, **kwargs):
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    with tf.variable_scope(name, 'bottleneck_simple', [inputs]):
        preact = batch_norm(inputs, activation_fn=tf.nn.relu,
                            name='preact', is_training=is_training, reuse=reuse)
        residual = squeeze(preact, squeeze_depth, **kwargs)
        kwargs.update({'batch_norm': None, 'activation': None})
        residual = expand(residual, expand_depth, **kwargs)

        output = tf.nn.relu(inputs + residual)
        return output


def model_v2(is_training, reuse, dropout_keep_prob=0.5):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    pred_args = make_args(
        activation=prelu, w_init=initz.he_normal(scale=1), **common_args)
    pool_args = make_args(padding='SAME', **common_args)
    inputs = input((None, crop_size[1], crop_size[0], 3), **common_args)
    with tf.variable_scope('squeezenet', values=[inputs]):
        net = conv2d(inputs, 96, stride=(2, 2), name='conv1', **conv_args)
        net = max_pool(net, name='maxpool1', **pool_args)
        net = fire_module(net, 16, 64, name='fire2', **conv_args)
        net = bottleneck_simple(net, 16, 64, name='fire3', **conv_args)
        net = batch_norm(net, activation_fn=tf.nn.relu,
                            name='preact', is_training=is_training, reuse=reuse)
        net = fire_module(net, 32, 128, name='fire4', **conv_args)
        net = max_pool(net, name='maxpool4', **pool_args)
        net = fire_module(net, 32, 128, name='fire5', **conv_args)
        net = fire_module(net, 48, 192, name='fire6', **conv_args)
        net = fire_module(net, 48, 192, name='fire7', **conv_args)
        net = fire_module(net, 64, 256, name='fire8', **conv_args)
        net = max_pool(net,  name='maxpool8', **pool_args)
        net = fire_module(net, 64, 256, name='fire9', **conv_args)
        # Reversed avg and conv layers per 'Network in Network'
        net = dropout(net, drop_p=1 - dropout_keep_prob,
                      scope='dropout6', **common_args)
        net = conv2d(net, 10, filter_size=(1, 1), name='conv10', **conv_args)
        logits = global_avg_pool(net, name='logits', **pool_args)
        predictions = softmax(logits, name='predictions', **pred_args)
        return end_points(is_training)



def bottleneck(inputs, depth, depth_bottleneck_1x1, depth_bottleneck_3x3, stride, rate=1, name=None, **kwargs):
    """Bottleneck residual unit variant with BN before convolutions.

    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck_1x1: The depth of the bottleneck layers.
      depth_bottleneck_3x3: The depth of the bottleneck layers.
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

        residual = fire_module(preact, depth_bottleneck_1x1, depth_bottleneck_3x3, name='fire2', **kwargs)
        residual = conv2d(residual, depth, is_training, reuse, filter_size=(1, 1), stride=(1, 1),
                          batch_norm=None, activation=None, name='conv3')

        output = tf.nn.relu(shortcut + residual)
        return output