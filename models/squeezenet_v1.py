from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.core import initializers as initz
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import dropout, batch_norm_tf as batch_norm
from tefla.core.layers import input, conv2d, max_pool, prelu, softmax, global_avg_pool
from tefla.utils import util

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
            1, 1), stride=(1, 1), name='1x1', **kwargs)
        e3x3 = conv2d(inputs, num_outputs,
                      filter_size=(3, 3), name='3x3', **kwargs)
    return tf.concat([e1x1, e3x3], 3)


def model(is_training, reuse, dropout_keep_prob=0.5):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
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
                      name='dropout6', **common_args)
        net = conv2d(net, 10, filter_size=(1, 1), name='conv10', **conv_args)
        logits = global_avg_pool(net, name='logits', **pool_args)
        predictions = softmax(logits, name='predictions', **common_args)
        return end_points(is_training)
