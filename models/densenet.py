from __future__ import division, print_function, absolute_import
import tensorflow as tf

from tefla.core import initializers as initz
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import dropout, rms_pool_2d, feature_max_pool_2d, global_avg_pool
from tefla.core.layers import input, conv2d, fully_connected, max_pool, prelu, softmax

# sizes - (width, height)
# image_size = (512, 512)
# crop_size = (448, 448)

image_size = (512, 512)
crop_size = (448, 448)


def conv_block(inputs, num_filters, drop_p=None, block_name='block', **kwargs):
    inter_filters = num_filters * 4
    x = conv2d(inputs, inter_filters, filter_size=(1, 1),
               name=block_name + "_conv1_1", **kwargs)
    if drop_p:
        is_training = kwargs.get('is_training')
        x = dropout(x, is_training, drop_p=drop_p, name=block_name +
                    "_conv_dropout")
    x = conv2d(x, num_filters, name=block_name + "_conv1_2", **kwargs)
    return x


def dense_block(inputs, num_filters, num_layers=4, growth_rate=32, block_name='dense', **kwargs):
    cum_inputs = inputs
    for i in range(0, num_layers):
        x = conv_block(cum_inputs, num_filters,
                       block_name=block_name + '_' + str(i))
        cum_inputs = tf.concat([cum_inputs, x], axis=3)
        num_filters += growth_rate
    return cum_inputs, num_filters


def trans_block(inputs, num_filters, drop_p=None, block_name='trans', **kwargs):
    x = conv2d(inputs, num_filters, filter_size=(1, 1),
               name=block_name + "_conv1_1", **kwargs)
    if drop_p:
        is_training = kwargs.get('is_training')
        x = dropout(x, is_training, drop_p=drop_p, name=block_name +
                    "_trans_dropout")
    x = rms_pool_2d(x, name=block_name + "rms_pool1", padding='SAME')
    return x


def model(is_training, reuse, input_size=image_size[0], drop_p_conv=0.0, drop_p_trans=0.0, n_filters=64, n_layers=[4, 6, 8, 8], num_classes=5):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, w_init=initz.he_normal(
        scale=1), untie_biases=True, **common_args)
    fc_args = make_args(
        activation=prelu, w_init=initz.he_normal(scale=1), **common_args)
    logit_args = make_args(
        activation=None, w_init=initz.he_normal(scale=1), **common_args)
    pred_args = make_args(
        activation=prelu, w_init=initz.he_normal(scale=1), **common_args)
    pool_args = make_args(padding='SAME', filter_size=(
        2, 2), stride=(2, 2), **common_args)

    x = input((None, crop_size[1], crop_size[0], 3), **common_args)
    x = conv2d(x, 48, filter_size=(7, 7), name="conv1", **conv_args)
    x = max_pool(x, name='pool1', **pool_args)
    x = conv2d(x, 64, name="conv2_1", **conv_args)
    x = conv2d(x, 64, name="conv2_2", **conv_args)
    x = max_pool(x, name='pool2', **pool_args)

    # 112
    for block_idx in range(3):
        x, n_filters = dense_block(
            x, n_filters, num_layers=n_layers[block_idx], drop_p=drop_p_conv, name='dense_' + str(block_idx))
        x = trans_block(x, n_filters, drop_p=drop_p,
                        block_name='trans_' + str(block_idx))

    x, n_filters = dense_block(
        x, n_filters, num_layers=n_layers[3], drop_p=drop_p_trans, name='dense_3')
    # 8
    x = global_avg_pool(x, name='avgpool_1a_8x8')
    logits = fully_connected(x, n_output=num_classes,
                             name="logits", **logit_args)

    predictions = softmax(logits, name='predictions', **common_args)
    return end_points(is_training)
