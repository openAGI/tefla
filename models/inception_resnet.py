from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import conv2d, fully_connected, max_pool, relu, repeat
from tefla.core.layers import input, softmax, dropout, global_avg_pool, avg_pool_2d, batch_norm_tf as batch_norm


# Block with 3x3 and 5x5 filters
def block35(net, scale=0.17, name='block35', **kwargs):
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    activation = kwargs.get('activation')
    with tf.variable_scope(name, 'block35'):
        with tf.variable_scope('Branch_0'):
            tower_conv = conv2d(net, 32, filter_size=(1, 1),
                                name='Conv2d_1x1', **kwargs)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = conv2d(net, 32, filter_size=(
                1, 1), name='Conv2d_0a_1x1', **kwargs)
            tower_conv1_1 = conv2d(
                tower_conv1_0, 32, name='Conv2d_0b_3x3', **kwargs)
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = conv2d(net, 32, filter_size=(
                1, 1), name='Conv2d_0a_1x1', **kwargs)
            tower_conv2_1 = conv2d(
                tower_conv2_0, 48, name='Conv2d_0b_3x3', **kwargs)
            tower_conv2_2 = conv2d(
                tower_conv2_1, 64, name='Conv2d_0c_3x3', **kwargs)
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = conv2d(mixed, net.get_shape()[3], is_training, reuse, filter_size=(
            1, 1), batch_norm=None, activation=None, name='Conv2d_1x1')
        net += scale * up
        if activation:
            net = activation(net, reuse)
    return net


# Block with 1x7 and 7x1 filters
def block17(net, scale=1.0, name='block17', **kwargs):
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    activation = kwargs.get('activation')
    with tf.variable_scope(name, 'block17'):
        with tf.variable_scope('Branch_0'):
            tower_conv = conv2d(net, 192, filter_size=(
                1, 1), name='Conv2d_1x1', **kwargs)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = conv2d(net, 128, filter_size=(
                1, 1), name='Conv2d_0a_1x1', **kwargs)
            tower_conv1_1 = conv2d(tower_conv1_0, 160, filter_size=(
                1, 7), name='Conv2d_0b_1x7', **kwargs)
            tower_conv1_2 = conv2d(tower_conv1_1, 192, filter_size=(
                7, 1), name='Conv2d_0c_7x1', **kwargs)
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = conv2d(mixed, net.get_shape()[3], is_training, reuse, filter_size=(
            1, 1), batch_norm=None, activation=None, name='Conv2d_1x1')
        net += scale * up
        if activation:
            net = activation(net, reuse)
    return net


# Block with 1x3 and 3x1 filters
def block8(net, scale=1.0, name='block8', **kwargs):
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    activation = kwargs.get('activation')
    with tf.variable_scope(name, 'block8'):
        with tf.variable_scope('Branch_0'):
            tower_conv = conv2d(net, 192, filter_size=(
                1, 1), name='Conv2d_1x1', **kwargs)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = conv2d(net, 192, filter_size=(
                1, 1), name='Conv2d_0a_1x1', **kwargs)
            tower_conv1_1 = conv2d(tower_conv1_0, 224, filter_size=(
                1, 3), name='Conv2d_0b_1x3', **kwargs)
            tower_conv1_2 = conv2d(tower_conv1_1, 256, filter_size=(
                3, 1), name='Conv2d_0c_3x1', **kwargs)
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = conv2d(mixed, net.get_shape()[3], is_training, reuse, filter_size=(
            1, 1), batch_norm=None, activation=None, name='Conv2d_1x1')
        net += scale * up
        if activation:
            net = activation(net, reuse)
    return net


# sizes - (width, height)
image_size = (339, 339)
crop_size = (299, 299)


def model(inputs, is_training, reuse, num_classes=5, drop_prob=0.2, name='InceptionResnetV2'):
    common_args = common_layer_args(is_training, reuse)
    rest_conv_params = make_args(
        use_bias=False, batch_norm=batch_norm, activation=relu, **common_args)
    conv_params_no_bias = make_args(
        use_bias=False, batch_norm=batch_norm, activation=relu, **common_args)
    conv_params = make_args(
        use_bias=True, batch_norm=batch_norm, activation=None, **common_args)
    rest_logit_params = make_args(activation=None, **common_args)
    rest_pool_params = make_args(padding='SAME', **common_args)
    rest_dropout_params = make_args(drop_p=drop_prob, **common_args)

    # inputs = input((None, crop_size[1], crop_size[0], 3), **common_args)

    with tf.variable_scope(name, 'InceptionResnetV2'):
        net = conv2d(inputs, 32, stride=(
            2, 2), name='Conv2d_1a_3x3', **conv_params_no_bias)
        net = conv2d(net, 32, name='Conv2d_2a_3x3', **conv_params_no_bias)
        # 112 x 112
        net = conv2d(net, 64, name='Conv2d_2b_3x3', **rest_conv_params)
        # 112 x 112
        net = max_pool(net, name='MaxPool_3a_3x3', **rest_pool_params)
        # 64 x 64
        net = conv2d(net, 80, filter_size=(1, 1),
                     name='Conv2d_3b_1x1', **rest_conv_params)
        # 64 x 64
        net = conv2d(net, 192, name='Conv2d_4a_3x3', **rest_conv_params)
        # 64 x 64
        net = max_pool(net, stride=(2, 2),
                       name='maxpool_5a_3x3', **rest_pool_params)

        # 32 x 32
        with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
                tower_conv = conv2d(net, 96, filter_size=(
                    1, 1), name='Conv2d_1x1', **rest_conv_params)
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = conv2d(net, 48, filter_size=(
                    1, 1), name='Conv2d_0a_1x1', **rest_conv_params)
                tower_conv1_1 = conv2d(tower_conv1_0, 64, filter_size=(
                    5, 5), name='Conv2d_0b_5x5', **rest_conv_params)
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = conv2d(net, 64, filter_size=(
                    1, 1), name='Conv2d_0a_1x1', **rest_conv_params)
                tower_conv2_1 = conv2d(
                    tower_conv2_0, 96, name='Conv2d_0b_3x3', **rest_conv_params)
                tower_conv2_2 = conv2d(
                    tower_conv2_1, 96, name='Conv2d_0c_3x3', **rest_conv_params)
            with tf.variable_scope('Branch_3'):
                tower_pool = avg_pool_2d(net, stride=(
                    1, 1), name='avgpool_0a_3x3', **rest_pool_params)
                tower_pool_1 = conv2d(tower_pool, 64, filter_size=(
                    1, 1), name='Conv2d_0b_1x1', **rest_conv_params)
            net = tf.concat([tower_conv, tower_conv1_1,
                             tower_conv2_2, tower_pool_1], 3)
        with tf.variable_scope('Repeat'):
            for i in range(1, 11):
                net = block35(net, name='block35_' +
                              str(i), scale=0.17, **conv_params_no_bias)

        # 32 x 32
        with tf.variable_scope('Mixed_6a'):
            with tf.variable_scope('Branch_0'):
                tower_conv = conv2d(net, 384, stride=(
                    2, 2), name='Conv2d_1a_3x3', **rest_conv_params)
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = conv2d(net, 256, filter_size=(
                    1, 1), name='Conv2d_0a_1x1', **rest_conv_params)
                tower_conv1_1 = conv2d(
                    tower_conv1_0, 256, name='Conv2d_0b_3x3', **rest_conv_params)
                tower_conv1_2 = conv2d(tower_conv1_1, 384, stride=(
                    2, 2), name='Conv2d_1a_3x3', **rest_conv_params)
            with tf.variable_scope('Branch_2'):
                tower_pool = max_pool(
                    net, name='maxpool_1a_3x3', **rest_pool_params)
            net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

        with tf.variable_scope('Repeat_1'):
            for i in range(1, 21):
                net = block17(net, name='block17_' +
                              str(i), scale=0.10, **conv_params_no_bias)

        with tf.variable_scope('Mixed_7a'):
            with tf.variable_scope('Branch_0'):
                tower_conv = conv2d(net, 256, filter_size=(
                    1, 1), name='Conv2d_0a_1x1', **rest_conv_params)
                tower_conv_1 = conv2d(tower_conv, 384, stride=(
                    2, 2), name='Conv2d_1a_3x3', **rest_conv_params)
            with tf.variable_scope('Branch_1'):
                tower_conv1 = conv2d(net, 256, filter_size=(
                    1, 1), name='Conv2d_0a_1x1', **rest_conv_params)
                tower_conv1_1 = conv2d(tower_conv1, 288, stride=(
                    2, 2), name='Conv2d_1a_3x3', **rest_conv_params)
            with tf.variable_scope('Branch_2'):
                tower_conv2 = conv2d(net, 256, filter_size=(
                    1, 1), name='Conv2d_0a_1x1', **rest_conv_params)
                tower_conv2_1 = conv2d(
                    tower_conv2, 288, name='Conv2d_0b_3x3', **rest_conv_params)
                tower_conv2_2 = conv2d(tower_conv2_1, 320, stride=(
                    2, 2), name='Conv2d_1a_3x3', **rest_conv_params)
            with tf.variable_scope('Branch_3'):
                tower_pool = max_pool(
                    net, name='maxpool_1a_3x3', **rest_pool_params)
            net = tf.concat([tower_conv_1, tower_conv1_1,
                             tower_conv2_2, tower_pool], 3)
        # 8 x 8
        with tf.variable_scope('Repeat_2'):
            for i in range(1, 10):
                net = block8(net, name='block8_' + str(i), scale=0.20,
                             **conv_params_no_bias)
        net = block8(net, name='Block8', **conv_params_no_bias)

        net = conv2d(net, 1536, filter_size=(1, 1),
                     name='Conv2d_7b_1x1', **rest_conv_params)

        with tf.variable_scope('Logits'):
            net = global_avg_pool(
                net, name='avgpool_1a_8x8')
            net = dropout(net,
                          name='dropout', **rest_dropout_params)
            logits = fully_connected(
                net, num_classes, name='Logits', **rest_logit_params)
            predictions = softmax(logits, name='Predictions',
                                  **common_args)

    return end_points(is_training)
