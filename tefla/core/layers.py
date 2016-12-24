from __future__ import division, print_function, absolute_import

from collections import namedtuple

import numpy as np
import tensorflow as tf
from tefla.core import initializers as initz
from tefla.utils import util as helper
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

rng = np.random.RandomState([2016, 6, 1])
NamedOutputs = namedtuple('NamedOutputs', ['name', 'outputs'])


def input(shape, name='inputs', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        inputs = tf.placeholder(tf.float32, shape=shape, name="input")
    return _collect_named_outputs(outputs_collections, name, inputs)


def fully_connected(x, n_output, is_training, reuse, trainable=True, w_init=initz.he_normal(), b_init=0.0,
                    w_regularizer=tf.nn.l2_loss, name='fc', batch_norm=None, batch_norm_args=None, activation=None,
                    outputs_collections=None, use_bias=True):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    if len(x.get_shape()) != 2:
        x = _flatten(x)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name, reuse=reuse):
        shape = [n_input, n_output] if hasattr(w_init, '__call__') else None
        W = tf.get_variable(
            name='W',
            shape=shape,
            dtype=tf.float32,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )
        output = tf.matmul(x, W)

        if use_bias:
            b = tf.get_variable(
                name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(b_init),
                trainable=trainable,
            )

            output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


def conv2d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1),
           padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
           name='conv2d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
           outputs_collections=None):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        shape = [filter_size[0], filter_size[1], x.get_shape()[-1], n_output_channels] if hasattr(w_init,
                                                                                                  '__call__') else None
        W = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.conv2d(
            input=x,
            filter=W,
            strides=[1, stride[0], stride[1], 1],
            padding=padding)

        if use_bias:
            if untie_biases:
                b = tf.get_variable(
                    name='b',
                    shape=output.get_shape()[1:],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable,
                )
                output = tf.add(output, b)
            else:
                b = tf.get_variable(
                    name='b',
                    shape=[n_output_channels],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable,
                )
                output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


def dilated_conv2d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(3, 3), dilation=1,
                   padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
                   name='dilated_conv2d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
                   outputs_collections=None):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        shape = [filter_size[0], filter_size[1], x.get_shape()[-1], n_output_channels] if hasattr(w_init,
                                                                                                  '__call__') else None
        W = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.atrous_conv2d(
            value=x,
            filters=W,
            rate=dilation,
            padding=padding)

        if use_bias:
            if untie_biases:
                b = tf.get_variable(
                    name='b',
                    shape=output.get_shape()[1:],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable,
                )
                output = tf.add(output, b)
            else:
                b = tf.get_variable(
                    name='b',
                    shape=[n_output_channels],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable,
                )
                output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


def separable_conv2d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1), dilation=1, depth_multiplier=8,
                     padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
                     name='separable_conv2d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
                     outputs_collections=None):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        depthwise_shape = [filter_size[0], filter_size[1], x.get_shape()[-1], depth_multiplier] if hasattr(w_init,
                                                                                                           '__call__') else None
        pointwise_shape = [1, 1, x.get_shape()[-1] * depth_multiplier, n_output_channels] if hasattr(w_init,
                                                                                                     '__call__') else None
        depthwise_W = tf.get_variable(
            name='depthwise_W',
            shape=depthwise_shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )
        pointwise_W = tf.get_variable(
            name='pointwise_W',
            shape=pointwise_shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.separable_conv2d(
            input=x,
            depthwise_filter=depthwise_W,
            pointwise_filter=pointwise_W,
            strides=[1, stride[0], stride[1], 1],
            padding=padding)

        if use_bias:
            if untie_biases:
                b = tf.get_variable(
                    name='b',
                    shape=output.get_shape()[1:],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable,
                )
                output = tf.add(output, b)
            else:
                b = tf.get_variable(
                    name='b',
                    shape=[n_output_channels],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable,
                )
                output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


def depthwise_conv2d(x, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1), depth_multiplier=8,
                     padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
                     name='depthwise_conv2d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
                     outputs_collections=None):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        shape = [filter_size[0], filter_size[1], x.get_shape()[-1], depth_multiplier] if hasattr(w_init,
                                                                                                 '__call__') else None
        W = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.depthwise_conv2d(
            input=x,
            filters=W,
            strides=[1, stride[0], stride[1], 1],
            padding=padding)

        if use_bias:
            if untie_biases:
                b = tf.get_variable(
                    name='b',
                    shape=output.get_shape()[1:],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable,
                )
                output = tf.add(output, b)
            else:
                b = tf.get_variable(
                    name='b',
                    shape=[x.get_shape()[-1] * depth_multiplier],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable,
                )
                output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


def upsample2d(input_, output_shape, is_training, reuse, filter_size=(5, 5), stride=(2, 2), init=initz.he_normal(seed=None),
               batch_norm=None, activation=None, name="deconv2d", use_bias=True, with_w=False, outputs_collections=None, **unused):
    input_shape = helper.get_input_shape(input_)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name or 'upsample2d', reuse=reuse):
        shape = [filter_size[0], filter_size[1], output_shape[-1], input_.get_shape()[-1]] if hasattr(init,
                                                                                                      '__call__') else None

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(name='W', shape=shape, initializer=init)

        output = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[
                                        1, stride[0], stride[1], 1])
        if use_bias:
            biases = tf.get_variable(
                'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            output = tf.reshape(tf.nn.bias_add(
                output, biases), output.get_shape())

        if batch_norm:
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, name='bn_upsample')

        if activation:
            output = activation(output, reuse=reuse)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)

        if with_w:
            return _collect_named_outputs(outputs_collections, name, output), w, biases
        else:
            return _collect_named_outputs(outputs_collections, name, output)


def _phase_shift(input_, r):
    bsize, a, b, c = helper.get_input_shape(input_)
    X = tf.reshape(input_, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))
    X = tf.split(1, a, X)
    X = tf.concat(2, [tf.squeeze(x) for x in X])
    X = tf.split(1, b, X)
    X = tf.concat(2, [tf.squeeze(x) for x in X])
    output = tf.reshape(X, (bsize, a * r, b * r, 1))
    return output


def subpixel2d(input_, r, color=False, name=None, outputs_collections=None, **unused):
    input_shape = helper.get_input_shape(input_)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name or "subpixel"):
        if color:
            inputc = tf.split(3, 3, input_)
            output = tf.concat(3, [_phase_shift(x, r) for x in inputc])
        else:
            output = _phase_shift(input_, r)
    return _collect_named_outputs(outputs_collections, name, output)


def highway_conv2d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1),
                   padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss,
                   name='highway_conv2d', activation=None, use_bias=True, outputs_collections=None):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        w_shape = [filter_size[0], filter_size[1], x.get_shape(
        )[-1], n_output_channels] if hasattr(w_init, '__call__') else None

        w_t_shape = [n_output_channels]
        b_shape = [n_output_channels]
        with tf.name_scope('main_gate'):
            W, b = helper.weight_bias(w_shape, b_shape, w_init=w_init,
                                      b_init=b_init, w_regularizer=w_regularizer, trainable=trainable)
        with tf.name_scope('transform_gate'):
            W_t, b_t = helper.weight_bias(
                w_t_shape, b_shape, w_init=w_init, b_init=b_init, w_regularizer=w_regularizer, trainable=trainable)
        output = tf.nn.conv2d(
            input=x,
            filter=W,
            strides=[1, stride[0], stride[1], 1],
            padding=padding)
        output = tf.add(output, b)
        H = activation(output, name='activation')
        T = tf.sigmoid(tf.matmul(output, W_t) + b_t, name='transform_gate')
        C = tf.sub(1.0, T, name="carry_gate")
        output = tf.add(tf.mul(H, T), tf.mul(output, C), name='output')

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)

    return output


def highway_fc2d(x, n_output, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1),
                 w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, name='highway_fc2d', activation=None, use_bias=True, outputs_collections=None):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    if len(x.get_shape()) != 2:
        x = _flatten(x)

    n_input = x.get_shape().as_list()[1]
    w_shape = [n_input, n_output] if hasattr(w_init, '__call__') else None
    b_shape = [n_output]
    with tf.variable_scope(name, reuse=reuse):
        with tf.name_scope('main_gate'):
            W, b = helper.weight_bias(w_shape, b_shape, w_init=w_init,
                                      b_init=b_init, w_regularizer=w_regularizer, trainable=trainable)
        with tf.name_scope('transform_gate'):
            W_t, b_t = helper.weight_bias(
                w_shape, b_shape, w_init=w_init, b_init=b_init, w_regularizer=w_regularizer, trainable=trainable)
        H = activation(tf.matmul(x, W) + b, name='activation')
        T = tf.sigmoid(tf.matmul(x, W_t) + b_t, name='transform_gate')
        C = tf.sub(1.0, T, name="carry_gate")
        output = tf.add(tf.mul(H, T), tf.mul(x, C), name='output')

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)

    return output


def max_pool(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', name='pool', outputs_collections=None, **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.nn.max_pool(
            value=x,
            ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding=padding,
        )
        return _collect_named_outputs(outputs_collections, name, output)


def fractional_pool(x, pooling_ratio=[1.0, 1.44, 1.73, 1.0], pseudo_random=None, determinastic=None, overlapping=None, name='fractional_pool', seed=None,
                    seed2=None, type='avg', outputs_collections=None, **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        if type == 'avg':
            output = tf.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random=pseudo_random,
                                               overlapping=overlapping, deterministic=determinastic, seed=seed, seed2=seed2, name=name)
        else:
            output = tf.nn.fractional_max_pool(value, pooling_ratio, pseudo_random=pseudo_random,
                                               overlapping=overlapping, deterministic=determinastic, seed=seed, seed2=seed2, name=name)
        return _collect_named_outputs(outputs_collections, name, output)


def rms_pool_2d(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', name='pool', epsilon=0.000000000001,
                outputs_collections=None, **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.nn.avg_pool(
            value=tf.square(x),
            ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding=padding,
        )
        output = tf.sqrt(output + epsilon)
        return _collect_named_outputs(outputs_collections, name, output)


def avg_pool_2d(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', name=None, outputs_collections=None, **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name or "pool"):
        output = tf.nn.avg_pool(
            value=x,
            ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding=padding,
            name="avg_pool")
        return _collect_named_outputs(outputs_collections, name, output)


def global_avg_pool(x, name="global_avg_pool", outputs_collections=None, **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.reduce_mean(x, [1, 2])
        return _collect_named_outputs(outputs_collections, name, output)


def feature_max_pool_1d(x, stride=2, name='pool', outputs_collections=None, **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 2, "Input Tensor shape must be 2-D"
    x = tf.reshape(x, (-1, input_shape[1] // stride, stride))
    with tf.name_scope(name):
        output = tf.reduce_max(
            input_tensor=x,
            reduction_indices=[2],
        )
        return _collect_named_outputs(outputs_collections, name, output)


def local_response_normalization(x, depth_radius=5, bias=1, alpha=1, beta=0.5, name='local_response_normalization', outputs_collections=None, **unused):
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.nn.local_response_normalization(
            input=x, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
        return _collect_named_outputs(outputs_collections, name, output)


def batch_norm_tf(x, name='bn', scale=False, updates_collections=None, **kwargs):
    return tf.contrib.layers.batch_norm(x, scope=name, scale=scale, updates_collections=updates_collections, **kwargs)


def batch_norm_lasagne(x, is_training, reuse, trainable=True, decay=0.9, epsilon=1e-4, name='bn',
                       updates_collections=tf.GraphKeys.UPDATE_OPS, outputs_collections=None):
    with tf.variable_scope(name, reuse=reuse):
        beta = tf.get_variable(
            name='beta',
            initializer=tf.constant(0.0, shape=[x.get_shape()[-1]]),
            trainable=trainable
        )
        gamma = tf.get_variable(
            name='gamma',
            initializer=tf.constant(1.0, shape=[x.get_shape()[-1]]),
            trainable=trainable
        )

        moving_mean = tf.get_variable(
            name='moving_mean',
            shape=[x.get_shape()[-1]],
            initializer=tf.zeros_initializer,
            trainable=False)

        moving_inv_std = tf.get_variable(
            name='moving_inv_std',
            shape=[x.get_shape()[-1]],
            initializer=tf.ones_initializer,
            trainable=False)

        def mean_inv_std_with_update():
            mean, variance = tf.nn.moments(
                x, [0, 1, 2], shift=moving_mean, name='bn-moments')
            inv_std = math_ops.rsqrt(variance + epsilon)
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            update_moving_inv_std = moving_averages.assign_moving_average(
                moving_inv_std, inv_std, decay)
            with tf.control_dependencies(
                    [update_moving_mean, update_moving_inv_std]):
                m, v = tf.identity(mean), tf.identity(inv_std)
                return m, v

        def mean_inv_std_with_pending_update():
            mean, variance = tf.nn.moments(
                x, [0, 1, 2], shift=moving_mean, name='bn-moments')
            inv_std = math_ops.rsqrt(variance + epsilon)
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            update_moving_inv_std = moving_averages.assign_moving_average(
                moving_inv_std, inv_std, decay)
            tf.add_to_collection(updates_collections, update_moving_mean)
            tf.add_to_collection(updates_collections, update_moving_inv_std)
            return mean, inv_std

        mean_inv_std_with_relevant_update = \
            mean_inv_std_with_pending_update if updates_collections is not None else mean_inv_std_with_update

        (mean, inv_std) = mean_inv_std_with_relevant_update(
        ) if is_training else (moving_mean, moving_inv_std)

        def _batch_normalization(x, mean, inv, offset, scale):
            with tf.name_scope(name, "batchnorm", [x, mean, inv, scale, offset]):
                if scale is not None:
                    inv *= scale
                return x * inv + (offset - mean * inv
                                  if offset is not None else -mean * inv)

        output = _batch_normalization(x, mean, inv_std, beta, gamma)
        return _collect_named_outputs(outputs_collections, name, output)


def prelu(x, reuse, trainable=True, name='prelu', outputs_collections=None):
    with tf.variable_scope(name, reuse=reuse):
        alphas = tf.get_variable(
            name='alpha',
            initializer=tf.constant(0.2, shape=[x.get_shape()[-1]]),
            trainable=trainable
        )

        output = tf.nn.relu(x) + tf.mul(alphas, (x - tf.abs(x))) * 0.5
        return _collect_named_outputs(outputs_collections, name, output)


def relu(x, name='relu', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.relu(x)
        return _collect_named_outputs(outputs_collections, name, output)


def relu6(x, name='relu6', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.relu6(x)
        return _collect_named_outputs(outputs_collections, name, output)


def softplus(x, name='softplus', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.softplus(x)
        return _collect_named_outputs(outputs_collections, name, output)


def crelu(x, name='crelu', outputs_collections=None, **unused):
    """
    Computes Concatenated ReLU.
    Concatenates a ReLU which selects only the positive part of the activation with
    a ReLU which selects only the negative part of the activation. Note that
    at as a result this non-linearity doubles the depth of the activations. Source: https://arxiv.org/abs/1603.05201
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.crelu(x)
        return _collect_named_outputs(outputs_collections, name, output)


def elu(x, name='elu', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.elu(x)
        return _collect_named_outputs(outputs_collections, name, output)


def leaky_relu(x, alpha=0.01, name='leaky_relu', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.relu(x) + tf.mul(alpha, (x - tf.abs(x))) * 0.5
        return _collect_named_outputs(outputs_collections, name, output)


def lrelu(x, leak=0.2, phase=0, name="lrelu", outputs_collections=None, **unused):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        output = f1 * x + f2 * abs(x)
        return _collect_named_outputs(outputs_collections, name, output)


def maxout(x, k=2, phase=0, name='maxout', outputs_collections=None, **unused):
    with tf.name_scope(name):
        shape = [int(e) for e in x.get_shape()]
        ax = len(shape)
        ch = shape[-1]
        assert ch % k == 0
        shape[-1] = ch / k
        shape.append(k)
        x = tf.reshape(x, shape)
        output = tf.reduce_max(x, ax)
        return _collect_named_outputs(outputs_collections, name, output)


def offset_maxout(x, k=2, phase=0, name='maxout', outputs_collections=None, **unused):
    with tf.name_scope(name):
        shape = [int(e) for e in x.get_shape()]
        ax = len(shape)
        ch = shape[-1]
        assert ch % k == 0
        shape[-1] = ch / k
        shape.append(k)
        x = tf.reshape(x, shape)
        ofs = rng.randn(1000, k).max(axis=1).mean()
        output = tf.reduce_max(x, ax) - ofs
        return _collect_named_outputs(outputs_collections, name, output)


def softmax(x, name='softmax', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.softmax(x)
        return _collect_named_outputs(outputs_collections, name, output)


def dropout(x, is_training, drop_p=0.5, name='dropout', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        keep_p = 1. - drop_p
        if is_training:
            output = tf.nn.dropout(x, keep_p, seed=None)
            return _collect_named_outputs(outputs_collections, name, output)
        else:
            return _collect_named_outputs(outputs_collections, name, x)


def _flatten(x, name='flatten'):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    with tf.name_scope(name):
        dims = int(np.prod(input_shape[1:]))
        flattened = tf.reshape(x, [-1, dims])
        return flattened


def repeat(inputs, repetitions, layer, name='repeat', outputs_collections=None, *args, **kwargs):
    with tf.variable_scope(name, 'Repeat'):
        inputs = tf.convert_to_tensor(inputs)
        if name is None:
            if hasattr(layer, '__name__'):
                name = layer.__name__
            elif hasattr(layer, 'func') and hasattr(layer.func, '__name__'):
                # In case layer is a functools.partial.
                name = layer.func.__name__
            else:
                name = 'repeat'
        outputs = inputs
        for i in range(repetitions):
            new_name = name + '_' + str(i + 1)
            outputs = layer(outputs, name=new_name, *args, **kwargs)
            tf.add_to_collection(outputs_collections,
                                 NamedOutputs(new_name, outputs))
        return outputs


def merge(tensors_list, mode, axis=1, name='merge', outputs_collections=None, **kwargs):
    assert len(tensors_list) > 1, "Merge required 2 or more tensors."

    with tf.name_scope(name):
        tensors = [l for l in tensors_list]
        if mode == 'concat':
            output = tf.concat(axis, tensors)
        elif mode == 'elemwise_sum':
            output = tensors[0]
            for i in range(1, len(tensors)):
                output = tf.add(output, tensors[i])
        elif mode == 'elemwise_mul':
            output = tensors[0]
            for i in range(1, len(tensors)):
                output = tf.mul(output, tensors[i])
        elif mode == 'sum':
            output = tf.reduce_sum(tf.concat(axis, tensors), axis=axis)
        elif mode == 'mean':
            output = tf.reduce_mean(tf.concat(axis, tensors), axis=axis)
        elif mode == 'prod':
            output = tf.reduce_prod(tf.concat(axis, tensors), axis=axis)
        elif mode == 'max':
            output = tf.reduce_max(tf.concat(axis, tensors), axis=axis)
        elif mode == 'min':
            output = tf.reduce_min(tf.concat(axis, tensors), axis=axis)
        elif mode == 'and':
            output = tf.reduce_all(tf.concat(axis, tensors), axis=axis)
        elif mode == 'or':
            output = tf.reduce_any(tf.concat(axis, tensors), axis=axis)
        else:
            raise Exception("Unknown merge mode", str(mode))
        return _collect_named_outputs(outputs_collections, name, output)

    return output


def _collect_named_outputs(outputs_collections, name, output):
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, NamedOutputs(name, output))
    return output


def _check_unused(unused, name):
    allowed_keys = ['is_training', 'reuse', 'outputs_collections', 'trainable']
    helper.veryify_args(unused, allowed_keys,
                        'Layer "%s" got unexpected argument(s):' % name)
