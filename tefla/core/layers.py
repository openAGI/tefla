from __future__ import division, print_function, absolute_import

from collections import namedtuple

import numpy as np
import tensorflow as tf
from tefla.core import initializers as initz
from tefla.utils import util as helper
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

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
                trainable=trainable
            )

            output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training, reuse=reuse, trainable=trainable, **batch_norm_args)

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
                    trainable=trainable
                )
                output = tf.add(output, b)
            else:
                b = tf.get_variable(
                    name='b',
                    shape=[n_output_channels],
                    initializer=tf.constant_initializer(b_init),
                    trainable=trainable
                )
                output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training, reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


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
            mean, variance = tf.nn.moments(x, [0, 1, 2], shift=moving_mean, name='bn-moments')
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
            mean, variance = tf.nn.moments(x, [0, 1, 2], shift=moving_mean, name='bn-moments')
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

        (mean, inv_std) = mean_inv_std_with_relevant_update() if is_training else (moving_mean, moving_inv_std)

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


def leaky_relu(x, alpha=0.01, name='leaky_relu', outputs_collections=None, **unused):
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.relu(x) + tf.mul(alpha, (x - tf.abs(x))) * 0.5
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
    with tf.variable_scope(name, 'Repeat', [inputs]):
        inputs = tf.convert_to_tensor(inputs)
        if name is None:
            if hasattr(layer, '__name__'):
                name = layer.__name__
            elif hasattr(layer, 'func') and hasattr(layer.func, '__name__'):
                name = layer.func.__name__  # In case layer is a functools.partial.
            else:
                name = 'repeat'
        outputs = inputs
        for i in range(repetitions):
            new_name = name + '_' + str(i + 1)
            outputs = layer(outputs, name=new_name, *args, **kwargs)
            tf.add_to_collection(outputs_collections, NamedOutputs(new_name, outputs))
        return outputs


def _collect_named_outputs(outputs_collections, name, output):
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, NamedOutputs(name, output))
    return output


def _check_unused(unused, name):
    allowed_keys = ['is_training', 'reuse', 'outputs_collections', 'trainable']
    helper.veryify_args(unused, allowed_keys, 'Layer "%s" got unexpected argument(s):' % name)
