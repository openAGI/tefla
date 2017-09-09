from __future__ import division, print_function, absolute_import

from collections import namedtuple

import numpy as np
import six
import numbers
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

from . import initializers as initz
from ..utils import util as helper
rng = np.random.RandomState([2016, 6, 1])
NamedOutputs = namedtuple('NamedOutputs', ['name', 'outputs'])


def input(shape, name='inputs', outputs_collections=None, **unused):
    """
    Define input layer.

    Args:
        shape: A `Tensor`, define the input shape
            e.g. for image input [batch_size, height, width, depth]
        name: A optional score/name for this op
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A placeholder for the input
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        inputs = tf.placeholder(tf.float32, shape=shape, name="input")
    return _collect_named_outputs(outputs_collections, name, inputs)


def register_to_collections(inputs, name=None, outputs_collections=None, **unused):
    """
    Add item to colelction.

    Args:
        shape: A `Tensor`, define the input shape
            e.g. for image input [batch_size, height, width, depth]
        name: A optional score/name for this op
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A placeholder for the input
    """
    _check_unused(unused, name)
    return _collect_named_outputs(outputs_collections, name, inputs)


def fully_connected(x, n_output, is_training, reuse, trainable=True, w_init=initz.he_normal(), b_init=0.0,
                    w_regularizer=tf.nn.l2_loss, w_normalized=False, name='fc', batch_norm=None, batch_norm_args=None, activation=None,
                    params=None, outputs_collections=None, use_bias=True):
    """Adds a fully connected layer.

        `fully_connected` creates a variable called `weights`, representing a fully
        connected weight matrix, which is multiplied by the `x` to produce a
        `Tensor` of hidden units. If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank greater than 2, then `x` is flattened
        prior to the initial matrix multiply by `weights`.

    Args:
        x: A `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
           `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 2-D `Tensor` variable representing the result of the series of operations.
        e.g: 2-D `Tensor` [batch, n_output].

    Raises:
        ValueError: if `x` has rank less than 2 or if its last dimension is not set.
    """
    if not (isinstance(n_output, six.integer_types)):
        raise ValueError('n_output should be int or long, got %s.', n_output)

    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    if len(x.get_shape()) != 2:
        x = _flatten(x)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name, reuse=reuse):
        shape = [n_input, n_output] if hasattr(w_init, '__call__') else None
        if params is not None:
            W, b = params
        else:
            W = tf.get_variable(
                name='W',
                shape=shape,
                dtype=tf.float32,
                initializer=w_init,
                regularizer=w_regularizer,
                trainable=trainable
            )
        if w_normalized:
            W = W / tf.reduce_sum(tf.square(W), 1, keep_dims=True)
        output = tf.matmul(x, W)

        if use_bias:
            if params is None:
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
    """Adds a 2D convolutional layer.

        `convolutional layer` creates a variable called `weights`, representing a conv
        weight matrix, which is multiplied by the `x` to produce a
        `Tensor` of hidden units. If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 4

    Args:
        x: A 4-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a int or tuple/list of 2 positive integers specifying the stride at which to
            compute output.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 4-D `Tensor` variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        shape = helper.filter_2d(filter_size, x.get_shape()[-1], n_output_channels) if hasattr(w_init,
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
            strides=helper.stride_2d(stride),
            padding=helper.kernel_padding(padding))

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


def dilated_conv2d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(3, 3), dilation=1, stride=1,
                   padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
                   name='dilated_conv2d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
                   outputs_collections=None):
    """Adds a 2D dilated convolutional layer

        also known as convolution with holes or atrous convolution.
        If the rate parameter is equal to one, it performs regular 2-D convolution.
        If the rate parameter
        is greater than one, it performs convolution with holes, sampling the input
        values every rate pixels in the height and width dimensions.
        `convolutional layer` creates a variable called `weights`, representing a conv
        weight matrix, which is multiplied by the `x` to produce a
        `Tensor` of hidden units. If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 4

    Args:
        x: A 4-D `Tensor` of with rank 4 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 2 positive integers specifying the spatial
        dimensions of of the filters.
        dilation:  A positive int32. The stride with which we sample input values across
            the height and width dimensions. Equivalently, the rate by which we upsample the
            filter values by inserting zeros across the height and width dimensions. In the literature,
            the same parameter is sometimes called input stride/rate or dilation.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 4-D `Tensor` variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if x has rank less than 4 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        shape = helper.filter_2d(filter_size, x.get_shape()[-1], n_output_channels) if hasattr(w_init,
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
            padding=helper.kernel_padding(padding))

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


def separable_conv2d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1), depth_multiplier=1,
                     padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
                     name='separable_conv2d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
                     outputs_collections=None):
    """Adds a 2D seperable convolutional layer.

        Performs a depthwise convolution that acts separately on channels followed by
        a pointwise convolution that mixes channels. Note that this is separability between
        dimensions [1, 2] and 3, not spatial separability between dimensions 1 and 2.
        `convolutional layer` creates two variable called `depthwise_W` and `pointwise_W`,
        `depthwise_W` is multiplied by `x` to produce depthwise conolution, which is multiplied by
        the `pointwise_W` to produce a output `Tensor`
        If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 4

    Args:
        x: A 4-D `Tensor` of with rank 4 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 2 positive integers specifying the spatial
        stride: a int or tuple/list of 2 positive integers specifying the stride at which to
            compute output.
        dimensions of of the filters.
        depth_multiplier:  A positive int32. the number of depthwise convolution output channels for
            each input channel. The total number of depthwise convolution output
            channels will be equal to `num_filters_in * depth_multiplier
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 4-D `Tensor` variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        depthwise_shape = helper.filter_2d(filter_size, x.get_shape()[-1], depth_multiplier) if hasattr(w_init,
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
            strides=helper.stride_2d(stride),
            padding=helper.kernel_padding(padding))

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


def depthwise_conv2d(x, depth_multiplier, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1),
                     padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
                     name='depthwise_conv2d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
                     outputs_collections=None):
    """Adds a 2D sdepthwise convolutional layer.

        Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter
        tensor of shape [filter_height, filter_width, in_channels, channel_multiplier] containing
        in_channels convolutional filters of depth 1, depthwise_conv2d applies a different filter
        to each input channel (expanding from 1 channel to channel_multiplier channels for each),
        then concatenates the results together. The output has in_channels * channel_multiplier channels.
        If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 4

    Args:
        x: A 4-D `Tensor` of with rank 4 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a int or tuple/list of 2 positive integers specifying the stride at which to
            compute output.
        depth_multiplier:  A positive int32. the number of depthwise convolution output channels for
            each input channel. The total number of depthwise convolution output
            channels will be equal to `num_filters_in * depth_multiplier
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The tensor variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        shape = helper.filter_2d(filter_size, x.get_shape()[-1], depth_multiplier) if hasattr(w_init,
                                                                                              '__call__') else None
        W = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.depthwise_conv2d(
            x,
            W,
            strides=helper.stride_2d(stride),
            padding=helper.kernel_padding(padding))

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


def conv3d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(3, 3, 3), stride=(1, 1, 1),
           padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
           name='conv3d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
           outputs_collections=None):
    """Adds a 3D convolutional layer.

        `convolutional layer` creates a variable called `weights`, representing a conv
        weight matrix, which is multiplied by the `x` to produce a
        `Tensor` of hidden units. If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 5

    Args:
        x: A 5-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, in_depth, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int, or  list/tuple of 3 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a int, or tuple/list of 3 positive integers specifying the stride at which to
            compute output.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 5-D `Tensor` variable representing the result of the series of operations.
        e.g.: 5-D `Tensor` [batch, new_depth, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 5 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 5, "Input Tensor shape must be 5-D"
    with tf.variable_scope(name, reuse=reuse):
        shape = helper.filter_3d(filter_size, x.get_shape(
        )[-1], n_output_channels) if hasattr(w_init, '__call__') else None
        W = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.conv3d(
            input=x,
            filter=W,
            strides=helper.stride_3d(stride),
            padding=helper.kernel_padding(padding))

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


def upsample2d(input_, output_shape, is_training, reuse, trainable=True, filter_size=(5, 5), stride=(2, 2), w_init=initz.he_normal(seed=None), b_init=0.0,
               w_regularizer=tf.nn.l2_loss, batch_norm=None, batch_norm_args=None, activation=None, name="deconv2d", use_bias=True, with_w=False, outputs_collections=None, **unused):
    """Adds a 2D upsampling or deconvolutional layer.

        his operation is sometimes called "deconvolution" after Deconvolutional Networks,
        but is actually the transpose (gradient) of conv2d rather than an actual deconvolution.
        If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 4

    Args:
        x: A 4-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        output_shape: 4D tensor, the output shape
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a int or tuple/list of 2 positive integers specifying the stride at which to
            compute output.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The tensor variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(input_)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name or 'upsample2d', reuse=reuse):
        shape = helper.filter_2d(filter_size, output_shape[-1], input_.get_shape()[-1]) if hasattr(w_init,
                                                                                                   '__call__') else None

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape, strides=helper.stride_2d(stride))
        if use_bias:
            biases = tf.get_variable(
                name='biases',
                shape=[output_shape[-1]],
                initializer=tf.constant_initializer(b_init),
                trainable=trainable
            )
            output = tf.reshape(tf.nn.bias_add(
                output, biases), output.get_shape())

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)

        if with_w:
            return _collect_named_outputs(outputs_collections, name, output), w, biases
        else:
            return _collect_named_outputs(outputs_collections, name, output)


def upsample3d(input_, output_shape, is_training, reuse, trainable=True, filter_size=(5, 5, 5), stride=(2, 2, 2), w_init=initz.he_normal(seed=None), b_init=0.0,
               w_regularizer=tf.nn.l2_loss, batch_norm=None, batch_norm_args=None, activation=None, name="deconv3d", use_bias=True, with_w=False, outputs_collections=None, **unused):
    """Adds a 3D upsampling or deconvolutional layer.

        his operation is sometimes called "deconvolution" after Deconvolutional Networks,
        but is actually the transpose (gradient) of conv2d rather than an actual deconvolution.
        If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 5

    Args:
        x: A 5-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, in_depth, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        output_shape: 5D tensor, the output shape
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 3 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a int or tuple/list of 3 positive integers specifying the stride at which to
            compute output.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The tensor variable representing the result of the series of operations.
        e.g.: 5-D `Tensor` [batch, new_depth, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(input_)
    assert len(input_shape) == 5, "Input Tensor shape must be 5-D"
    with tf.variable_scope(name or 'upsample3d', reuse=reuse):
        shape = helper.filter_3d(filter_size, output_shape[-1], input_.get_shape()[-1]) if hasattr(w_init,
                                                                                                   '__call__') else None

        # filter : [depth, height, width, output_channels, in_channels]
        w = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.conv3d_transpose(
            input_, w, output_shape=output_shape, strides=helper.stride_3d(stride))
        if use_bias:
            biases = tf.get_variable(
                name='biases',
                shape=[output_shape[-1]],
                initializer=tf.constant_initializer(b_init),
                trainable=trainable
            )
            output = tf.reshape(tf.nn.bias_add(
                output, biases), output.get_shape())

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)

        if with_w:
            return _collect_named_outputs(outputs_collections, name, output), w, biases
        else:
            return _collect_named_outputs(outputs_collections, name, output)


def conv1d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=3, stride=1,
           padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, untie_biases=False,
           name='conv1d', batch_norm=None, batch_norm_args=None, activation=None, use_bias=True,
           outputs_collections=None):
    """Adds a 1D convolutional layer.

        `convolutional layer` creates a variable called `weights`, representing a conv
        weight matrix, which is multiplied by the `x` to produce a
        `Tensor` of hidden units. If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 4

    Args:
        x: A 3-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, in_width, depth]`,
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a `int specifying the spatial
            dimensions of of the filters.
        stride: a `int` specifying the stride at which to
            compute output.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 3-D `Tensor` variable representing the result of the series of operations.
        e.g.: 3-D `Tensor` [batch, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 3, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        shape = (filter_size, x.get_shape()
                 [-1], n_output_channels) if hasattr(w_init, '__call__') else None
        W = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.conv1d(
            x,
            W,
            stride,
            padding=helper.kernel_padding(padding))

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


def max_pool_1d(x, filter_size=3, stride=2, padding='SAME', name='maxpool1d', outputs_collections=None, **unused):
    """ Max Pooling 1D.

    Args:
        x: a 3-D `Tensor` [batch_size, steps, in_channels].
        kernel_size: `int` or `list of int`. Pooling kernel size.
        strides: `int` or `list of int`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'maxpool1d'.

    Returns:
        3-D Tensor [batch, pooled steps, in_channels].
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 3, "Input Tensor shape must be 3-D"

    filter_size = helper.kernel_2d(filter_size)
    filter_size = [1, filter_size[1], 1, 1]
    stride = helper.stride_2d(stride)
    stride = [1, stride[1], 1, 1]

    with tf.name_scope(name):
        x = tf.expand_dims(x, 2)
        output = tf.nn.max_pool(
            value=x,
            ksize=filter_size,
            strides=stride,
            padding=helper.kernel_padding(padding),
        )
        output = tf.squeeze(output, [2])
        return _collect_named_outputs(outputs_collections, name, output)


def avg_pool_1d(x, filter_size=3, stride=2, padding='SAME', name='avgpool1d', outputs_collections=None, **unused):
    """ Avg Pooling 1D.

    Args:
        x: a 3-D `Tensor` [batch_size, steps, in_channels].
        kernel_size: `int` or `list of int`. Pooling kernel size.
        strides: `int` or `list of int`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'avgpool1d'.

    Returns:
        3-D Tensor [batch, pooled steps, in_channels].
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 3, "Input Tensor shape must be 3-D"

    filter_size = helper.kernel_2d(filter_size)
    filter_size = [1, filter_size[1], 1, 1]
    stride = helper.stride_2d(stride)
    stride = [1, stride[1], 1, 1]

    with tf.name_scope(name):
        x = tf.expand_dims(x, 2)
        output = tf.nn.avg_pool(
            value=x,
            ksize=filter_size,
            strides=stride,
            padding=helper.kernel_padding(padding),
        )
        output = tf.squeeze(output, [2])
        return _collect_named_outputs(outputs_collections, name, output)


def _phase_shift(input_, r):
    bsize, a, b, c = helper.get_input_shape(input_)
    X = tf.reshape(input_, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))
    X = tf.split(1, a, X)
    X = tf.concat([tf.squeeze(x) for x in X], axis=2)
    X = tf.split(1, b, X)
    X = tf.concat([tf.squeeze(x) for x in X], axis=2)
    output = tf.reshape(X, (bsize, a * r, b * r, 1))
    return output


def subpixel2d(input_, r, color=True, name=None, outputs_collections=None, **unused):
    input_shape = helper.get_input_shape(input_)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name or "subpixel"):
        if color:
            inputc = tf.split(3, 3, input_)
            output = tf.concat([_phase_shift(x, r) for x in inputc], axis=3)
        else:
            output = _phase_shift(input_, r)
    return _collect_named_outputs(outputs_collections, name, output)


def highway_conv2d(x, n_output, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1),
                   padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss,
                   name='highway_conv2d', batch_norm=None, batch_norm_args=None, activation=tf.nn.relu, use_bias=True, outputs_collections=None):
    """Adds a 2D highway convolutional layer.

        https://arxiv.org/abs/1505.00387
        If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank 4

    Args:
        x: A 4-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/ tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a int or tuple/list of 2 positive integers specifying the stride at which to
            compute output.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The `Tensor` variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.variable_scope(name, reuse=reuse):
        w_shape = helper.filter_2d(filter_size, x.get_shape(
        )[-1], n_output) if hasattr(w_init, '__call__') else None

        w_t_shape = [n_output]
        b_shape = [n_output]
        W, b = helper.weight_bias(w_shape, b_shape, w_init=w_init,
                                  b_init=b_init, w_regularizer=w_regularizer, trainable=trainable, name='main_gate/')
        W_t, b_t = helper.weight_bias(
            w_t_shape, b_shape, w_init=w_init, b_init=-3.0,
            w_regularizer=None, trainable=trainable,
            name='transform_gate/')
        output_conv2d = tf.nn.conv2d(
            input=x,
            filter=W,
            strides=helper.stride_2d(stride),
            padding=helper.kernel_padding(padding))
        H = activation(output_conv2d + b, reuse=reuse,
                       trainable=trainable, name='activation_H')
        T = tf.sigmoid(tf.multiply(output_conv2d, W_t) +
                       b_t, name='transform_gate')
        C = tf.subtract(1.0, T, name="carry_gate")
        output = tf.add(tf.multiply(H, T), tf.multiply(
            output_conv2d, C), name='output')
        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training,
                                reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


def highway_fc2d(x, n_output, is_training, reuse, trainable=True, filter_size=(3, 3),
                 w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, name='highway_fc2d', activation=None, use_bias=True, outputs_collections=None):
    """Adds a fully connected highway layer.

        https://arxiv.org/abs/1505.00387
        If a `batch_norm` is provided (such as
        `batch_norm`), it is then applied. Otherwise, if `batch_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank greater than 2, then `x` is flattened
        prior to the initial matrix multiply by `weights`.

    Args:
        x: A 2-D/4-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        b_init: An initializer for the biases. If None skip biases.
        outputs_collections: The collections to which the outputs are added.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 2-D `Tensor` variable representing the result of the series of operations.
        e.g.: 2-D `Tensor` [batch_size, n_output]

    Raises:
        ValueError: if `x` has rank less than 2 or if its last dimension is not set.
    """
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    if len(x.get_shape()) != 2:
        x = _flatten(x)

    n_input = x.get_shape().as_list()[1]
    w_shape = [n_input, n_output] if hasattr(w_init, '__call__') else None
    b_shape = [n_output]
    with tf.variable_scope(name, reuse=reuse):
        W, b = helper.weight_bias(w_shape, b_shape, w_init=w_init,
                                  b_init=b_init, w_regularizer=w_regularizer, trainable=trainable, name='main_gate')
        W_t, b_t = helper.weight_bias(
            w_shape, b_shape, w_init=w_init, b_init=b_init, w_regularizer=w_regularizer, trainable=trainable, name='transform_gate')
        H = activation(tf.matmul(x, W) + b, name='activation')
        T = tf.sigmoid(tf.matmul(x, W_t) + b_t, name='transform_gate')
        try:
            C = tf.sub(1.0, T, name="carry_gate")
        except Exception:
            C = tf.subtract(1.0, T, name="carry_gate")
        try:
            output = tf.add(tf.mul(H, T), tf.mul(x, C), name='output')
        except Exception:
            output = tf.add(tf.multiply(H, T),
                            tf.multiply(x, C), name='output')

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


def down_shifted_conv2d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(2, 3), stride=(1, 1), **kwargs):
    """ Down shifted convolution for PIXEL CNN
    """
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0],
                   [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)], [0, 0]])
    return conv2d(x, n_output_channels, is_training, reuse, trainable=trainable, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


def down_shifted_upsample2d(x, output_shape, is_training, reuse, trainable=True, filter_size=(2, 3), stride=(1, 1), **kwargs):
    """ Down shifted deconvolution for PIXEL CNN
    """
    x = upsample2d(x, output_shape, is_training, reuse, trainable=trainable,
                   filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = list(map(int, x.get_shape()))
    return x[:, :(xs[1] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[2] - int((filter_size[1] - 1) / 2)), :]


def down_right_shifted_conv2d(x, n_output_channels, is_training, reuse, filter_size=(2, 3), stride=(1, 1), **kwargs):
    """ Down right shifted convolution for PIXEL CNN
    """
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0],
                   [filter_size[1] - 1, 0], [0, 0]])
    return conv2d(x, n_output_channels, is_training, reuse, trainable=trainable, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


def down_right_upsample2d(x, output_shape, is_training, reuse, trainable=True, filter_size=(2, 2), stride=(1, 1), **kwargs):
    """ Down right shifted deconvolution for PIXEL CNN
    """
    x = upsample2d(x, output_shape, is_training, reuse, trainable=trainable,
                   filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = list(map(int, x.get_shape()))
    return x[:, :(xs[1] - filter_size[0] + 1):, :(xs[2] - filter_size[1] + 1), :]


def gated_resnet(x, is_training, reuse, a=None, h=None, activation=tf.nn.relu, conv=conv2d, init=False, counters={}, ema=None, dropout_p=0., name='gated_resnet', outputs_collections=None, **kwargs):
    """Gated Resnet block
    """
    with tf.name_scope(name):
        xs = list(map(int, x.get_shape()))
        num_filters = xs[-1]

        c1 = conv(activation(x), num_filters, is_training, reuse, **kwargs)
        if a is not None:
            c1 += conv(activation(a), num_filters, is_training,
                       reuse, filter_size=(1, 1), **kwargs)
        c1 = activation(c1)
        if dropout_p > 0:
            c1 = dropout(c1, drop_p=dropout_p)
        c2 = conv(c1, num_filters * 2, is_training,
                  reuse, init_scale=0.1, **kwargs)

        if h is not None:
            with tf.variable_scope(get_name('conditional_weights', counters)):
                hw = helper.get_var_maybe_avg('hw', ema, shape=[int_shape(h)[-1], 2 * num_filters], dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
            if init:
                hw = hw.initialized_value()
            c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])

        a, b = tf.split(3, 2, c2)
        c3 = a * tf.nn.sigmoid(b)
        output = x + c3
        return _collect_named_outputs(outputs_collections, name, output)


def max_pool(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', name='pool', outputs_collections=None, **unused):
    """
    Max pooling layer

    Args:
        x: A 4-D 'Tensor` of shape `[batch_size, height, width, channels]`
        filter_size: A int or list/tuple of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A int or list/tuple of length 2: [stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

    Raises:
        ValueError: If `input` is not 4-D array
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.nn.max_pool(
            value=x,
            ksize=helper.kernel_2d(filter_size),
            strides=helper.stride_2d(stride),
            padding=helper.kernel_padding(padding),
        )
        return _collect_named_outputs(outputs_collections, name, output)


def max_pool_3d(x, filter_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME', name='pool', outputs_collections=None, **unused):
    """
    Max pooling layer

    Args:
        x: A 5-D 'Tensor` of shape `[batch_size, depth, height, width, channels]`
        filter_size: A int or list/tuple of length 3: [kernel_depth, kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A int or list/tuple of length 3: [stride_depth, stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A `Tensor` representing the results of the pooling operation.
        e.g.: 5-D `Tensor` [batch, new_depth, new_height, new_width, channels].

    Raises:
        ValueError: If `input` is not 5-D array
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 5, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.nn.max_pool3d(
            value=x,
            ksize=helper.kernel_3d(filter_size),
            strides=helper.stride_3d(stride),
            padding=helper.kernel_padding(padding),
        )
        return _collect_named_outputs(outputs_collections, name, output)


def fractional_pool(x, pooling_ratio=[1.0, 1.44, 1.73, 1.0], pseudo_random=None, determinastic=None, overlapping=None, name='fractional_pool', seed=None,
                    seed2=None, type='avg', outputs_collections=None, **unused):
    """
    Fractional pooling layer

    Args:
        x: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
        pooling_ratio: A list of floats that has length >= 4. Pooling ratio for each
            dimension of value, currently only supports row and col dimension and should
            be >= 1.0. For example, a valid pooling ratio looks like [1.0, 1.44, 1.73, 1.0].
            The first and last elements must be 1.0 because we don't allow pooling on batch and
            channels dimensions. 1.44 and 1.73 are pooling ratio on height and width
            dimensions respectively.
        pseudo_random: An optional bool. Defaults to False. When set to True, generates
            the pooling sequence in a pseudorandom fashion, otherwise, in a random fashion.
            Check paper Benjamin Graham, Fractional Max-Pooling for difference between
            pseudorandom and random.
        overlapping: An optional bool. Defaults to False. When set to True, it means when pooling,
            the values at the boundary of adjacent pooling cells are used by both cells.
            For example: index 0 1 2 3 4
            value 20 5 16 3 7; If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used
            twice. The result would be [41/3, 26/3] for fractional avg pooling.
        deterministic: An optional bool. Defaults to False. When set to True, a fixed pooling
            region will be used when iterating over a FractionalAvgPool node in the computation
            graph. Mainly used in unit test to make FractionalAvgPool deterministic.
        seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero,
            the random number generator is seeded by the given seed. Otherwise,
            it is seeded by a random seed.
        seed2: An optional int. Defaults to 0. An second seed to avoid seed collision.
        outputs_collections: The collections to which the outputs are added.
        type: avg or max pool
        name: Optional scope/name for name_scope.

    Returns:
        A 4-D `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

    Raises:
        ValueError: If 'kernel_size' is not a 2-D list
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        if type == 'avg':
            outputs = tf.nn.fractional_avg_pool(x, pooling_ratio, pseudo_random=pseudo_random,
                                                overlapping=overlapping, deterministic=determinastic, seed=seed, seed2=seed2, name=name)
        else:
            outputs = tf.nn.fractional_max_pool(x, pooling_ratio, pseudo_random=pseudo_random,
                                                overlapping=overlapping, deterministic=determinastic, seed=seed, seed2=seed2, name=name)
        output = outputs[0]
        return _collect_named_outputs(outputs_collections, name, output)


def rms_pool_2d(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', name='pool', epsilon=0.000000000001,
                outputs_collections=None, **unused):
    """
    RMS pooling layer

    Args:
        x: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
        filter_size: A int or list/tuple of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A int or list/tuple of length 2: [stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.
        epsilon: prevents divide by zero

    Returns:
        A 4-D `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

    Raises:
        ValueError: If 'input` not 4-D array
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.nn.avg_pool(
            value=tf.square(x),
            ksize=helper.kernel_2d(filter_size),
            strides=helper.stride_2d(stride),
            padding=helper.kernel_padding(padding),
        )
        output = tf.sqrt(output + epsilon)
        return _collect_named_outputs(outputs_collections, name, output)


def rms_pool_3d(x, filter_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME', name='pool', epsilon=0.000000000001,
                outputs_collections=None, **unused):
    """
    RMS pooling layer

    Args:
        x: A 5-D `Tensor` of shape `[batch_size, depth, height, width, channels]`
        filter_size: A int or list/tuple of length 3: [kernel_depth, kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A int or list/tuple of length 3: [stride_depth, stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.
        epsilon: prevents divide by zero

    Returns:
        A 5-D `Tensor` representing the results of the pooling operation.
        e.g.: 5-D `Tensor` [batch, new_height, new_width, channels].

    Raises:
        ValueError: If 'input' is not a 5-D array
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.nn.avg_pool3d(
            value=tf.square(x),
            ksize=helper.kernel_3d(filter_size),
            strides=helper.stride_3d(stride),
            padding=helper.kernel_padding(padding),
        )
        output = tf.sqrt(output + epsilon)
        return _collect_named_outputs(outputs_collections, name, output)


def avg_pool_3d(x, filter_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME', name=None, outputs_collections=None, **unused):
    """
    Avg pooling layer

    Args:
        x: A 4-D `Tensor` of shape `[batch_size, depth, height, width, channels]`
        filter_size: A int or list/tuple of length 3: [kernel_depth, kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A int or list/tuple of length 3: [stride_depth, stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A 5-D `Tensor` representing the results of the pooling operation.
        e.g.: 5-D `Tensor` [batch, new_depth, new_height, new_width, channels].

    Raises:
        ValueError: If 'input' is not a 5-D array
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 5, "Input Tensor shape must be 5-D"
    with tf.name_scope(name or "pool"):
        output = tf.nn.avg_pool3d(
            value=x,
            ksize=helper.kernel_3d(filter_size),
            strides=helper.stride_3d(stride),
            padding=helper.kernel_padding(padding),
            name="avg_pool")
        return _collect_named_outputs(outputs_collections, name, output)


def avg_pool_2d(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', name=None, outputs_collections=None, **unused):
    """
    Avg pooling layer

    Args:
        x: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
        filter_size: A int or list/tuple of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A int or list/tuple of length 2: [stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A 4-D `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

    Raises:
        ValueError: If 'kernel_size' is not a 2-D list
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name or "pool"):
        output = tf.nn.avg_pool(
            value=x,
            ksize=helper.kernel_2d(filter_size),
            strides=helper.stride_2d(stride),
            padding=helper.kernel_padding(padding),
            name="avg_pool")
        return _collect_named_outputs(outputs_collections, name, output)


def global_avg_pool(x, name="global_avg_pool", outputs_collections=None, **unused):
    """
    Gloabl pooling layer

    Args:
        x: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A 4-D `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch, 1, 1, channels].

    Raises:
        ValueError: If 'kernel_size' is not a 2-D list
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.reduce_mean(x, [1, 2])
        return _collect_named_outputs(outputs_collections, name, output)


def global_max_pool(x, name="global_max_pool", outputs_collections=None, **unused):
    """
    Gloabl max pooling layer

    Args:
        x: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A 4-D `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch, 1, 1, channels].

    Raises:
        ValueError: If 'kernel_size' is not a 2-D list
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    with tf.name_scope(name):
        output = tf.reduce_max(x, [1, 2])
        return _collect_named_outputs(outputs_collections, name, output)


def feature_max_pool_1d(x, stride=2, name='feature_max_pool_1d', outputs_collections=None, **unused):
    """
    Feature max pooling layer

    Args:
        x: A 2-D tensor of shape `[batch_size, channels]`
        stride: A int.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A 2-D `Tensor` representing the results of the pooling operation.
        e.g.: 2-D `Tensor` [batch_size, new_channels]

    Raises:
        ValueError: If 'kernel_size' is not a 2-D list
    """
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


def feature_max_pool_2d(x, stride=2, name='feature_max_pool_2d', outputs_collections=None, **unused):
    """
    Feature max pooling layer

    Args:
        x: A 4-D tensor of shape `[batch_size, height, width, channels]`
        stride: A int.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A 4-D `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch_size, height, width, new_channels]

    Raises:
        ValueError: If 'kernel_size' is None
    """
    _check_unused(unused, name)
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) == 4, "Input Tensor shape must be 4-D"
    x = tf.reshape(
        x, (-1, input_shape[1], input_shape[2], input_shape[3] // stride, stride))
    with tf.name_scope(name):
        output = tf.reduce_max(
            input_tensor=x,
            reduction_indices=[4],
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
    """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
        "Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift", Sergey Ioffe, Christian Szegedy
        Can be used as a normalizer function for conv2d and fully_connected.
        Note: When is_training is True the moving_mean and moving_variance need to be
        updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
        they need to be added as a dependency to the `train_op`, example:
        `update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)`
        `if update_ops:`
            `updates = tf.group(*update_ops)`
        `total_loss = control_flow_ops.with_dependencies([updates], total_loss)`
        One can set updates_collections=None to force the updates in place, but that
        can have speed penalty, specially in distributed settings.
    Args:
        x: a `Tensor` with 2 or more dimensions, where the first dimension has
            `batch_size`. The normalization is over all but the last dimension if
            `data_format` is `NHWC` and the second dimension if `data_format` is
            `NCHW`.
        decay: decay for the moving average. Reasonable values for `decay` are close
            to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
            Lower `decay` value (recommend trying `decay`=0.9) if model experiences
            reasonably good training performance but poor validation and/or test
            performance. Try zero_debias_moving_mean=True for improved stability.
        center: If True, subtract `beta`. If False, `beta` is ignored.
        scale: If True, multiply by `gamma`. If False, `gamma` is
            not used. When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling can be done by the next layer.
        epsilon: small float added to variance to avoid dividing by zero.
        activation_fn: activation function, default set to None to skip it and
            maintain a linear activation.
        param_initializers: optional initializers for beta, gamma, moving mean and
            moving variance.
        updates_collections: collections to collect the update ops for computation.
            The updates_ops need to be executed with the train_op.
            If None, a control dependency would be added to make sure the updates are
            computed in place.
        is_training: whether or not the layer is in training mode. In training mode
            it would accumulate the statistics of the moments into `moving_mean` and
            `moving_variance` using an exponential moving average with the given
            `decay`. When it is not in training mode then it would use the values of
            the `moving_mean` and the `moving_variance`.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
            outputs_collections: collections to add the outputs.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        batch_weights: An optional tensor of shape `[batch_size]`,
            containing a frequency weight for each batch item. If present,
            then the batch normalization uses weighted mean and
            variance. (This can be used to correct for bias in training
            example selection.)
        fused:  Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
        name: Optional scope/name for `variable_scope`.

    Returns:
        A `Tensor` representing the output of the operation.

    Raises:
        ValueError: if `batch_weights` is not None and `fused` is True.
        ValueError: if the rank of `inputs` is undefined.
        ValueError: if rank or channels dimension of `inputs` is undefined.
    """
    return tf.contrib.layers.batch_norm(x, scope=name, scale=scale, updates_collections=updates_collections, **kwargs)


def batch_norm_lasagne(x, is_training, reuse, trainable=True, decay=0.9, epsilon=1e-4, name='bn',
                       updates_collections=tf.GraphKeys.UPDATE_OPS, outputs_collections=None):
    """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
        Instead of storing and updating moving variance, this layer store and
        update moving inverse standard deviation
        "Batch Normalization: Accelerating Deep Network Training by Reducin Internal Covariate Shift"
        Sergey Ioffe, Christian Szegedy
        Can be used as a normalizer function for conv2d and fully_connected.
        Note: When is_training is True the moving_mean and moving_variance need to be
        updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
        they need to be added as a dependency to the `train_op`, example:
        `update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)`
        `if update_ops:`
            `updates = tf.group(*update_ops)`
        `total_loss = control_flow_ops.with_dependencies([updates], total_loss)`
        One can set updates_collections=None to force the updates in place, but that
        can have speed penalty, specially in distributed settings.

    Args:
        x: a tensor with 2 or more dimensions, where the first dimension has
            `batch_size`. The normalization is over all but the last dimension if
            `data_format` is `NHWC` and the second dimension if `data_format` is
            `NCHW`.
        decay: decay for the moving average. Reasonable values for `decay` are close
            to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
            Lower `decay` value (recommend trying `decay`=0.9) if model experiences
            reasonably good training performance but poor validation and/or test
            performance. Try zero_debias_moving_mean=True for improved stability.
        epsilon: small float added to variance to avoid dividing by zero.
        updates_collections: collections to collect the update ops for computation.
            The updates_ops need to be executed with the train_op.
            If None, a control dependency would be added to make sure the updates are
            computed in place.
        is_training: whether or not the layer is in training mode. In training mode
            it would accumulate the statistics of the moments into `moving_mean` and
            `moving_variance` using an exponential moving average with the given
            `decay`. When it is not in training mode then it would use the values of
            the `moving_mean` and the `moving_variance`.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        outputs_collections: collections to add the outputs.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: Optional scope/name for `variable_scope`.

    Returns:
        A `Tensor` representing the output of the operation.

    Raises:
        ValueError: if the rank of `x` is undefined.
        ValueError: if rank or channels dimension of `inputs` is undefined.
    """
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
            initializer=tf.zeros_initializer(dtype=tf.float32),
            trainable=False)

        moving_inv_std = tf.get_variable(
            name='moving_inv_std',
            shape=[x.get_shape()[-1]],
            initializer=tf.ones_initializer(dtype=tf.float32),
            trainable=False)

        def mean_inv_std_with_update():
            mean, variance = tf.nn.moments(
                x, [0, 1, 2], shift=moving_mean, name='bn-moments')
            inv_std = math_ops.rsqrt(variance + epsilon)
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False)
            update_moving_inv_std = moving_averages.assign_moving_average(
                moving_inv_std, inv_std, decay, zero_debias=False)
            with tf.control_dependencies(
                    [update_moving_mean, update_moving_inv_std]):
                m, v = tf.identity(mean), tf.identity(inv_std)
                return m, v

        def mean_inv_std_with_pending_update():
            mean, variance = tf.nn.moments(
                x, [0, 1, 2], shift=moving_mean, name='bn-moments')
            inv_std = math_ops.rsqrt(variance + epsilon)
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False)
            update_moving_inv_std = moving_averages.assign_moving_average(
                moving_inv_std, inv_std, decay, zero_debias=False)
            tf.add_to_collection(updates_collections, update_moving_mean)
            tf.add_to_collection(updates_collections, update_moving_inv_std)
            return mean, inv_std

        mean_inv_std_with_relevant_update = mean_inv_std_with_pending_update if updates_collections is not None else mean_inv_std_with_update

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


def prelu(x, reuse, alpha_init=0.2, trainable=True, name='prelu', outputs_collections=None):
    """
    Prametric rectifier linear layer

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        alpha_init: initalization value for alpha
        trainable: a bool, training or fixed value
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the prelu activation operation.
    """
    with tf.variable_scope(name, reuse=reuse):
        alphas = tf.get_variable(
            name='alpha',
            initializer=tf.constant(alpha_init, shape=[x.get_shape()[-1]]),
            trainable=trainable
        )
        try:
            output = tf.nn.relu(x) + tf.mul(alphas, (x - tf.abs(x))) * 0.5
        except Exception:
            output = tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5
        return _collect_named_outputs(outputs_collections, name, output)


def relu(x, name='relu', outputs_collections=None, **unused):
    """
    Rectifier linear layer

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the relu activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.relu(x)
        return _collect_named_outputs(outputs_collections, name, output)


def relu6(x, name='relu6', outputs_collections=None, **unused):
    """
    Rectifier linear relu6 layer

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the relu6 activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.relu6(x)
        return _collect_named_outputs(outputs_collections, name, output)


def softplus(x, name='softplus', outputs_collections=None, **unused):
    """
    Softplus layer
    Computes softplus: log(exp(x) + 1).

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.softplus(x)
        return _collect_named_outputs(outputs_collections, name, output)


def softsign(x, name='softsign', outputs_collections=None, **unused):
    """
    Softsign layer
    Computes softsign: x / (abs(x) + 1).

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.softsign(x)
        return _collect_named_outputs(outputs_collections, name, output)


def crelu(x, name='crelu', outputs_collections=None, **unused):
    """Computes Concatenated ReLU.
        Concatenates a ReLU which selects only the positive part of the activation with
        a ReLU which selects only the negative part of the activation. Note that
        at as a result this non-linearity doubles the depth of the activations.
        Source: https://arxiv.org/abs/1603.05201

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.crelu(x)
        return _collect_named_outputs(outputs_collections, name, output)


def elu(x, name='elu', outputs_collections=None, **unused):
    """Computes exponential linear: exp(features) - 1 if < 0, features otherwise.
        See "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.elu(x)
        return _collect_named_outputs(outputs_collections, name, output)


def concat_elu(x, name='concat_elu', outputs_collections=None, **unused):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    with tf.name_scope(name):
        axis = len(x.get_shape()) - 1
        output = tf.nn.elu(tf.concat([x, -x], axis=axis))
        return _collect_named_outputs(outputs_collections, name, output)


def leaky_relu(x, alpha=0.01, name='leaky_relu', outputs_collections=None, **unused):
    """
    Computes leaky relu

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        aplha: the conatant fro scalling the activation
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        try:
            output = tf.nn.relu(x) + tf.mul(alpha, (x - tf.abs(x))) * 0.5
        except Exception:
            output = tf.nn.relu(x) + tf.multiply(alpha, (x - tf.abs(x))) * 0.5
        return _collect_named_outputs(outputs_collections, name, output)


def relu(x, name='relu', outputs_collections=None, **unused):
    """
    Computes relu

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.relu(x)
        return _collect_named_outputs(outputs_collections, name, output)


def lrelu(x, leak=0.2, name="lrelu", outputs_collections=None, **unused):
    """
    Computes reaky relu lasagne style

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        leak: the conatant fro scalling the activation
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        output = f1 * x + f2 * abs(x)
        return _collect_named_outputs(outputs_collections, name, output)


def maxout(x, k=2, name='maxout', outputs_collections=None, **unused):
    """
    Computes maxout activation

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        k: output channel splitting factor
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
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


def offset_maxout(x, k=2, name='maxout', outputs_collections=None, **unused):
    """
    Computes maxout activation

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        k: output channel splitting factor
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
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
    """
    Computes softmax activation

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        output = tf.nn.softmax(x)
        return _collect_named_outputs(outputs_collections, name, output)


def spatial_softmax(features, reuse, temperature=None, name='spatial_softmax', trainable=True, outputs_collections=None, **unused):
    """Computes the spatial softmax of a convolutional feature map.
    First computes the softmax over the spatial extent of each channel of a
    convolutional feature map. Then computes the expected 2D position of the
    points of maximal activation for each channel, resulting in a set of
    feature keypoints [x1, y1, ... xN, yN] for all N channels.
    Read more here:
    "Learning visual feature spaces for robotic manipulation with
    deep spatial autoencoders." Finn et. al, http://arxiv.org/abs/1509.06113.

    Args:
        features: A `Tensor` of size [batch_size, W, H, num_channels]; the
            convolutional feature map.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        outputs_collections: The collections to which the outputs are added.
        temperature: Softmax temperature (optional). If None, a learnable
            temperature is created.
        name: A name for this operation (optional).
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).

    Returns:
        feature_keypoints: A `Tensor` with size [batch_size, num_channels * 2];
            the expected 2D locations of each channel's feature keypoint (normalized
            to the range (-1,1)). The inner dimension is arranged as
           [x1, y1, ... xN, yN].

    Raises:
        ValueError: If unexpected data_format specified.
        ValueError: If num_channels dimension is unspecified.
    """
    shape = tf.shape(features)
    static_shape = features.shape
    height, width, num_channels = shape[1], shape[2], static_shape[3]
    if num_channels.value is None:
        raise ValueError('The num_channels dimension of the inputs to '
                         '`spatial_softmax` should be defined. Found `None`.')

    with tf.name_scope(name, 'spatial_softmax', [features]) as name:
        # Create tensors for x and y coordinate values, scaled to range [-1, 1].
        pos_x, pos_y = tf.meshgrid(tf.lin_space(-1., 1., num=height),
                                   tf.lin_space(-1., 1., num=width), indexing='ij')
        pos_x = tf.reshape(pos_x, [height * width])
        pos_y = tf.reshape(pos_y, [height * width])
        if temperature is None:
            with tf.variable_scope(name + '_temperature', reuse=reuse):
                temperature = tf.get_variable(
                    'temperature',
                    shape=(),
                    dtype=tf.float32,
                    initializer=tf.ones_initializer(),
                    trainable=trainable)
        features = tf.reshape(tf.transpose(
            features, [0, 3, 1, 2]), [-1, height * width])

        softmax_attention = tf.nn.softmax(features / temperature)
        expected_x = tf.reduce_sum(
            pos_x * softmax_attention, [1], keep_dims=True)
        expected_y = tf.reduce_sum(
            pos_y * softmax_attention, [1], keep_dims=True)
        expected_xy = tf.concat([expected_x, expected_y], 1)
        feature_keypoints = tf.reshape(
            expected_xy, [-1, num_channels.value * 2])
        feature_keypoints.set_shape([None, num_channels.value * 2])

        return _collect_named_outputs(outputs_collections, name, feature_keypoints)


def selu(x, alpha=None, scale=None, name='selu', outputs_collections=None, **unused):
    """
    Computes selu

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        alpha: float, selu parameters calculated from fixed points
        scale: float, selu parameters calculated from fixed points
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the selu activation operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        if None in (alpha, scale):
            # using parameters from 0 mean, unit variance points
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
        output = scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
        return _collect_named_outputs(outputs_collections, name, output)


def dropout_selu(x, is_training, drop_p=0.2, alpha=-1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name='dropout_selu', outputs_collections=None, **unused):
    """
    Dropout layer for self normalizing networks
    Args:
        x: a `Tensor`.
        is_training: a bool, training or validation
        drop_p: probability of droping unit
        fixedPointsMean: float, the mean used to calculate the selu parameters
        fixedPointsVar: float, the Variance used to calculate the selu parameters
        alpha: float, product of the two selu parameters
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the dropout operation.
    """
    _check_unused(unused, name)

    def dropout_selu_impl(x, drop_p, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - drop_p
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = tf.convert_to_tensor(
            keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_has_rank(0)

        alpha = tf.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        alpha.get_shape().assert_has_rank(0)

        if tf.contrib.util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else x.get_shape().as_list()
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape,
                                           seed=seed, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1 - binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob * ((1 - keep_prob) *
                                                  math_ops.pow(alpha - fixedPointMean, 2) + fixedPointVar)))

        b = fixedPointMean - a * \
            (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with tf.name_scope(name, [x]):
        if is_training:
            output = dropout_selu_impl(
                x, drop_p, alpha, noise_shape, seed, name)
        else:
            output = x
        return _collect_named_outputs(outputs_collections, name, output)


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Computes Gumbel Softmax
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    https://arxiv.org/abs/1611.01144

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(
            tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def pixel_wise_softmax(inputs):
    """Computes pixel wise softmax activation

    Args:
        x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the activation operation.
    """
    exponential_map = tf.exp(inputs)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(inputs)[3]]))
    return tf.div(exponential_map, tensor_sum_exp)


def dropout(x, is_training, drop_p=0.5, seed=None, name='dropout', outputs_collections=None, **unused):
    """
    Dropout layer
    Args:
        x: a `Tensor`.
        is_training: a bool, training or validation
        drop_p: probability of droping unit
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the dropout operation.
    """
    _check_unused(unused, name)
    with tf.name_scope(name):
        keep_p = 1. - drop_p
        if is_training:
            output = tf.nn.dropout(x, keep_p, seed=seed)
        else:
            output = x
        return _collect_named_outputs(outputs_collections, name, output)


def _flatten(x, name='flatten'):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    with tf.name_scope(name):
        dims = int(np.prod(input_shape[1:]))
        flattened = tf.reshape(x, [-1, dims])
        return flattened


def flatten(x, name='flatten'):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    with tf.name_scope(name):
        dims = int(np.prod(input_shape[1:]))
        flattened = tf.reshape(x, [-1, dims])
        return flattened


def repeat(x, repetitions, layer, num_outputs=None, name='Repeat', outputs_collections=None, *args, **kwargs):
    """
    Repeat op

    Args:
        x: a `Tensor`.
        repetitions: a int, number of times to apply the same operation
        layer: the layer function with arguments to repeat
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the repetition operation.
    """
    with tf.variable_scope(name, 'Repeat'):
        inputs = tf.convert_to_tensor(x)
        if name is None:
            if hasattr(layer, '__name__'):
                name = layer.__name__
            elif hasattr(layer, 'func') and hasattr(layer.func, '__name__'):
                # In case layer is a functools.partial.
                name = layer.func.__name__
            else:
                name = 'Repeat'
        outputs = inputs
        for i in range(repetitions):
            new_name = name + '_' + str(i + 1)
            if num_outputs is None:
                outputs = layer(outputs,
                                name=new_name, *args, **kwargs)
            else:
                outputs = layer(outputs, num_outputs,
                                name=new_name, *args, **kwargs)
            tf.add_to_collection(outputs_collections,
                                 NamedOutputs(new_name, outputs))
        return outputs


def merge(tensors_list, mode, axis=1, name='merge', outputs_collections=None, **kwargs):
    """
    Merge op

    Args:
        tensor_list: A list `Tensors` to merge
        mode: str, available modes are
            ['concat', 'elemwise_sum', 'elemwise_mul', 'sum',
                'mean', 'prod', 'max', 'min', 'and', 'or']
        name: a optional scope/name of the layer
        outputs_collections: The collections to which the outputs are added.

    Returns:
        A `Tensor` representing the results of the repetition operation.

    Raises:
        ValueError: If 'kernel_size' is not a 2-D list
    """
    assert len(tensors_list) > 1, "Merge required 2 or more tensors."

    with tf.name_scope(name):
        tensors = [l for l in tensors_list]
        if mode == 'concat':
            output = tf.concat(tensors, axis=axis)
        elif mode == 'elemwise_sum':
            output = tensors[0]
            for i in range(1, len(tensors)):
                output = tf.add(output, tensors[i])
        elif mode == 'elemwise_mul':
            output = tensors[0]
            for i in range(1, len(tensors)):
                output = tf.multiply(output, tensors[i])
        elif mode == 'sum':
            output = tf.reduce_sum(tf.concat(tensors, axis=axis), axis=axis)
        elif mode == 'mean':
            output = tf.reduce_mean(tf.concat(tensors, axis=axis), axis=axis)
        elif mode == 'prod':
            output = tf.reduce_prod(tf.concat(tensors, axis=axis), axis=axis)
        elif mode == 'max':
            output = tf.reduce_max(tf.concat(tensors, axis=axis), axis=axis)
        elif mode == 'min':
            output = tf.reduce_min(tf.concat(tensors, axis=axis), axis=axis)
        elif mode == 'and':
            output = tf.reduce_all(tf.concat(tensors, axis=axis), axis=axis)
        elif mode == 'or':
            output = tf.reduce_any(tf.concat(tensors, axis=axis), axis=axis)
        else:
            raise Exception("Unknown merge mode", str(mode))
        return _collect_named_outputs(outputs_collections, name, output)

    return output


def stack(inputs, layer, stack_args, is_training, reuse, outputs_collections=None, **kwargs):
    """Builds a stack of layers by applying layer repeatedly using stack_args.
    `stack` allows you to repeatedly apply the same operation with different
    arguments `stack_args[i]`. For each application of the layer, `stack` creates
    a new scope appended with an increasing number. For example:
    ```python
        y = stack(x, fully_connected, [32, 64, 128], scope='fc')
       # It is equivalent to:
       x = fully_connected(x, 32, scope='fc/fc_1')
       x = fully_connected(x, 64, scope='fc/fc_2')
       y = fully_connected(x, 128, scope='fc/fc_3')
    ```
    If the `scope` argument is not given in `kwargs`, it is set to
    `layer.__name__`, or `layer.func.__name__` (for `functools.partial`
    objects). If neither `__name__` nor `func.__name__` is available, the
    layers are called with `scope='stack'`.

    Args:
        inputs: A `Tensor` suitable for layer.
        layer: A layer with arguments `(inputs, *args, **kwargs)`
        stack_args: A list/tuple of parameters for each call of layer.
        outputs_collections: The collections to which the outputs are added.
        **kwargs: Extra kwargs for the layer.

    Returns:
       a `Tensor` result of applying the stacked layers.

    Raises:
        ValueError: if the op is unknown or wrong.
    """
    name = kwargs.pop('name', None)
    if not isinstance(stack_args, (list, tuple)):
        raise ValueError('stack_args need to be a list or tuple')
    with tf.variable_scope(name, 'Stack', [inputs]):
        inputs = tf.convert_to_tensor(inputs)
        if name is None:
            if hasattr(layer, '__name__'):
                name = layer.__name__
            elif hasattr(layer, 'func') and hasattr(layer.func, '__name__'):
                # In case layer is a functools.partial.
                name = layer.func.__name__
            else:
                name = 'stack'
        outputs = inputs
        for i in range(len(stack_args)):
            kwargs['name'] = name + '_' + str(i + 1)
            layer_args = stack_args[i]
            # if not isinstance(layer_args, (list, tuple)):
            #    layer_args = [layer_args]
            outputs = layer(outputs, layer_args, is_training, reuse, **kwargs)
        return _collect_named_outputs(outputs_collections, name, outputs)


def unit_norm(inputs, dim, epsilon=1e-7, scope=None):
    """Normalizes the given input across the specified dimension to unit length.
    Note that the rank of `input` must be known.

    Args:
        inputs: A `Tensor` of arbitrary size.
        dim: The dimension along which the input is normalized.
        epsilon: A small value to add to the inputs to avoid dividing by zero.
        scope: Optional scope for variable_scope.

    Returns:
        The normalized `Tensor`.

    Raises:
        ValueError: If dim is larger than the number of dimensions in 'inputs'.
    """
    with tf.variable_scope(scope, 'UnitNorm', [inputs]):
        if not inputs.get_shape():
            raise ValueError('The input rank must be known.')
        input_rank = len(inputs.get_shape().as_list())
        if dim < 0 or dim >= input_rank:
            raise ValueError(
                'dim must be positive but smaller than the input rank.')

        lengths = tf.sqrt(
            epsilon + tf.reduce_sum(tf.square(inputs), dim, True))
        multiples = []
        if dim > 0:
            multiples.append(tf.ones([dim], tf.int32))
        multiples.append(tf.strided_slice(
            tf.shape(inputs), [dim], [dim + 1], [1]))
        if dim < (input_rank - 1):
            multiples.append(tf.ones([input_rank - 1 - dim], tf.int32))
        multiples = tf.concat(multiples, 0)
        return tf.div(inputs, tf.tile(lengths, multiples))


def crop_and_concat(inputs1, inputs2, name='crop_concat'):
    """Concates two features maps
      concates different sizes feature maps cropping the larger map
      concatenation across output channels

    Args:
        inputs1: A `Tensor`
        inputs2: A `Tensor`

    Returns:
       concated output tensor
    """
    with tf.name_scope(name):
        inputs1_shape = tf.shape(inputs1)
        inputs2_shape = tf.shape(inputs2)
        # offsets for the top left corner of the crop
        offsets = [0, (inputs1_shape[1] - inputs2_shape[1]) // 2,
                   (inputs1_shape[2] - inputs2_shape[2]) // 2, 0]
        size = [-1, inputs2_shape[1], inputs2_shape[2], -1]
        inputs1_crop = tf.slice(inputs1, offsets, size)
        return tf.concat([inputs1_crop, inputs2], axis=3)


class GradientReverseLayer(object):

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, gamma=1.0):
        grad_name = "GradientReverse%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _gradients_reverse(op, grad):
            return [tf.neg(grad) * gamma]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


gradient_reverse = GradientReverseLayer()


def _collect_named_outputs(outputs_collections, name, output):
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, NamedOutputs(name, output))
    return output


def _check_unused(unused, name):
    allowed_keys = ['is_training', 'reuse',
                    'outputs_collections', 'trainable', 'padding']
    helper.veryify_args(unused, allowed_keys,
                        'Layer "%s" got unexpected argument(s):' % name)
