from __future__ import division, print_function, absolute_import

from collections import namedtuple

import numpy as np
import six
import tensorflow as tf
from tefla.core import initializers as initz
from tefla.utils import util as helper
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages

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


def fully_connected(x, n_output, is_training, reuse, trainable=True, w_init=initz.he_normal(), b_init=0.0,
                    w_regularizer=tf.nn.l2_loss, name='fc', batch_norm=None, batch_norm_args=None, activation=None,
                    outputs_collections=None, use_bias=True):
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

        filter_size: a list or tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a tuple or list of 2 positive integers specifying the stride at which to
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

        filter_size: a list or tuple of 2 positive integers specifying the spatial
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


def separable_conv2d(x, n_output_channels, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1), depth_multiplier=8,
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
        filter_size: a list or tuple of 2 positive integers specifying the spatial
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
        filter_size: a list or tuple of 2 positive integers specifying the spatial
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
        The tensor variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
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


def upsample2d(input_, output_shape, is_training, reuse, trainable=True, filter_size=(5, 5), stride=(2, 2), w_init=initz.he_normal(seed=None), b_init=0.0,
               w_regularizer=tf.nn.l2_loss, batch_norm=None, activation=None, name="deconv2d", use_bias=True, with_w=False, outputs_collections=None, **unused):
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
        filter_size: a list or tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a tuple or list of 2 positive integers specifying the stride at which to
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
        shape = [filter_size[0], filter_size[1], output_shape[-1], input_.get_shape()[-1]] if hasattr(w_init,
                                                                                                      '__call__') else None

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(
            name='W',
            shape=shape,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )

        output = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[
                                        1, stride[0], stride[1], 1])
        if use_bias:
            biases = tf.get_variable(
                name='biases',
                shape=[output_shape[-1]],
                initializer=tf.constant_initializer(b_init),
                trainable=trainable
            )
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


def highway_conv2d(x, n_output, is_training, reuse, trainable=True, filter_size=(3, 3), stride=(1, 1),
                   padding='SAME', w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss,
                   name='highway_conv2d', activation=None, use_bias=True, outputs_collections=None):
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
        filter_size: a list or tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a tuple or list of 2 positive integers specifying the stride at which to
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
        w_shape = [filter_size[0], filter_size[1], x.get_shape(
        )[-1], n_output] if hasattr(w_init, '__call__') else None

        w_t_shape = [n_output]
        b_shape = [n_output]
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
    """
    Max pooling layer

    Args:
        x: A 4-D 'Tensor` of shape `[batch_size, height, width, channels]`
        filter_size: A list of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A list of length 2: [stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.

    Returns:
        A `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

    Raises:
        ValueError: If `kernel_size' is not a 2-D list
    """
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
            output = tf.nn.fractional_avg_pool(x, pooling_ratio, pseudo_random=pseudo_random,
                                               overlapping=overlapping, deterministic=determinastic, seed=seed, seed2=seed2, name=name)
        else:
            output = tf.nn.fractional_max_pool(x, pooling_ratio, pseudo_random=pseudo_random,
                                               overlapping=overlapping, deterministic=determinastic, seed=seed, seed2=seed2, name=name)
        return _collect_named_outputs(outputs_collections, name, output)


def rms_pool_2d(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', name='pool', epsilon=0.000000000001,
                outputs_collections=None, **unused):
    """
    RMS pooling layer

    Args:
        x: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
        filter_size: A list of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A list of length 2: [stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.
        epsilon: prevents divide by zero

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
        output = tf.nn.avg_pool(
            value=tf.square(x),
            ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding=padding,
        )
        output = tf.sqrt(output + epsilon)
        return _collect_named_outputs(outputs_collections, name, output)


def avg_pool_2d(x, filter_size=(3, 3), stride=(2, 2), padding='SAME', name=None, outputs_collections=None, **unused):
    """
    Avg pooling layer

    Args:
        x: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
        filter_size: A list of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A list of length 2: [stride_height, stride_width].
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
            ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding=padding,
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


def feature_max_pool_1d(x, stride=2, name='pool', outputs_collections=None, **unused):
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
            initializer=tf.zeros_initializer,
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

        output = tf.nn.relu(x) + tf.mul(alphas, (x - tf.abs(x))) * 0.5
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
    Softpluas layer
    Computes softplus: log(exp(features) + 1).

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


def leaky_relu(x, alpha=0.01, name='leaky_relu', outputs_collections=None, **unused):
    """
    Computes reaky relu

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
        output = tf.nn.relu(x) + tf.mul(alpha, (x - tf.abs(x))) * 0.5
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


def repeat(x, repetitions, layer, name='repeat', outputs_collections=None, *args, **kwargs):
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
                name = 'repeat'
        outputs = inputs
        for i in range(repetitions):
            new_name = name + '_' + str(i + 1)
            outputs = layer(outputs, name=new_name, *args, **kwargs)
            tf.add_to_collection(outputs_collections,
                                 NamedOutputs(new_name, outputs))
        return outputs


def merge(tensors_list, mode, axis=1, name='merge', outputs_collections=None, **kwargs):
    """
    Merge op

    Args:
        tensor_list: A list `Tensors` to merge
        mode: str, available modes are
            ['concat', 'elemwise_sum', 'elemwise_mul', 'sum', 'mean', 'prod', 'max', 'min', 'and', 'or']
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
