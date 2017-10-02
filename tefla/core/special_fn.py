import random
import tensorflow as tf
from tensorflow.python.framework import function
from .layers import dilated_conv2d, layer_norm, _collect_named_outputs
from ..utils import util as helper


def fn_with_custom_grad(grad_fn, use_global_vars=False):
    """Decorator to create a subgraph with a custom gradient function.

    The subgraph created by the decorated function is NOT put in a Defun and so
    does not suffer from the limitations of the Defun (all subgraph ops on the
    same device, no summaries).

    Args:
        grad_fn: function with signature
          (inputs, variables, outputs, output_grads) -> (grad_inputs, grad_vars),
           all of which are lists of Tensors.
        use_global_vars: if True, variables will be the global variables created.
            If False, will be the trainable variables.

    Returns:
        Decorator for function such that the gradient is defined by grad_fn.
    """

    def dec(fn):

        def wrapped(*args):
            return _fn_with_custom_grad(fn, args, grad_fn, use_global_vars=use_global_vars)

        return wrapped

    return dec


def _fn_with_custom_grad(fn, inputs, grad_fn, use_global_vars=False):
    """Create a subgraph with a custom gradient.

    Args:
        fn: function that takes inputs as arguments and produces 1 or more Tensors.
        inputs: list<Tensor>, will be passed as fn(*inputs).
        grad_fn: function with signature
            (inputs, vars, outputs, output_grads) -> (grad_inputs, grad_vars),
            all of which are lists of Tensors.
        use_global_vars: if True, variables will be the global variables created.
           If False, will be the trainable variables.

    Returns:
        fn(*inputs)
    """
    with tf.variable_scope(None, default_name="fn_with_custom_grad") as vs:
        inputs = list(inputs)
        outputs = fn(*inputs)
        if use_global_vars:
            train_vars = list(vs.global_variables())
        else:
            train_vars = list(vs.trainable_variables())

    if grad_fn is None:
        return outputs
    else:
        if not (isinstance(outputs, tuple) or isinstance(outputs, list)):
            outputs = [outputs]
        outputs = list(outputs)

        in_types = [t.dtype for t in inputs]
        out_types = [t.dtype for t in outputs]
        var_types = [t.dtype for t in train_vars]

        def custom_grad_fn(op, *dys):
            """Custom grad fn applying grad_fn for identity Defun."""
            dys = list(dys)
            fn_inputs = op.inputs[:len(inputs)]
            fn_vars = op.inputs[len(inputs):len(inputs) + len(train_vars)]
            fn_outputs = op.inputs[len(inputs) + len(train_vars):]
            assert len(fn_outputs) == len(outputs)
            assert len(fn_outputs) == len(dys)

            grad_inputs, grad_vars = grad_fn(
                fn_inputs, fn_vars, fn_outputs, dys)
            grad_outputs = [None] * len(fn_outputs)
            return tuple(grad_inputs + grad_vars + grad_outputs)

        # The Defun takes as input the original inputs, the trainable variables
        # created in fn, and the outputs. In the forward it passes through the
        # outputs. In the backwards, it produces gradients for the original inputs
        # and the trainable variables.
        @function.Defun(
            *(in_types + var_types + out_types),
            func_name="identity_custom_grad%d" % random.randint(1, 10**9),
            python_grad_func=custom_grad_fn,
            shape_func=lambda _: [t.get_shape() for t in outputs])
        def identity(*args):
            outs = args[len(inputs) + len(train_vars):]
            return tuple([tf.identity(t) for t in outs])

        id_out = identity(*(inputs + train_vars + outputs))
        return id_out


def format_input_left_padding(inputs, **kwargs):
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
        raise ValueError(
            "Inputs to conv must have statically known rank 4. Shape: " + str(static_shape))
    dilation = (1, 1)
    assert kwargs['filter_size'] is not None
    filter_size = kwargs['filter_size']
    if isinstance(filter_size, int):
        filter_size = [filter_size, filter_size]
    if "dilation" in kwargs:
        dilation_rate = kwargs["dilation"]
    assert filter_size[0] % 2 == 1 and filter_size[1] % 2 == 1
    height_padding = 2 * (filter_size[0] // 2) * dilation[0]
    cond_padding = tf.cond(
        tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (filter_size[1] // 2) * dilation[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    # Set middle two dimensions to None to prevent convolution from complaining
    inputs.set_shape([static_shape[0], None, None, static_shape[3]])
    kwargs["padding"] = "VALID"
    return inputs, kwargs


def saturating_sigmoid(x):
    """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
    with tf.name_scope("saturating_sigmoid", [x]):
        y = tf.sigmoid(x)
        return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1))


def hard_sigmoid(x, saturation_limit=0.9):
    saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
    x_shifted = 0.5 * x + 0.5
    return tf.minimum(1.0, tf.nn.relu(x_shifted)), saturation_cost


def hard_tanh(x, saturation_limit=0.9):
    saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
    return tf.minimum(1.0, tf.maximum(x, -1.0)), saturation_cost


def conv2d_v2(inputs, n_output_channels, is_training, reuse, **kwargs):
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
        padding: one of `"VALID"` or `"SAME"`. IF padding is LEFT, it preprocess the input to use Valid padding
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
    if 'padding' in kwargs and kwargs['padding'] == 'LEFT':
        inputs, kwargs = format_input_left_padding(inputs, **kwargs)
    return dilated_conv2d(inputs, n_output_channels, is_training, reuse, **kwargs)


def conv2d_gru(inputs, n_output_channels, is_training, reuse, filter_size=3, padding="SAME", dilation=1, name='conv2d_gru', outputs_collections=None, **kwargs):
    """Adds a convolutional GRU layer in 1 dimension

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
        padding: one of `"VALID"` or `"SAME"`. IF padding is LEFT, it preprocess the input to use Valid padding
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
    def conv2d_fn(x, name, bias_start, padding):
        return conv2d_v2(x, n_output_channels, is_training, reuse, filter_size=filter_size, padding=padding, b_init=bias_start, dilation=dilation, name=name, **kwargs)

    with tf.variable_scope(name, reuse=reuse):
        reset = saturating_sigmoid(conv2d_fn(inputs, "reset", 1.0, padding))
        gate = saturating_sigmoid(conv2d_fn(inputs, "gate", 1.0, padding))
        candidate = tf.tanh(
            conv2d_fn(reset * inputs, "candidate", 0.0, padding))
        outputs = gate * inputs + (1 - gate) * candidate
        return _collect_named_outputs(outputs_collections, name, outputs)


def conv2d_lstm(inputs, n_output_channels, is_training, reuse, filter_size=3, padding="SAME", dilation=1, name='conv2d_gru', outputs_collections=None, **kwargs):
    """Adds a convolutional LSTM layer in 1 dimension

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
        padding: one of `"VALID"` or `"SAME"`. IF padding is LEFT, it preprocess the input to use Valid padding
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
    with tf.variable_scope(name, reuse=reuse):
        gates = conv2d_v2(inputs, 4 * n_output_channels, is_training, reuse,
                          filter_size=filter_size, padding=padding, dilation=dilation, name=name, **kwargs)
        g = tf.split(layer_norm(gates, 4 * n_ouput_channels), 4, axis=3)
        new_cell = tf.sigmoid(g[0]) * x + tf.sigmoid(g[1]) * tf.tanh(g[3])
        outputs = tf.sigmoid(g[2]) * tf.tanh(new_cell)
        return _collect_named_outputs(outputs_collections, name, outputs)


def conv2d_diagonal_gru(inputs, n_output_channels, is_training, reuse, filter_size=3, padding="SAME", dilation=1, dropout=0.0, name='conv2d_gru', outputs_collections=None, **kwargs):
    """Adds a convolutional diagonal GRU layer in 1 dimension

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
        padding: one of `"VALID"` or `"SAME"`. IF padding is LEFT, it preprocess the input to use Valid padding
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
    def conv2d_fn(x, name, bias_start):
        return conv2d_v2(x, n_output_channels, is_training, reuse, filter_size=filter_size, padding=padding, b_init=bias_start, dilation=dilation, name=name, **kwargs)

    with tf.variable_scope(name, reuse=reuse):
        reset, reset_cost = hard_sigmoid(conv2d_fn(x, "reset", 0.5))
        gate, gate_cost = hard_sigmoid(conv2d_fn(x, "gate", 0.7))
        candidate = tf.tanh(conv2d_fn(reset * x, "candidate", 0.0))

        if dropout > 0.0:
            candidate = tf.layers.dropout(
                candidate, dropout, training=is_training)

        # Diagonal shift.
        shift_filters = n_output_channels // 3
        base_filter = ([[0, 1, 0]] * (n_output_channels - 2 * shift_filters) +
                       [[1, 0, 0]] * shift_filters + [[0, 0, 1]] * shift_filters)
        shift_filter = tf.constant(np.transpose(base_filter), dtype=tf.float32)
        shift_filter = tf.expand_dims(tf.expand_dims(shift_filter, 0), 3)
        x_shifted = tf.nn.depthwise_conv2d(
            x, shift_filter, [1, 1, 1, 1], padding="SAME")

        # Return the gated result and cost.
        total_cost_avg = 0.5 * (reset_cost + gate_cost)
        outputs = gate * x_shifted + (1 - gate) * candidate, total_cost_avg
        return _collect_named_outputs(outputs_collections, name, outputs)


def multiscale_conv2d_sum(inputs, n_output_channels, is_training, reuse, dilation_rates_and_filter_sizes,
                          pooling_type, name='multiscale_conv2d_sum', outputs_collections=None, **kwargs):
    """Sum of several dilated convolutions.

    For all convolutions with dilation_rate > 1, we first pool the input with
    width dilation_rate.

    Args:
        x: A 4-D `Tensor` of with rank 4 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 2 positive integers specifying the spatial
        dimensions of of the filters.
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
        dilation_rates_and_kernel_sizes: a list of pairs (dilation, kernel_size)
        pooling_type: "AVG" or "MAX"
        **kwargs: additional

    Returns:
        The 4-D `Tensor` variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if x has rank less than 4 or if its last dimension is not set.
    """
    with tf.variable_scope(name, reuse=reuse):
        padding = kwargs["padding"]
        results, counter = [], -1
        for dilation_rate, filter_size in dilation_rates_and_filter_sizes:
            counter += 1
            if dilation_rate[0] > 1:
                pooled = pool2d(inputs, filter_size, pooling_type, padding)
            else:
                pooled = inputs
            results.append(
                conv2d_v2(pooled, n_output_channels, is_training, reuse, filter_size=filter_size,
                          dilation=dilation_rate, name="conv_layer%d" % counter, **kwargs))
        outputs = tf.add_n(results) * (len(results)**-0.5)
        return _collect_named_outputs(outputs_collections, name, outputs)


def pool2d(inputs, filter_size=(3, 3), pooling_type='AVG', padding='SAME', strides=(1, 1), outputs_collections=None, name='general_pool', **kwargs):
    """
    General pooling layer; Supports LEFT padding

    Args:
        x: A 4-D 'Tensor` of shape `[batch_size, height, width, channels]`
        filter_size: A int or list/tuple of length 2: [kernel_height, kernel_width] of the
            pooling kernel over which the op is computed. Can be an int if both
            values are the same.
        stride: A int or list/tuple of length 2: [stride_height, stride_width].
        padding: The padding method, either 'VALID' or 'SAME'.
        outputs_collections: The collections to which the outputs are added.
        name: Optional scope/name for name_scope.
        pooling_type: "AVG" or "MAX"
        **kwargs: additional

    Returns:
        A `Tensor` representing the results of the pooling operation.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

    Raises:
        ValueError: If `input` is not 4-D array
    """
    with tf.name_scope("pool", [inputs]):
        static_shape = inputs.get_shape()
        if not static_shape or len(static_shape) != 4:
            raise ValueError(
                "Inputs to conv must have statically known rank 4.")
        # Add support for left padding.
        if padding == "LEFT":
            assert filter_size[0] % 2 == 1 and filter_size[1] % 2 == 1
            if len(static_shape) == 3:
                width_padding = 2 * (filter_size[1] // 2)
                padding_ = [[0, 0], [width_padding, 0], [0, 0]]
            else:
                height_padding = 2 * (filter_size[0] // 2)
                cond_padding = tf.cond(
                    tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
                    lambda: tf.constant(2 * (filter_size[1] // 2)))
                width_padding = 0 if static_shape[2] == 1 else cond_padding
                padding_ = [[0, 0], [height_padding, 0],
                            [width_padding, 0], [0, 0]]
            inputs = tf.pad(inputs, padding_)
            inputs.set_shape([static_shape[0], None, None, static_shape[3]])
            padding = "VALID"

        outputs = tf.nn.pool(inputs, filter_size, pooling_type,
                             padding, strides=strides)
        return _collect_named_outputs(outputs_collections, name, outputs)
