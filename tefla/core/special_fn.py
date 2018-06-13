import re
import functools
from collections import defaultdict
import contextlib
import random
import tensorflow as tf
from tensorflow.python.framework import function
from .layers import dilated_conv2d, conv1d, layer_norm, _layer_norm_compute_python, \
    _collect_named_outputs
from .optimizer import VariableClippingOptimizer
from ..utils import util as helper
from . import initializers as initz

_function_cache = {}


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

      grad_inputs, grad_vars = grad_fn(fn_inputs, fn_vars, fn_outputs, dys)
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
    raise ValueError("Inputs to conv must have statically known rank 4. Shape: " + str(static_shape))
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


def shift_right(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
  return shifted_targets


def shift_right_3d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
  return shifted_targets


def shift_right_2d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0]])[:, :-1]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1]
  return shifted_targets


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


def top_k_gpu(x, k):
  """GPU-compatible version of top-k that works for very small constant k.

  Calls argmax repeatedly.

  tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
  seems not to be, so if we use tf.nn.top_k, then both the top_k and its
  gradient go on cpu.  Once this is not an issue, this function becomes
  obselete and should be replaced by tf.nn.top_k.

  Args:
    x: a 2d Tensor.
    k: a small integer.

  Returns:
    values: a Tensor of shape [batch_size, k]
    indices: a int32 Tensor of shape [batch_size, k]
  """
  if k > 10:
    return tf.nn.top_k(x, k)
  values = []
  indices = []
  depth = tf.shape(x)[1]
  for i in range(k):
    values.append(tf.reduce_max(x, 1))
    argmax = tf.argmax(x, 1)
    indices.append(argmax)
    if i + 1 < k:
      x += tf.one_hot(argmax, depth, -1e9)
  return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))


def conv2d_v2(inputs, n_output_channels, is_training, reuse, **kwargs):
  """Adds a 2D dilated convolutional layer.

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
      padding: one of `"VALID"` or `"SAME"`. IF padding is LEFT, it preprocess the input to use
          Valid padding
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


def conv2d_gru(inputs,
               n_output_channels,
               is_training,
               reuse,
               filter_size=3,
               padding="SAME",
               dilation=1,
               name='conv2d_gru',
               outputs_collections=None,
               **kwargs):
  """Adds a convolutional GRU layer in 1 dimension.

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
      padding: one of `"VALID"` or `"SAME"`. IF padding is LEFT, it preprocess the
          input to use Valid padding
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
    return conv2d_v2(
        x,
        n_output_channels,
        is_training,
        reuse,
        filter_size=filter_size,
        padding=padding,
        b_init=bias_start,
        dilation=dilation,
        name=name,
        **kwargs)

  with tf.variable_scope(name, reuse=reuse):
    reset = saturating_sigmoid(conv2d_fn(inputs, "reset", 1.0, padding))
    gate = saturating_sigmoid(conv2d_fn(inputs, "gate", 1.0, padding))
    candidate = tf.tanh(conv2d_fn(reset * inputs, "candidate", 0.0, padding))
    outputs = gate * inputs + (1 - gate) * candidate
    return _collect_named_outputs(outputs_collections, name, outputs)


def conv2d_lstm(inputs,
                n_output_channels,
                is_training,
                reuse,
                filter_size=3,
                padding="SAME",
                dilation=1,
                name='conv2d_gru',
                outputs_collections=None,
                **kwargs):
  """Adds a convolutional LSTM layer in 1 dimension.

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
      padding: one of `"VALID"` or `"SAME"`. IF padding is LEFT, it preprocess the
          input to use Valid padding
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
    gates = conv2d_v2(
        inputs,
        4 * n_output_channels,
        is_training,
        reuse,
        filter_size=filter_size,
        padding=padding,
        dilation=dilation,
        name=name,
        **kwargs)
    g = tf.split(layer_norm(gates, 4 * n_ouput_channels), 4, axis=3)
    new_cell = tf.sigmoid(g[0]) * x + tf.sigmoid(g[1]) * tf.tanh(g[3])
    outputs = tf.sigmoid(g[2]) * tf.tanh(new_cell)
    return _collect_named_outputs(outputs_collections, name, outputs)


def conv2d_diagonal_gru(inputs,
                        n_output_channels,
                        is_training,
                        reuse,
                        filter_size=3,
                        padding="SAME",
                        dilation=1,
                        dropout=0.0,
                        name='conv2d_gru',
                        outputs_collections=None,
                        **kwargs):
  """Adds a convolutional diagonal GRU layer in 1 dimension.

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
      padding: one of `"VALID"` or `"SAME"`. IF padding is LEFT, it preprocess the
          input to use Valid padding
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
    return conv2d_v2(
        x,
        n_output_channels,
        is_training,
        reuse,
        filter_size=filter_size,
        padding=padding,
        b_init=bias_start,
        dilation=dilation,
        name=name,
        **kwargs)

  with tf.variable_scope(name, reuse=reuse):
    reset, reset_cost = hard_sigmoid(conv2d_fn(x, "reset", 0.5))
    gate, gate_cost = hard_sigmoid(conv2d_fn(x, "gate", 0.7))
    candidate = tf.tanh(conv2d_fn(reset * x, "candidate", 0.0))

    if dropout > 0.0:
      candidate = tf.layers.dropout(candidate, dropout, training=is_training)

    # Diagonal shift.
    shift_filters = n_output_channels // 3
    base_filter = ([[0, 1, 0]] * (n_output_channels - 2 * shift_filters) +
                   [[1, 0, 0]] * shift_filters + [[0, 0, 1]] * shift_filters)
    shift_filter = tf.constant(np.transpose(base_filter), dtype=tf.float32)
    shift_filter = tf.expand_dims(tf.expand_dims(shift_filter, 0), 3)
    x_shifted = tf.nn.depthwise_conv2d(x, shift_filter, [1, 1, 1, 1], padding="SAME")

    # Return the gated result and cost.
    total_cost_avg = 0.5 * (reset_cost + gate_cost)
    outputs = gate * x_shifted + (1 - gate) * candidate, total_cost_avg
    return _collect_named_outputs(outputs_collections, name, outputs)


def multiscale_conv2d_sum(inputs,
                          n_output_channels,
                          is_training,
                          reuse,
                          dilation_rates_and_filter_sizes,
                          pooling_type,
                          name='multiscale_conv2d_sum',
                          outputs_collections=None,
                          **kwargs):
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
          conv2d_v2(
              pooled,
              n_output_channels,
              is_training,
              reuse,
              filter_size=filter_size,
              dilation=dilation_rate,
              name="conv_layer%d" % counter,
              **kwargs))
    outputs = tf.add_n(results) * (len(results)**-0.5)
    return _collect_named_outputs(outputs_collections, name, outputs)


def conv1d_memory_efficient(x,
                            n_output,
                            is_training,
                            reuse,
                            trainable=True,
                            w_init=initz.he_normal(),
                            w_regularizer=tf.nn.l2_loss,
                            epsilon=1e-6,
                            forget=True,
                            test_vars=None,
                            name='conv1d_memory_efficient'):
  """LayerNorm, Conv, ReLU, Conv.

  All convolutions have kernel size 1.

  returns conv(relu(conv(layer_norm(x))))

  Args:
    x: input Tensor with shape [batch, length, io_size]
    n_output: an integer - size of the hidden layer.
    is_training: Bool, training or testing
    reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
    epsilon: a float (for layer norm)
    forget: a boolean - forget forwards activations and recompute on backprop
    test_vars: optional tuple of variables for testing purposes
    name: an optional string

  Returns:
    a Tensor with shape [batch, length, io_size]
  """
  io_size = x.get_shape().as_list()[-1]

  def forward_internal(x, f1, f2, scale, bias):
    """Forward function."""
    num_splits = 4
    x_flat = tf.reshape(x, [-1, 1, tf.shape(x)[2]])
    xs = approximate_split(x_flat, num_splits)
    ys = []
    for i in range(num_splits):
      with tf.control_dependencies(ys[-1:]):
        n = _layer_norm_compute_python(xs[i], epsilon, scale, bias)
        y = tf.nn.conv1d(n, f1, 1, "SAME")
        y = tf.nn.relu(y)
        y = tf.nn.conv1d(y, f2, 1, "SAME")
        ys.append(y)
    y = tf.concat(ys, 0)
    y = tf.reshape(y, tf.shape(x))
    return y

  key = ("conv1d_memory_efficient %s" % epsilon)
  if not forget:
    forward_fn = forward_internal
  elif key in _function_cache:
    forward_fn = _function_cache[key]
  else:

    @function.Defun(compiled=True)
    def grad_fn(x, f1, f2, scale, bias, dy):
      with tf.control_dependencies([dy]):
        num_splits = 4
        x_shape = tf.shape(x)
        flat_shape = [-1, 1, x_shape[2]]
        x = tf.reshape(x, flat_shape)
        dy = tf.reshape(dy, flat_shape)
        xs = approximate_split(x, num_splits)
        dys = approximate_split(dy, num_splits)
        dxs = []
        df1 = 0
        df2 = 0
        dscale = 0
        dbias = 0
        deps = []
        for i in range(num_splits):
          with tf.control_dependencies(deps):
            n = _layer_norm_compute_python(xs[i], epsilon, scale, bias)
            y = tf.nn.conv1d(n, f1, 1, "SAME")
            y = tf.nn.relu(y)
            y = tf.nn.conv1d(y, f2, 1, "SAME")
            dxi, pdf1, pdf2, pdscale, pdbias = tf.gradients(
                ys=[y], xs=[xs[i], f1, f2, scale, bias], grad_ys=[dys[i]])
            df1 += pdf1
            df2 += pdf2
            dscale += pdscale
            dbias += pdbias
            dxs.append(dxi)
            deps = [dxi, df1, df2, dscale, dbias]
        with tf.control_dependencies(deps):
          dx = tf.concat(dxs, 0)
          dx = tf.reshape(dx, x_shape)
          return dx, df1, df2, dscale, dbias

    @function.Defun(grad_func=grad_fn, compiled=True, separate_compiled_gradients=True)
    def forward_fn(x, f1, f2, scale, bias):
      return forward_internal(x, f1, f2, scale, bias)

  with tf.variable_scope(name, reuse=reuse, default_name="ffn2", values=[x]):
    if test_vars is not None:
      f1, f2, scale, bias = list(test_vars)
    else:
      f1 = tf.get_variable(
          "f1", [1, io_size, n_output],
          trainable=trainable,
          initializer=w_init,
          regularizer=w_regularizer)
      f2 = tf.get_variable(
          "f2", [1, n_output, io_size],
          trainable=trainable,
          initializer=w_init,
          regularizer=w_regularizer)
      scale = tf.get_variable(
          "layer_norm_scale", [io_size], initializer=tf.ones_initializer(), trainable=trainable)
      bias = tf.get_variable(
          "layer_norm_bias", [io_size], initializer=tf.zeros_initializer(), trainable=trainable)
    if forget:
      y = forward_fn(x, f1, f2, scale, bias)
    else:
      y = forward_internal(x, f1, f2, scale, bias)
    y.set_shape(x.get_shape())
    return y


def approximate_split(x, num_splits, axis=0):
  """Split approximately equally into num_splits parts.

  Args:
    x: a Tensor
    num_splits: an integer
    axis: an integer.

  Returns:
    a list of num_splits Tensors.
  """
  size = tf.shape(x)[axis]
  size_splits = [tf.div(size + i, num_splits) for i in range(num_splits)]
  return tf.split(x, size_splits, axis=axis)


def pool2d(inputs,
           filter_size=(3, 3),
           pooling_type='AVG',
           padding='SAME',
           strides=(1, 1),
           outputs_collections=None,
           name='general_pool',
           **kwargs):
  """General pooling layer; Supports LEFT padding.

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
      raise ValueError("Inputs to conv must have statically known rank 4.")
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
        padding_ = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
      inputs = tf.pad(inputs, padding_)
      inputs.set_shape([static_shape[0], None, None, static_shape[3]])
      padding = "VALID"

    outputs = tf.nn.pool(inputs, filter_size, pooling_type, padding, strides=strides)
    return _collect_named_outputs(outputs_collections, name, outputs)


def variable_ref(t):
  """Find the variable ref, ignoring Identity ops.

  Args:
    t: a Tensor

  Returns:
    a Tensor that is a variable ref, or None on error.
  """
  while t.op.type == "Identity":
    t = t.op.inputs[0]
  if "Variable" in t.op.type:
    return t
  else:
    return None


def _acc_grads(*lists_of_grads):
  """Accumulates lists of gradients."""
  acc_grads = []
  for grads in zip(*lists_of_grads):
    grads = [g for g in grads if g is not None]
    if grads:
      acc_grads.append(tf.add_n(grads))
    else:
      acc_grads.append(None)
  return acc_grads


def _rev_layer_forward(xs, f, g, f_side_input, g_side_input, gate_outputs=False):
  """Forward for 1 reversible layer."""
  x1, x2 = xs
  with tf.variable_scope("f"):
    y1 = x1 + (f(x2, f_side_input) if f_side_input else f(x2))
  with tf.variable_scope("g"):
    y2 = x2 + (g(y1, g_side_input) if g_side_input else g(y1))
  if gate_outputs:
    return tf.tuple([y1, y2])
  else:
    return (y1, y2)


def _rev_layer_backward(ys, grad_ys, f, g, f_vars, f_side_input, g_vars, g_side_input):
  """Backprop for 1 layer."""
  y1, y2 = ys
  grad_y1, grad_y2 = grad_ys

  # Reconstruct intermediates and inputs (x1, x2)
  # stop_gradients required on fn inputs to prevent infinite recursion into this
  # grad function on the calls to tf.gradients.
  y1_stop = tf.stop_gradient(y1)
  g_side_input = [tf.stop_gradient(t) for t in g_side_input]
  with tf.variable_scope("g"):
    gy1 = g(y1_stop, g_side_input) if g_side_input else g(y1_stop)

  x2 = y2 - gy1
  x2_stop = tf.stop_gradient(x2)
  f_side_input = [tf.stop_gradient(t) for t in f_side_input]
  with tf.variable_scope("f"):
    fx2 = f(x2_stop, f_side_input) if f_side_input else f(x2_stop)

  x1 = y1 - fx2

  # Compute gradients wrt to inputs
  # dL/dy2 * dG(y1)/y1
  grad_gy1_y2 = tf.gradients(gy1, y1_stop, grad_y2)[0]
  grad_x1 = grad_y1 + grad_gy1_y2
  grad_x2 = (
      tf.gradients(fx2, x2_stop, grad_y1)[0] + grad_y2 + tf.gradients(fx2, x2_stop, grad_gy1_y2)[0])

  # Compute gradients wrt to vars and side inputs in f and g
  grads1 = tf.gradients(gy1, g_vars + g_side_input, grad_y2)
  grad_g_vars, grad_g_side = grads1[:len(g_vars)], grads1[len(g_vars):]
  grads2 = tf.gradients(fx2, f_vars + f_side_input, grad_y1)
  grad_f_y1, grad_f_side1 = grads2[:len(f_vars)], grads2[len(f_vars):]
  grads3 = tf.gradients(fx2, f_vars + f_side_input, grad_gy1_y2)
  grad_f_y2, grad_f_side2 = grads3[:len(f_vars)], grads3[len(f_vars):]
  grad_f_vars = _acc_grads(grad_f_y1, grad_f_y2)

  grad_f_side = _acc_grads(grad_f_side1, grad_f_side2)

  # Put returns in a tuple to ensure a constant memory budget (i.e. don't want
  # the subsequent layer to start computing and consuming memory based on a
  # subset of these values).
  outs = tf.tuple([x1, x2, grad_x1, grad_x2] + grad_f_vars + grad_g_vars + grad_f_side + grad_g_side)
  x1, x2, grad_x1, grad_x2 = outs[:4]
  grad_f_vars_end = 4 + len(grad_f_vars)
  grad_g_vars_end = grad_f_vars_end + len(grad_g_vars)
  grad_f_side_end = grad_g_vars_end + len(grad_f_side)

  grad_f_vars = outs[4:grad_f_vars_end]
  grad_g_vars = outs[grad_f_vars_end:grad_g_vars_end]
  grad_f_side = outs[grad_g_vars_end:grad_f_side_end]
  grad_g_side = outs[grad_f_side_end:]

  return ((x1, x2), (grad_x1, grad_x2), (grad_f_vars, grad_f_side), (grad_g_vars, grad_g_side))


def _rev_block_forward(x1,
                       x2,
                       f,
                       g,
                       num_layers=1,
                       f_side_input=None,
                       g_side_input=None,
                       layer_scopes=None,
                       gate_outputs=False,
                       name=None):
  """Forward for a series of reversible layers."""
  out = (x1, x2)
  with tf.variable_scope(name, default_name="revblock"):
    for i in range(num_layers):
      with tf.variable_scope("revlayer_%d" % i) as layer_vs:
        if layer_scopes is not None:
          layer_scopes.append(layer_vs)
        out = _rev_layer_forward(
            out, f[i], g[i], f_side_input, g_side_input, gate_outputs=gate_outputs)

  y1, y2 = out
  return y1, y2


LAYER_RE = re.compile(".*revlayer_([0-9]*)/([fg])/.*")


def rev_block(x1, x2, f, g, num_layers=1, f_side_input=None, g_side_input=None, is_training=True):
  """A block of reversible residual layers.

  A reversible residual layer is defined as:

  ```
  y1 = x1 + f(x2, f_side_input)
  y2 = x2 + g(y1, g_side_input)
  ```

  A reversible residual block, defined here, is a series of reversible residual
  layers.

  Limitations:
  * f and g must not close over any Tensors; all side inputs to f and g should
    be passed in with f_side_input and g_side_input which will be forwarded to
    f and g.
  * f and g must not change the dimensionality of their inputs in order for the
    addition in the equations above to work.

  Args:
    x1: a float Tensor.
    x2: a float Tensor.
    f: a function, (Tensor) -> (Tensor) (or list of such of length num_layers).
      Should not change the shape of the Tensor. Expected to create variables.
      See f_side_input if there are side inputs.
    g: a function, (Tensor) -> (Tensor) (or list of such of length num_layers).
      Should not change the shape of the Tensor. Expected to create variables.
      See g_side_input if there are side inputs.
    num_layers: int, number of reversible residual layers. Each layer will
      apply f and g according to the equations above, with new variables in each
      layer.
    f_side_input: list of Tensors, side input to f. If not None, signature of f
      should be (Tensor, list<Tensor>) -> (Tensor).
    g_side_input: list of Tensors, side input to g. If not None, signature of g
      should be (Tensor, list<Tensor>) -> (Tensor).
    is_training: bool, whether to actually use the efficient backprop codepath.

  Returns:
    y1, y2: tuple of float Tensors.
  """
  if f_side_input is None:
    f_side_input = []
  if g_side_input is None:
    g_side_input = []
  if isinstance(f, list):
    assert len(f) == num_layers
  else:
    f = [f] * num_layers
  if isinstance(g, list):
    assert len(g) == num_layers
  else:
    g = [g] * num_layers

  # Filled by the forward function below
  layer_scopes = []

  def custom_grad_fn(inputs, variables, ys, grad_ys):
    """Custom gradient fn for a block of reversible residual layers."""
    side_inputs = inputs[2:]
    f_side_idxs = [None] * len(f_side_input)
    g_side_idxs = [None] * len(g_side_input)
    assert len(side_inputs) == len(f_side_input) + len(g_side_input)

    for i, t in enumerate(side_inputs):
      if t in f_side_input:
        f_side_idxs[f_side_input.index(t)] = i
      elif t in g_side_input:
        g_side_idxs[g_side_input.index(t)] = i
      else:
        assert False

    f_vars = [[] for _ in range(num_layers)]
    g_vars = [[] for _ in range(num_layers)]
    f_vars_idxs = [[] for _ in range(num_layers)]
    g_vars_idxs = [[] for _ in range(num_layers)]

    for i, t in enumerate(variables):
      ref = variable_ref(t)

      # Use the name to identify the layer number and function (f or g)
      regex = LAYER_RE.match(ref.name)
      layer_no = int(regex.group(1))
      fn_name = regex.group(2)
      if fn_name == "f":
        f_vars[layer_no].append(ref)
        f_vars_idxs[layer_no].append(i)
      else:
        assert fn_name == "g"
        g_vars[layer_no].append(ref)
        g_vars_idxs[layer_no].append(i)

    f_var_grads = []
    g_var_grads = []
    f_side_grads = []
    g_side_grads = []

    # Reverse variable containers to go backward
    layer_scopes.reverse()
    f_vars.reverse()
    g_vars.reverse()
    f.reverse()
    g.reverse()

    for i in range(num_layers):
      with tf.variable_scope(layer_scopes[i], reuse=True):

        ys, grad_ys, f_ret, g_ret = _rev_layer_backward(ys, grad_ys, f[i], g[i], f_vars[i],
                                                        f_side_input, g_vars[i], g_side_input)

        grad_f_vars, grad_f_side = f_ret
        grad_g_vars, grad_g_side = g_ret
        f_var_grads.append(grad_f_vars)
        g_var_grads.append(grad_g_vars)
        f_side_grads.append(grad_f_side)
        g_side_grads.append(grad_g_side)

    # Accumulate layer gradients for f_side_input and g_side_input
    acc_f_side_grads = _acc_grads(*f_side_grads)
    acc_g_side_grads = _acc_grads(*g_side_grads)

    # Use the stored idxs to put gradients in the passed-in order.
    side_input_grads = [None] * len(side_inputs)
    variable_grads = [None] * len(variables)

    # Variable gradients were collected in reverse layer order. Reverse to match
    # idxs.
    f_var_grads.reverse()
    g_var_grads.reverse()
    for idxs, grads in list(zip(f_vars_idxs, f_var_grads)) + list(zip(g_vars_idxs, g_var_grads)):
      for i, grad in zip(idxs, grads):
        variable_grads[i] = grad

    for i, grad in zip(f_side_idxs, acc_f_side_grads):
      side_input_grads[i] = grad
    for i, grad in zip(g_side_idxs, acc_g_side_grads):
      side_input_grads[i] = grad

    grad_x1, grad_x2 = grad_ys
    return [grad_x1, grad_x2] + side_input_grads, variable_grads

  # Need a forward function with positional arguments
  @fn_with_custom_grad(custom_grad_fn if is_training else None)
  def forward(x1, x2, *side_inputs):
    f_side = side_inputs[:len(f_side_input)]
    g_side = side_inputs[len(f_side_input):]
    return _rev_block_forward(
        x1,
        x2,
        f,
        g,
        num_layers=num_layers,
        f_side_input=f_side,
        g_side_input=g_side,
        layer_scopes=layer_scopes,
        gate_outputs=is_training)

  return forward(x1, x2, *(f_side_input + g_side_input))


def recompute_grad(fn):
  """Decorator that recomputes the function on the backwards pass.

  Args:
    fn: a function that takes Tensors (all as positional arguments) and returns
      a tuple of Tensors.

  Returns:
    A wrapped fn that is identical to fn when called, but its activations will
    be discarded and recomputed on the backwards pass (i.e. on a call to
    tf.gradients).
  """

  @functools.wraps(fn)
  def wrapped(*args):
    return _recompute_grad(fn, args)

  return wrapped


def _recompute_grad(fn, args):
  """See recompute_grad."""

  def grad_fn(inputs, variables, outputs, output_grads):
    del outputs
    # recompute outputs
    outputs = list(fn(*inputs))
    grads = tf.gradients(outputs, inputs + variables, output_grads)
    grad_inputs = grads[:len(inputs)]
    grad_vars = grads[len(inputs):]
    return grad_inputs, grad_vars

  @fn_with_custom_grad(grad_fn)
  def fn_with_recompute(*args):
    return fn(*args)

  return fn_with_recompute(*args)


def ffn_self_attention_layer(x,
                             filter_depth,
                             output_depth,
                             is_training,
                             reuse,
                             num_parts,
                             dropout_rate,
                             share_kv=False,
                             name=None):
  """Self-attention feedforward layer.

  We use self-attention to do feedforward computations. We apply this function
  positionwise where for each position, we linearly transform the output to have
  depth filter_depth, and break up the result depth-wise into num_parts
  contiguous parts.  The parts self-attentd, we concatenate the results
  depth-wise, and we linearly transform to a depth of output_depth. The
  goal is to get multiplicative interactions between components of a
  representation.

  Args:
    x: a Tensor with shape [batch, length, channels]
    filter_depth: an integer
    output_depth: an integer
    num_parts: an integer dividing filter depth
    dropout_rate: a floating point number
    share_kv: Share the key value transform
    name: an optional string

  Returns:
    A Tensor.
  """

  with tf.variable_scope(name, default_name="feedforward_self_attention", values=[x]):
    x_shape = tf.shape(x)
    part_depth = filter_depth // num_parts
    if not share_kv:
      combined = conv1d(x, filter_depth * 3, is_training, reuse, filter_size=1, name="qkv_transform")
      combined = tf.expand_dims(combined, axis=2)
      q, k, v = tf.split(combined, 3, axis=3)
    else:
      q = tf.expand_dims(
          conv1d(x, filter_depth, is_training, reuse, filter_size=1, name="q_transform"), axis=2)
      kv_combined = tf.expand_dims(
          conv1d(
              tf.concat([x, x], axis=1),
              filter_depth,
              is_training,
              reuse,
              filter_size=1,
              name="kv_transform"),
          axis=2)
      k, v = tf.split(kv_combined, [x_shape[1], x_shape[1]], axis=1)

    batch_q = tf.reshape(q, [-1, 1, num_parts, part_depth])
    batch_k = tf.reshape(k, [-1, 1, num_parts, part_depth])
    batch_v = tf.reshape(v, [-1, 1, num_parts, part_depth])

    batch_q *= part_depth**-0.5
    # non-masked bias
    bias = None
    x = dot_product_attention(batch_q, batch_k, batch_v, bias, dropout_rate)
    x = tf.reshape(x, [x_shape[0], x_shape[1], filter_depth])
    x = conv1d(x, output_depth, is_training, reuse, filter_size=1, name="output_transform")
    return x


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True):
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.

  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    return tf.matmul(weights, v)


def fn_device_dependency_dict():
  """State container for fn_device_dependency."""
  if not hasattr(tf.get_default_graph(), "dependency_dict"):
    setattr(tf.get_default_graph(), "dependency_dict", defaultdict(list))
  return tf.get_default_graph().dependency_dict


@contextlib.contextmanager
def fn_device_dependency(name, device=""):
  """Add control deps for name and device."""
  key = name + "_" + device
  outs = []

  def body():
    with tf.control_dependencies(fn_device_dependency_dict()[key]):
      yield outs
      assert outs

      deps = outs
      if isinstance(outs[0], list) or isinstance(outs[0], tuple):
        assert len(outs) == 1
        deps = outs[0]
      fn_device_dependency_dict()[key] = deps

  if device:
    with tf.device(device):
      return body()
  else:
    return body()


def underlying_variable(t):
  """Find the underlying tf.Variable object.

  Args:
    t: a Tensor

  Returns:
    a tf.Varaible object.
  """
  t = variable_ref(t)
  assert t is not None
  # make sure that the graph has a variable index and that it is up-to-date
  if not hasattr(tf.get_default_graph(), "var_index"):
    tf.get_default_graph().var_index = {}
  var_index = tf.get_default_graph().var_index
  for v in tf.global_variables()[len(var_index):]:
    var_index[v.name] = v
  return var_index[t.name]


def clip_gradient(net, clip_value_min, clip_value_max, name='clip_gradient'):
  """Clips respective gradients of a given tensor. Acts as identity for the
  forward pass, but clips gradient tensor element-wise by value during the
  backward pass. Any gradient values less than `clip_value_min` or greater than
  `clip_values_max` are set to the respective limit values.

  Args:
    net: A `tf.Tensor`.
    clip_value_min: A 0-D Tensor or scalar. The minimum value to clip by.
    clip_value_max: A 0-D Tensor or scalar. The maximum value to clip by.
    name: A name for the operation (optional, default 'clip_gradient').

  Returns:
    A `tf.Tensor` with the same type as the input tensor.
  """

  def _clip_gradient_backward(unused_op, grad):
    return tf.clip_by_value(grad, clip_value_min, clip_value_max)

  @function.Defun(net.dtype, python_grad_func=_clip_gradient_backward, func_name="ClipGradient")
  def _clip_gradient_forward(x):
    return x

  with tf.name_scope(name, values=[net]):
    output = _clip_gradient_forward(net)
    output.set_shape(net.shape)
  return output


def scale_gradient(net, scale, name="scale_gradient"):
  """Scales gradients for the backwards pass.
    This might be used to, for example, allow one part of a model to learn at a
    lower rate than the rest.
    WARNING: Think carefully about how your optimizer works. If, for example, you
    use rmsprop, the gradient is always rescaled (with some additional epsilon)
    towards unity. This means `scale_gradient` won't have the effect of
    lowering the learning rate.
    If `scale` is `0.0`, this op reduces to `tf.stop_gradient`. If `scale`
    is `1.0`, this op reduces to `tf.identity`.

    Args:
      net: A `tf.Tensor`.
      scale: The scale factor for the gradient on the backwards pass.
      name: A name for the operation (optional).

    Returns:
      A `tf.Tensor` with the same type as the input tensor.
    """
  if scale == 0.0:
    return tf.stop_gradient(net, name=name)
  elif scale == 1.0:
    return tf.identity(net, name=name)
  else:

    def _scale_gradient_backward(unused, grad):
      return tf.multiply(tf.convert_to_tensor(scale), grad)

    @function.Defun(tf.float32, python_grad_func=_scale_gradient_backward, func_name="ScaleGradient")
    def _scale_gradient_forward(x):
      return x

    with tf.name_scope(name, values=[net]):
      output = _scale_gradient_forward(net, name=name)
      output.set_shape(net.shape)
    return output


def normalize_gradient(grad_scales=None, name='normalize_gradient'):
  if grad_scales is not None:
    grad_scales = np.float32(grad_scales)

  def _normalize_grad_backward(op, grad):
    grad_norm = tf.sqrt(tf.reduce_sum(grad**2, [1, 2, 3], keep_dims=True))
    if grad_scales is not None:
      grad *= grad_scales[:, None, None, None]
    return grad / grad_norm

  @function.Defun(
      tf.float32, python_grad_func=_normalize_grad_backward, func_name="NormalizeGradient")
  def _normalize_grad_forward(x):
    return x

  with tf.name_scope(name):
    output = _normalize_grad_forward(net, name=name)
    output.set_shape(net.shape)
  return output


def relu_density_logit(x, reduce_dims):
  """logit(density(x)).

  Useful for histograms.

  Args:
    x: a Tensor, typilcally the output of tf.relu
    reduce_dims: a list of dimensions

  Returns:
    a Tensor
  """
  frac = tf.reduce_mean(tf.to_float(x > 0.0), reduce_dims)
  scaled = tf.log(frac + math.exp(-10)) - \
      tf.log((1.0 - frac) + math.exp(-10))
  return scaled


def clip_variables(optimizer, variables, weight_clip):
  """Modifies an optimizer so it clips weights to a certain value.

  Args:
    optimizer: An optimizer to perform variable weight clipping.
    variables: A list of TensorFlow variables.
    weight_clip: Positive python float to clip discriminator weights. Used to
      enforce a K-lipschitz condition, which is useful for some GAN training
      schemes (ex WGAN: https://arxiv.org/pdf/1701.07875).

  Returns:
    An optimizer to perform weight clipping after updates.

  Raises:
    ValueError: If `weight_clip` is less than 0.
  """
  if weight_clip < 0:
    raise ValueError('`discriminator_weight_clip` must be positive. Instead, was %s', weight_clip)
  return VariableClippingOptimizer(
      opt=optimizer,
      # Do no reduction, so clipping happens per-value.
      vars_to_clip_dims={var: []
                         for var in variables},
      max_norm=weight_clip,
      use_locking=True,
      colocate_clip_ops_with_vars=True)


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def approximate_split(x, num_splits, axis=0):
  """Split approximately equally into num_splits parts.

  Args:
    x: a Tensor
    num_splits: an integer
    axis: an integer.

  Returns:
    a list of num_splits Tensors.
  """
  size = shape_list(x)[axis]
  size_splits = [tf.div(size + i, num_splits) for i in range(num_splits)]
  return tf.split(x, size_splits, axis=axis)


class FactoredTensor(object):
  """A concise factored representation of Tensor as two tensors.

  This class represents the tensor tf.matmul(a, b, transpose_b=True)
  by storing the values of Tensors a and b.

  The reason for this is that the product may be too big to fully realize at
  once, so it can be realized a part at a time.

  "a" may have extra leading dimensions, in which case they are flattened out
  before computing the matrix product, then re-expanded afterwards.
  """

  def __init__(self, a, b):
    self._a = a
    self._b = b

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  def to_tensor(self):
    """Convert to Tensor."""
    a_shape = shape_list(self.a)
    b_shape = shape_list(self.b)
    inner_dim = b_shape[1]
    result_dim = b_shape[0]
    flat_a = tf.reshape(self.a, [-1, inner_dim])
    product = tf.matmul(flat_a, self.b, transpose_b=True)
    product_shape = a_shape[:-1] + [result_dim]
    product = tf.reshape(product, product_shape)
    product.set_shape(self.a.get_shape().as_list()[:-1] + [self.b.get_shape()[0]])
    return product


def _convert_factored_tensor_to_tensor(value, *args, **kwargs):
  # call ops.convert_to_tensor to handle optional arguments appropriately
  return ops.internal_convert_to_tensor(value.to_tensor(), *args, **kwargs)


tf.register_tensor_conversion_function(FactoredTensor, _convert_factored_tensor_to_tensor)


def maybe_zero_out_padding(inputs, kernel_size, nonpadding_mask):
  """If necessary, zero out inputs to a conv for padding positions.

  Args:
    inputs: a Tensor with shape [batch, length, ...]
    kernel_size: an integer or pair of integers
    nonpadding_mask: a Tensor with shape [batch, length]

  Returns:
    a Tensor with the same shape as inputs
  """
  if (kernel_size != 1 and kernel_size != (1, 1) and nonpadding_mask is not None):
    while nonpadding_mask.get_shape().ndims < inputs.get_shape().ndims:
      nonpadding_mask = tf.expand_dims(nonpadding_mask, -1)
    return inputs * nonpadding_mask
  else:
    return inputs
