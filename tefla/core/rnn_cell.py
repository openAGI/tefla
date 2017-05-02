import tensorflow as tf
import six
import numpy as np
from collections import namedtuple
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from . import initializers as initz
from .layers import conv2d, dropout, softmax
from . import logger as log
from ..utils import util as helper

NamedOutputs = namedtuple('NamedOutputs', ['name', 'outputs'])
core_rnn_cell = tf.contrib.rnn


class BasicRNNCell(core_rnn_cell.RNNCell):
    """The most basic RNN cell

    Args:
        num_units: int, The number of units in the LSTM cell.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        input_size: Deprecated and unused.
        activation: Activation function of the states.
        layer_norm: If `True`, layer normalization will be applied.
        layer_norm_args: optional dict, layer_norm arguments
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        outputs_collections: The collections to which the outputs are added.
    """

    def __init__(self, num_units, reuse, trainable=True, input_size=None, activation=tf.tanh, layer_norm=None, layer_norm_args=None, outputs_collections=None):
        if input_size is not None:
            log.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self.reuse = reuse
        self.trainable = trainable
        self.layer_norm = layer_norm
        self.layer_norm_args = layer_norm_args or {}
        self.outputs_collections = outputs_collections

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope='basic_rnn_cell'):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        with tf.variable_scope(scope):
            output = self._activation(_linear(
                [inputs, state], self._num_units, self.reuse, trainable=self.trainable, name=scope))
            if self.layer_norm is not None:
                output = self.layer_norm(
                    output, self.reuse, trainable=self.trainable, **self.layer_norm_args)
            output = _collect_named_outputs(
                self.outputs_collections, scope + '_output', output)
        return output, output


class LSTMCell(core_rnn_cell.RNNCell):
    """LSTM unit

    This class adds layer normalization and recurrent dropout to a
    basic LSTM unit. Layer normalization implementation is based on:
    https://arxiv.org/abs/1607.06450.
    "Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    and is applied before the internal nonlinearities.
    Recurrent dropout is base on:
    https://arxiv.org/abs/1603.05118
    "Recurrent Dropout without Memory Loss"
    Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.

    Args:
        num_units: int, The number of units in the LSTM cell.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        forget_bias: float, The bias added to forget gates (see above).
        input_size: Deprecated and unused.
        activation: Activation function of the states.
        inner_activation: Activation function of the inner states.
        layer_norm: If `True`, layer normalization will be applied.
        layer_norm_args: optional dict, layer_norm arguments
        cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
        keep_prob: unit Tensor or float between 0 and 1 representing the
            recurrent dropout probability value. If float and 1.0, no dropout will
            be applied.
        dropout_seed: (optional) integer, the randomness seed.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        outputs_collections: The collections to which the outputs are added.
    """

    def __init__(self, num_units, reuse, trainable=True, forget_bias=1.0, input_size=None, activation=tf.tanh, inner_activation=tf.sigmoid, keep_prob=1.0, dropout_seed=None, cell_clip=None, layer_norm=None, layer_norm_args=None, outputs_collections=None):
        if input_size is not None:
            ValueError("%s: the input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self.layer_norm = layer_norm
        self.layer_norm_args = layer_norm_args or {}
        self._cell_clip = cell_clip
        self._activation = activation
        self._inner_activation = inner_activation
        self._keep_prob = keep_prob
        self._dropout_seed = dropout_seed
        self.trainable = trainable
        self.reuse = reuse
        self.outputs_collections = outputs_collections

    @property
    def state_size(self):
        return core_rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope='basiclstm'):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope):
            c, h = state
            concat = _linear([inputs, h], 4 * self._num_units, self.reuse,
                             trainable=self.trainable, use_bias=False, name=scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            try:
                i, j, f, o = tf.split(1, 4, concat)
            except Exception as e:
                print('Upgrade to recent version >= r.12 %s' % str(e.message))
                i, j, f, o = tf.split(concat, 4, 1)

            # apply batch normalization to inner state and gates
            if self.layer_norm is not None:
                i = self.layer_norm(
                    i, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                j = self.layer_norm(
                    j, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                f = self.layer_norm(
                    f, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                o = self.layer_norm(
                    o, self.reuse, trainable=self.trainable, **self.layer_norm_args)

            j = self._activation(j)
            if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
                j = dropout(j, self._keep_prob, seed=self._dropout_seed)
            new_c = (c * self._inner_activation(f + self._forget_bias) +
                     self._inner_activation(i) *
                     self._activation(j))

            if self._cell_clip is not None:
                new_c = tf.clip_by_value(
                    new_c, -self._cell_clip, self._cell_clip)

            if self.layer_norm is not None:
                layer_norm_new_c = self.layer_norm(
                    new_c, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                new_h = self._activation(
                    layer_norm_new_c) * self._inner_activation(o)
            else:
                new_h = self._activation(new_c) * self._inner_activation(o)

            new_state = core_rnn_cell.LSTMStateTuple(new_c, new_h)

            new_h = _collect_named_outputs(
                self.outputs_collections, scope + '_h', new_h)
            new_state = _collect_named_outputs(
                self.outputs_collections, scope + '_state', new_state)
            return new_h, new_state


class AttentionCell(core_rnn_cell.RNNCell):
    """Basic attention cell

    Implementation based on https://arxiv.org/abs/1409.0473.
    Create a cell with attention.

    Args:
        cell: an RNNCell, an attention is added to it.
            e.g.: a LSTMCell
        attn_length: integer, the size of an attention window.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        attn_size: integer, the size of an attention vector. Equal to
            cell.output_size by default.
        attn_vec_size: integer, the number of convolutional features calculated
            on attention state and a size of the hidden layer built from
            base cell state. Equal attn_size to by default.
        input_size: integer, the size of a hidden linear layer,
        layer_norm: If `True`, layer normalization will be applied.
        layer_norm_args: optional dict, layer_norm arguments
            built from inputs and attention. Derived from the input tensor by default.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        outputs_collections: The collections to which the outputs are added.

    Raises:
        TypeError: if cell is not an RNNCell.
        ValueError: Cell state should be tuple of states
    """

    def __init__(self, cell, attn_length, reuse, trainable=True, attn_size=None, attn_vec_size=None, input_size=None, layer_norm=None, layer_norm_args=None, outputs_collections=None):
        if not isinstance(cell, core_rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        if not helper.is_sequence(cell.state_size):
            raise ValueError("Cell state should be tuple of states")
        if attn_length <= 0:
            raise ValueError(
                "attn_length should be greater than zero, got %s" % str(attn_length))
        if attn_size is None:
            attn_size = cell.output_size
        if attn_vec_size is None:
            attn_vec_size = attn_size
        self._cell = cell
        self._attn_vec_size = attn_vec_size
        self._input_size = input_size
        self._attn_size = attn_size
        self._attn_length = attn_length
        self.layer_norm = layer_norm
        self.layer_norm_args = layer_norm_args or {}
        self.reuse = reuse
        self.trainable = trainable
        self.outputs_collections = outputs_collections

    @property
    def state_size(self):
        return (self._cell.state_size, self._attn_size, self._attn_size * self._attn_length)

    @property
    def output_size(self):
        return self._attn_size

    def __call__(self, inputs, state, scope='attention_cell'):
        """Long short-term memory cell with attention (LSTMA)."""
        with tf.variable_scope(scope, reuse=self.reuse):
            state, attns, attn_states = state
            attn_states = tf.reshape(
                attn_states, [-1, self._attn_length, self._attn_size])
            input_size = self._input_size
            if input_size is None:
                input_size = inputs.get_shape().as_list()[1]
            inputs = _linear([inputs, attns], input_size,
                             self.reuse, trainable=self.trainable, name=scope)
            lstm_output, new_state = self._cell(inputs, state)
            new_state_cat = tf.concat(helper.flatten_sq(new_state), 1)
            new_attns, new_attn_states = _attention(
                new_state_cat, attn_states, True, self.reuse, self._attn_size, self._attn_vec_size, self._attn_length, trainable=self.trainable)
            with tf.variable_scope("attn_output_projection"):
                output = _linear([lstm_output, new_attns], self._attn_size,
                                 self.reuse, trainable=self.trainable, name=scope)
            new_attn_states = tf.concat(
                [new_attn_states, tf.expand_dims(output, 1)], 1)
            new_attn_states = tf.reshape(
                new_attn_states, [-1, self._attn_length * self._attn_size])
            new_state = (new_state, new_attns, new_attn_states)
            if self.layer_norm is not None:
                output = self.layer_norm(
                    output, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                new_state = self.layer_norm(
                    new_state, self.reuse, trainable=self.trainable, **self.layer_norm_args)

            output = _collect_named_outputs(
                self.outputs_collections, scope + '_output', output)
            new_state = _collect_named_outputs(
                self.outputs_collections, scope + '_state', new_state)
            return output, new_state


class GRUCell(core_rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)

    Args:
        num_units: int, The number of units in the LSTM cell.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        input_size: Deprecated and unused.
        activation: Activation function of the states.
        inner_activation: Activation function of the inner states.
        layer_norm: If `True`, layer normalization will be applied.
        layer_norm_args: optional dict, layer_norm arguments
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        outputs_collections: The collections to which the outputs are added.
    """

    def __init__(self, num_units, reuse, trainable=True, input_size=None, activation=tf.tanh, inner_activation=tf.sigmoid, b_init=1.0, layer_norm=None, layer_norm_args=None, outputs_collections=None):
        if input_size is not None:
            log.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self.resue = reuse
        self.trainable = trainable
        self._activation = activation
        self._b_init = b_init
        self._inner_activation = inner_activation
        self.layer_norm = layer_norm
        self.layer_norm_args = layer_norm_args or {}

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope='gru_cell'):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                try:
                    r, u = tf.split(2, 1, value=_linear(
                        [inputs, state], 2 * self._num_units, self.reuse, b_init=self._b_init, trainable=self.trainable, name=scope))
                except Exception as e:
                    print('Upgrade to recent version >= r.12 %s' %
                          str(e.message))
                    r, u = tf.split(_linear([inputs, state], 2 * self._num_units, self.reuse,
                                            b_init=self._b_init, trainable=self.trainable, name=scope), 2, 1)
                r, u = self._inner_activation(r), self._inner_activation(u)
                if self.layer_norm is not None:
                    u = self.layer_norm(
                        u, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                    r = self.layer_norm(
                        r, self.reuse, trainable=self.trainable, **self.layer_norm_args)
            with tf.variable_scope("candidate"):
                c = self._activation(
                    _linear([inputs, r * state], self._num_units, True, name=scope))
            new_h = u * state + (1 - u) * c
            if self.layer_norm is not None:
                c = self.layer_norm(
                    c, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                new_h = self.layer_norm(
                    new_h, self.reuse, trainable=self.trainable)

            new_h = _collect_named_outputs(
                self.outputs_collections, scope + '_h', new_h)
        return new_h, new_h


class MultiRNNCell(core_rnn_cell.RNNCell):
    """RNN cell composed sequentially of multiple simple cells

    Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
        cells: list of RNNCells that will be composed in this order.

    Raises:
        ValueError: Cell state should be tuple of states
    """

    def __init__(self, cells, state_is_tuple=True):
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self._cells = cells
        if any(helper.is_sequence(c.state_size) for c in self._cells):
            raise ValueError("Cell state should be tuple of states")

        @property
        def state_size(self):
            return sum([cell.state_size for cell in self._cells])

        @property
        def output_size(self):
            return self._cells[-1].output_size

        def __call__(self, inputs, state, scope='multi_rnn_cell'):
            """Run this multi-layer cell on inputs, starting from state."""
            with tf.variable_scope(scope):
                cur_inp = inputs
                new_states = []
                for i, cell in enumerate(self._cells):
                    with tf.variable_scope("cell_%d" % i):
                        if not helper.is_sequence(state):
                            raise ValueError("Expected state to be a tuple of length %d, but received: %s" % (
                                len(self.state_size), state))
                        cur_state = state[i]
                        cur_inp, new_state = cell(cur_inp, cur_state)
                        new_states.append(new_state)
            new_states = (
                tuple(new_states) if self._state_is_tuple else tf.concat(new_states, 1))
            return cur_inp, new_states


class DropoutWrapper(core_rnn_cell.RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell

    Create a cell with added input and/or output dropout.
    Dropout is never used on the state.

    Args:
        cell: an RNNCell, a projection to output_size is added to it.
        is_training: a bool, training if true else validation/testing
        input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is float and 1, no input dropout will be added.
        output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is float and 1, no output dropout will be added.
        seed: (optional) integer, the randomness seed.

    Raises:
        TypeError: if cell is not an RNNCell.
        ValueError: if keep_prob is not between 0 and 1.
    """

    def __init__(self, cell, is_training, input_keep_prob=1.0, output_keep_prob=1.0, seed=None):
        if not isinstance(cell, core_rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if not any(keep_prob >= 0.0 and keep_prob <= 1.0 for keep_prob in [input_keep_prob, output_keep_prob]):
            raise ValueError(
                "Parameter input/output_keep_prob must be float, between 0 and 1")
        self._cell = cell
        self._is_training = is_training
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""

        if (not isinstance(self._input_keep_prob, float) or self._input_keep_prob < 1):
            inputs = _dropout(inputs, self._is_training,
                              keep_prob=self._input_keep_prob, seed=self._seed)
        output, new_state = self._cell(inputs, state)
        if (not isinstance(self._output_keep_prob, float) or self._output_keep_prob < 1):
            output = _dropout(output, self._is_training,
                              keep_prob=self._output_keep_prob, seed=self._seed)
        return output, new_state


def _linear(x, n_output, reuse, trainable=True, w_init=initz.he_normal(), b_init=0.0, w_regularizer=tf.nn.l2_loss, name='fc', layer_norm=None, layer_norm_args=None, activation=None, outputs_collections=None, use_bias=True):
    """Adds a fully connected layer.

        `fully_connected` creates a variable called `weights`, representing a fully
        connected weight matrix, which is multiplied by the `x` to produce a
        `Tensor` of hidden units. If a `layer_norm` is provided (such as
        `layer_norm`), it is then applied. Otherwise, if `layer_norm` is
        None and a `b_init` and `use_bias` is provided then a `biases` variable would be
        created and added the hidden units. Finally, if `activation` is not `None`,
        it is applied to the hidden units as well.
        Note: that if `x` have a rank greater than 2, then `x` is flattened
        prior to the initial matrix multiply by `weights`.

    Args:
        x: A `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
        n_output: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        layer_norm: normalization function to use. If
           `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        layer_norm_args: normalization function parameters.
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
        ValueError: linear is expecting 2D arguments
        ValueError: linear expects shape[1] to be provided for shape
    """
    if not (isinstance(n_output, six.integer_types)):
        raise ValueError('n_output should be int or long, got %s.', n_output)

    if not helper.is_sequence(x):
        x = [x]

    n_input = 0
    shapes = [_x.get_shape() for _x in x]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError(
                "linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
        else:
            n_input += shape[1].value

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
        if len(x) == 1:
            output = tf.matmul(x[0], W)
        else:
            output = tf.matmul(tf.concat(x, 1), W)

        if use_bias:
            b = tf.get_variable(
                name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(b_init),
                trainable=trainable,
            )

            output = tf.nn.bias_add(value=output, bias=b)

        if layer_norm is not None:
            layer_norm_args = layer_norm_args or {}
            output = layer_norm(output, reuse=reuse,
                                trainable=trainable, **layer_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)

        return _collect_named_outputs(outputs_collections, name, output)


def layer_norm(x, reuse, center=True, scale=True, trainable=True, epsilon=1e-12, name='bn', outputs_collections=None):
    """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.
    "Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    Can be used as a normalizer function for conv2d and fully_connected.

    Args:
        x: a tensor with 2 or more dimensions, where the first dimension has
            `batch_size`. The normalization is over all but the last dimension if
            `data_format` is `NHWC` and the second dimension if `data_format` is
            `NCHW`.
        center: If True, subtract `beta`. If False, `beta` is ignored.
        scale: If True, multiply by `gamma`. If False, `gamma` is
            not used. When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling can be done by the next layer.
        epsilon: small float added to variance to avoid dividing by zero.
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
    """
    with tf.variable_scope(name, reuse=reuse):
        x = tf.convert_to_tensor(x)
        x_shape = x.get_shape()
        x_rank = x_shape.ndims
        if x_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % x.name)
        axis = list(range(1, x_rank))
        beta, gamma = None, None
        if center:
            beta = tf.get_variable(name='beta', initializer=tf.constant(
                0.0, shape=[x.get_shape()[-1:]]), trainable=trainable)
        if scale:
            gamma = tf.get_variable(name='gamma', initializer=tf.constant(
                1.0, shape=[x.get_shape()[-1:]]), trainable=trainable)

        mean, variance = tf.nn.moments(x, axis, keep_dims=True)
        output = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, epsilon)
        output.set_shape(x_shape)
        return _collect_named_outputs(outputs_collections, name, output)


def _attention(query, attn_states, is_training, reuse, attn_size, attn_vec_size, attn_length, trainable=True, name='attention'):
    with tf.variable_scope(name, reuse=reuse):
        v = tf.get_variable(
            name="V", shape=[attn_vec_size], trainable=trainable)
        attn_states_reshaped = tf.reshape(
            attn_states, shape=[-1, attn_length, 1, attn_size])
        attn_conv = conv2d(attn_states_reshaped, attn_vec_size, is_training, reuse, filter_size=(
            1, 1), stride=(1, 1), trainable=trainable, use_bias=False)
        y = _linear(query, attn_vec_size, reuse)
        y = tf.reshape(y, [-1, 1, 1, attn_vec_size])
        s = tf.reduce_sum(v * tf.tanh(attn_conv + y), [2, 3])
        a = softmax(s)
        d = tf.reduce_sum(tf.reshape(
            a, [-1, attn_length, 1, 1]) * attn_states_reshaped, [1, 2])
        new_attns = tf.reshape(d, [-1, attn_size])
        new_attn_states = tf.slice(attn_states, [0, 1, 0], [-1, -1, -1])
        return new_attns, new_attn_states


def _dropout(x, is_training, keep_prob=0.5, seed=None):
    return tf.cond(is_training, lambda: tf.nn.dropout(x, tf.cast(keep_prob, tf.float32), seed=seed), lambda: x)


def _flatten(x, name='flatten'):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    with tf.name_scope(name):
        dims = int(np.prod(input_shape[1:]))
        flattened = tf.reshape(x, [-1, dims])
        return flattened


def _collect_named_outputs(outputs_collections, name, output):
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, NamedOutputs(name, output))
    return output
