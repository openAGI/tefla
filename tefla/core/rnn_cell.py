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
    """

    def __init__(self, num_units, reuse, trainable=True, w_init=initz.he_normal(), use_bias=False, input_size=None, activation=tf.tanh, layer_norm=None, layer_norm_args=None):
        if input_size is not None:
            log.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._w_init = w_init
        self._use_bias = use_bias
        self.reuse = reuse
        self.trainable = trainable
        self.layer_norm = layer_norm
        self.layer_norm_args = layer_norm_args or {}

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
                [inputs, state], self._num_units, self.reuse, w_init=self._w_init, use_bias=self._use_bias, trainable=self.trainable, name=scope))
            if self.layer_norm is not None:
                output = self.layer_norm(
                    output, self.reuse, trainable=self.trainable, **self.layer_norm_args)
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
    """

    def __init__(self, num_units, reuse, trainable=True, w_init=initz.he_normal(), forget_bias=1.0, use_bias=False, input_size=None, activation=tf.tanh, inner_activation=tf.sigmoid, keep_prob=1.0, dropout_seed=None, cell_clip=None, layer_norm=None, layer_norm_args=None):
        if input_size is not None:
            ValueError("%s: the input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._use_bias = use_bias
        self._w_init = w_init
        self.layer_norm = layer_norm
        self.layer_norm_args = layer_norm_args or {}
        self._cell_clip = cell_clip
        self._activation = activation
        self._inner_activation = inner_activation
        self._keep_prob = keep_prob
        self._dropout_seed = dropout_seed
        self.trainable = trainable
        self.reuse = reuse

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
                             trainable=self.trainable, w_init=self._w_init, use_bias=self._use_bias, name=scope)

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

    Raises:
        TypeError: if cell is not an RNNCell.
        ValueError: Cell state should be tuple of states
    """

    def __init__(self, cell, attn_length, reuse, w_init=initz.he_normal(), use_bias=False, trainable=True, attn_size=None, attn_vec_size=None, input_size=None, layer_norm=None, layer_norm_args=None):
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
        self._w_init = w_init
        self._use_bias = use_bias
        self.layer_norm = layer_norm
        self.layer_norm_args = layer_norm_args or {}
        self.reuse = reuse
        self.trainable = trainable

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
                             self.reuse, w_init=self._w_init, use_bias=self._use_bias, trainable=self.trainable, name=scope)
            lstm_output, new_state = self._cell(inputs, state)
            new_state_cat = tf.concat(helper.flatten_sq(new_state), 1)
            new_attns, new_attn_states = _attention(
                new_state_cat, attn_states, True, self.reuse, self._attn_size, self._attn_vec_size, self._attn_length, trainable=self.trainable)
            with tf.variable_scope("attn_output_projection"):
                output = _linear([lstm_output, new_attns], self._attn_size,
                                 self.reuse, w_init=self._w_init, use_bias=self._use_bias, trainable=self.trainable, name=scope)
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
    """

    def __init__(self, num_units, reuse, w_init=initz.he_normal(), use_bias=False, trainable=True, input_size=None, activation=tf.tanh, inner_activation=tf.sigmoid, b_init=1.0, layer_norm=None, layer_norm_args=None):
        if input_size is not None:
            log.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self.resue = reuse
        self.trainable = trainable
        self._activation = activation
        self._b_init = b_init
        self._w_init = w_init
        self._use_bias = use_bias
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
                        [inputs, state], 2 * self._num_units, self.reuse, w_init=self._w_init, b_init=self._b_init, use_bias=self._use_bias, trainable=self.trainable, name=scope))
                except Exception as e:
                    print('Upgrade to recent version >= r.12 %s' %
                          str(e.message))
                    r, u = tf.split(_linear([inputs, state], 2 * self._num_units, self.reuse,
                                            w_init=self._w_init, b_init=self._b_init, use_bias=self._use_bias, trainable=self.trainable, name=scope), 2, 1)
                r, u = self._inner_activation(r), self._inner_activation(u)
                if self.layer_norm is not None:
                    u = self.layer_norm(
                        u, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                    r = self.layer_norm(
                        r, self.reuse, trainable=self.trainable, **self.layer_norm_args)
            with tf.variable_scope("candidate"):
                c = self._activation(
                    _linear([inputs, r * state], self._num_units, True, w_init=self._w_init, b_init=self._b_init, use_bias=self._use_bias, name=scope))
            new_h = u * state + (1 - u) * c
            if self.layer_norm is not None:
                c = self.layer_norm(
                    c, self.reuse, trainable=self.trainable, **self.layer_norm_args)
                new_h = self.layer_norm(
                    new_h, self.reuse, trainable=self.trainable)

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
            raise ValueError(
                "Must specify at least one cell for MultiRNNCell.")
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


class ExtendedMultiRNNCell(MultiRNNCell):
    """Extends the Tensorflow MultiRNNCell with residual connections"""

    def __init__(self,
                 cells,
                 residual_connections=False,
                 residual_combiner="add",
                 residual_dense=False):
        """Create a RNN cell composed sequentially of a number of RNNCells.

        Args:
          cells: list of RNNCells that will be composed in this order.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.
          residual_connections: If true, add residual connections between all cells.
            This requires all cells to have the same output_size. Also, iff the
            input size is not equal to the cell output size, a linear transform
            is added before the first layer.
          residual_combiner: One of "add" or "concat". To create inputs for layer
            t+1 either "add" the inputs from the prev layer or concat them.
          residual_dense: Densely connect each layer to all other layers

        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(ExtendedMultiRNNCell, self).__init__(cells, state_is_tuple=True)
        assert residual_combiner in ["add", "concat", "mean"]

        self._residual_connections = residual_connections
        self._residual_combiner = residual_combiner
        self._residual_dense = residual_dense

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        if not self._residual_connections:
            return super(ExtendedMultiRNNCell, self).__call__(
                inputs, state, (scope or "extended_multi_rnn_cell"))

        with tf.variable_scope(scope or "extended_multi_rnn_cell"):
            # Adding Residual connections are only possible when input and output
            # sizes are equal. Optionally transform the initial inputs to
            # `cell[0].output_size`
            if self._cells[0].output_size != inputs.get_shape().as_list()[1] and \
                    (self._residual_combiner in ["add", "mean"]):
                inputs = tf.contrib.layers.fully_connected(
                    inputs=inputs,
                    num_outputs=self._cells[0].output_size,
                    activation_fn=None,
                    scope="input_transform")

            # Iterate through all layers (code from MultiRNNCell)
            cur_inp = inputs
            prev_inputs = [cur_inp]
            new_states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("cell_%d" % i):
                    if not helper.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, but received: %s" %
                            (len(self.state_size), state))
                    cur_state = state[i]
                    next_input, new_state = cell(cur_inp, cur_state)

                    # Either combine all previous inputs or only the current
                    # input
                    input_to_combine = prev_inputs[-1:]
                    if self._residual_dense:
                        input_to_combine = prev_inputs

                    # Add Residual connection
                    if self._residual_combiner == "add":
                        next_input = next_input + sum(input_to_combine)
                    if self._residual_combiner == "mean":
                        combined_mean = tf.reduce_mean(
                            tf.stack(input_to_combine), 0)
                        next_input = next_input + combined_mean
                    elif self._residual_combiner == "concat":
                        next_input = tf.concat(
                            [next_input] + input_to_combine, 1)
                    cur_inp = next_input
                    prev_inputs.append(cur_inp)

                    new_states.append(new_state)
        new_states = (tuple(new_states)
                      if self._state_is_tuple else tf.concat(new_states, 1))
        return cur_inp, new_states


class GLSTMCell(core_rnn_cell.RNNCell):
    """Group LSTM cell (G-LSTM).
    The implementation is based on:
    https://arxiv.org/abs/1703.10722
    O. Kuchaiev and B. Ginsburg
    "Factorization Tricks for LSTM Networks", ICLR 2017 workshop.
    """

    def __init__(self, num_units, reuse, w_init=initz.he_normal(), b_init=0.0, use_bias=False, initializer=None, num_proj=None, number_of_groups=1, forget_bias=1.0, activation=tf.tanh):
        """Initialize the parameters of G-LSTM cell.

        Args:
            num_units: int, The number of units in the G-LSTM cell
            initializer: (optional) The initializer to use for the weight and
                projection matrices.
            num_proj: (optional) int, The output dimensionality for the projection
                matrices.  If None, no projection is performed.
            number_of_groups: (optional) int, number of groups to use.
                If `number_of_groups` is 1, then it should be equivalent to LSTM cell
            forget_bias: Biases of the forget gate are initialized by default to 1
                in order to reduce the scale of forgetting at the beginning of
                the training.
            activation: Activation function of the inner states.
            reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.  If not `True`, and the existing scope already
                has the given variables, an error is raised.

        Raises:
            ValueError: If `num_units` or `num_proj` is not divisible by
                `number_of_groups`.
        """
        self._num_units = num_units
        self._initializer = initializer
        self._num_proj = num_proj
        self._forget_bias = forget_bias
        self._activation = activation
        self._number_of_groups = number_of_groups
        self._w_init = w_init
        self._b_init = b_init
        self._use_bias = use_bias
        self.reuse = reuse

        if self._num_units % self._number_of_groups != 0:
            raise ValueError("num_units must be divisible by number_of_groups")
        if self._num_proj:
            if self._num_proj % self._number_of_groups != 0:
                raise ValueError(
                    "num_proj must be divisible by number_of_groups")
            self._group_shape = [int(self._num_proj / self._number_of_groups),
                                 int(self._num_units / self._number_of_groups)]
        else:
            self._group_shape = [int(self._num_units / self._number_of_groups),
                                 int(self._num_units / self._number_of_groups)]

        if num_proj:
            self._state_size = core_rnn_cell.LSTMStateTuple(
                num_units, num_proj)
            self._output_size = num_proj
        else:
            self._state_size = core_rnn_cell.LSTMStateTuple(
                num_units, num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def _get_input_for_group(self, inputs, group_id, group_size):
        """Slices inputs into groups to prepare for processing by cell's groups

        Args:
            inputs: cell input or it's previous state,
               a Tensor, 2D, [batch x num_units]
            group_id: group id, a Scalar, for which to prepare input
            group_size: size of the group

        Returns:
            subset of inputs corresponding to group "group_id",
            a Tensor, 2D, [batch x num_units/number_of_groups]
        """
        return tf.slice(input_=inputs, begin=[0, group_id * group_size], size=[self._batch_size, group_size], name=("GLSTM_group%d_input_generation" % group_id))

    def __call__(self, inputs, state, scope='glstm_cell'):
        """Run one step of G-LSTM.

        Args:
            inputs: input Tensor, 2D, [batch x num_units].
            state: this must be a tuple of state Tensors, both `2-D`,
                with column sizes `c_state` and `m_state`.

        Returns:
            A tuple containing:
                - A `2-D, [batch x output_dim]`, Tensor representing the output of the
                G-LSTM after reading `inputs` when previous state was `state`.
                Here output_dim is: num_proj if num_proj was set, num_units otherwise.
                - LSTMStateTuple representing the new state of G-LSTM  cell
                after reading `inputs` when the previous state was `state`.

        Raises:
            ValueError: If input size cannot be inferred from inputs via static shape inference.
        """
        (c_prev, m_prev) = state

        self._batch_size = inputs.shape[0].value or tf.shape(inputs)[0]
        dtype = inputs.dtype
        with tf.variable_scope(scope, initializer=self._initializer):
            i_parts = []
            j_parts = []
            f_parts = []
            o_parts = []

            for group_id in range(self._number_of_groups):
                with tf.variable_scope("group%d" % group_id):
                    x_g_id = tf.concat([self._get_input_for_group(inputs, group_id, self._group_shape[
                                       0]), self._get_input_for_group(m_prev, group_id, self._group_shape[0])], axis=1)
                    R_k = _linear(
                        x_g_id, 4 * self._group_shape[1], self.reuse, w_init=self._w_init, b_init=self._b_init, use_bias=self._use_bias, name=scope + "group%d" % group_id)
                    i_k, j_k, f_k, o_k = tf.split(R_k, 4, 1)

                i_parts.append(i_k)
                j_parts.append(j_k)
                f_parts.append(f_k)
                o_parts.append(o_k)

            bi = tf.get_variable(name="bias_i",
                                 shape=[self._num_units],
                                 dtype=dtype,
                                 initializer=tf.constant_initializer(0.0, dtype=dtype))
            bj = tf.get_variable(name="bias_j",
                                 shape=[self._num_units],
                                 dtype=dtype,
                                 initializer=tf.constant_initializer(0.0, dtype=dtype))
            bf = tf.get_variable(name="bias_f",
                                 shape=[self._num_units],
                                 dtype=dtype,
                                 initializer=tf.constant_initializer(0.0, dtype=dtype))
            bo = tf.get_variable(name="bias_o",
                                 shape=[self._num_units],
                                 dtype=dtype,
                                 initializer=tf.constant_initializer(0.0, dtype=dtype))

            i = tf.nn.bias_add(tf.concat(i_parts, axis=1), bi)
            j = tf.nn.bias_add(tf.concat(j_parts, axis=1), bj)
            f = tf.nn.bias_add(tf.concat(f_parts, axis=1), bf)
            o = tf.nn.bias_add(tf.concat(o_parts, axis=1), bo)

        c = (tf.sigmoid(f + self._forget_bias) *
             c_prev + tf.sigmoid(i) * tf.tanh(j))
        m = tf.sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            with tf.variable_scope("projection"):
                m = _linear(m, self._num_proj, self.reuse, w_init=self._w_init,
                            b_init=self._b_init, use_bias=self._use_bias, name=scope + 'projection')

        new_state = core_rnn_cell.LSTMStateTuple(c, m)
        return m, new_state


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


def _rnn_wrapper(inputs, cell, reuse, is_training, dropout=None, return_seq=False,
                 return_state=False, initial_state=None, dynamic=False,
                 scope="rnn_wrapper"):
    """ RNN cell Wrapper. """
    sequence_length = None
    if dynamic:
        sequence_length = helper.retrieve_seq_length(
            inputs if isinstance(inputs, tf.Tensor) else tf.stack(inputs))

    input_shape = helper.get_input_shape(inputs)

    with tf.variable_scope(scope, reuse=reuse) as scope:
        name = scope.name
        if dropout:
            if type(dropout) in [tuple, list]:
                input_keep_prob = dropout[0]
                output_keep_prob = dropout[1]
            elif isinstance(dropout, float):
                input_keep_prob, output_keep_prob = dropout, dropout
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                "float)")
            cell = DropoutWrapper(
                cell, is_training, input_keep_prob, output_keep_prob)

        outputs = inputs
        if type(outputs) not in [list, np.array]:
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            outputs = tf.transpose(outputs, (axes))
            outputs = tf.unstack(outputs)

        outputs, state = core_rnn_cell.static_rnn(cell, outputs, dtype=tf.float32,
                                                  initial_state=initial_state, scope=name,
                                                  sequence_length=sequence_length)

    if dynamic:
        if return_seq:
            o = outputs
        else:
            outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
            o = helper.advanced_indexing(outputs, sequence_length)
    else:
        o = outputs if return_seq else outputs[-1]

    return (o, state) if return_state else o


def lstm(inputs, n_units, reuse, is_training, activation=tf.tanh, inner_activation=tf.sigmoid,
         dropout=None, use_bias=True, w_init=initz.he_normal(), forget_bias=1.0,
         return_seq=False, return_state=False, initial_state=None,
         dynamic=True, trainable=True, scope='lstm'):
    """ LSTM.
    Long Short Term Memory Recurrent Layer.

    Args:
        inputs: `Tensor`. Inputs 3-D Tensor [samples, timesteps, input_dim].
        n_units: `int`, number of units for this layer.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        is_training: `bool`, training if True
        activation: `function` (returning a `Tensor`).
        inner_activation: `function` (returning a `Tensor`).
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        use_bias: `bool`. If True, a bias is used.
        w_init: `function` (returning a `Tensor`). Weights initialization.
        forget_bias: `float`. Bias of the forget gate. Default: 1.0.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        dynamic: `bool`. If True, dynamic computation is performed. It will not
            compute RNN steps above the sequence length. Note that because TF
            requires to feed sequences of same length, 0 is used as a mask.
            So a sequence padded with 0 at the end must be provided. When
            computation is performed, it will stop when it meets a step with
            a value of 0.
        trainable: `bool`. If True, weights will be trainable.
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.

    Returns:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].
    """
    cell = LSTMCell(n_units, reuse, activation=activation,
                    inner_activation=inner_activation,
                    forget_bias=forget_bias, use_bias=use_bias,
                    w_init=w_init, trainable=trainable)
    x = _rnn_wrapper(inputs, cell, reuse, is_training, dropout=dropout,
                     return_seq=return_seq, return_state=return_state,
                     initial_state=initial_state, dynamic=dynamic,
                     scope=scope)

    return x


def gru(inputs, n_units, reuse, is_training, activation=tf.tanh, inner_activation=tf.sigmoid,
        dropout=None, use_bias=True, w_init=initz.he_normal(), forget_bias=1.0,
        return_seq=False, return_state=False, initial_state=None,
        dynamic=True, trainable=True, scope='gru'):
    """ GRU.
    Gated Recurrent Layer.

    Args:
        inputs: `Tensor`. Inputs 3-D Tensor [samples, timesteps, input_dim].
        n_units: `int`, number of units for this layer.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        is_training: `bool`, training if True
        activation: `function` (returning a `Tensor`).
        inner_activation: `function` (returning a `Tensor`).
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        use_bias: `bool`. If True, a bias is used.
        w_init: `function` (returning a `Tensor`). Weights initialization.
        forget_bias: `float`. Bias of the forget gate. Default: 1.0.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        dynamic: `bool`. If True, dynamic computation is performed. It will not
            compute RNN steps above the sequence length. Note that because TF
            requires to feed sequences of same length, 0 is used as a mask.
            So a sequence padded with 0 at the end must be provided. When
            computation is performed, it will stop when it meets a step with
            a value of 0.
        trainable: `bool`. If True, weights will be trainable.
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.

    Returns:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].
    """
    cell = GRUCell(n_units, reuse, activation=activation,
                   inner_activation=inner_activation,
                   forget_bias=forget_bias, use_bias=use_bias,
                   w_init=w_init, trainable=trainable)
    x = _rnn_wrapper(inputs, cell, reuse, is_training, dropout=dropout,
                     return_seq=return_seq, return_state=return_state,
                     initial_state=initial_state, dynamic=dynamic,
                     scope=scope)

    return x


def bidirectional_rnn(inputs, rnncell_fw, rnncell_bw, reuse, is_training, dropout_fw=None, dropout_bw=None, return_seq=False,
                      return_states=False, initial_state_fw=None,
                      initial_state_bw=None, dynamic=False, scope="BiRNN", outputs_collections=None):
    """ Bidirectional RNN.
    Build a bidirectional recurrent neural network, it requires 2 RNN Cells
    to process sequence in forward and backward order. Any RNN Cell can be
    used i.e. SimpleRNN, LSTM, GRU... with its own parameters. But the two
    cells number of units must match.

    Args:
        inputs: `Tensor`. The 3D inputs Tensor [samples, timesteps, input_dim].
        rnncell_fw: `RNNCell`. The RNN Cell to use for foward computation.
        rnncell_bw: `RNNCell`. The RNN Cell to use for backward computation.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        is_training: `bool`, training if True
        dropout_fw: `tuple` of `float`: (input_keep_prob, output_keep_prob). the
            input and output keep probability.
        dropout_bw: `tuple` of `float`: (input_keep_prob, output_keep_prob). the
            input and output keep probability.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_states: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state_fw: `Tensor`. An initial state for the forward RNN.
            This must be a tensor of appropriate type and shape [batch_size
            x cell.state_size].
        initial_state_bw: `Tensor`. An initial state for the backward RNN.
            This must be a tensor of appropriate type and shape [batch_size
            x cell.state_size].
        dynamic: `bool`. If True, dynamic computation is performed. It will not
            compute RNN steps above the sequence length. Note that because TF
            requires to feed sequences of same length, 0 is used as a mask.
            So a sequence padded with 0 at the end must be provided. When
            computation is performed, it will stop when it meets a step with
            a value of 0.
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.

    Returns:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor Layer [samples, output dim].
    """
    assert (rnncell_fw._num_units == rnncell_bw._num_units), \
        "RNN Cells number of units must match!"

    sequence_length = None
    if dynamic:
        sequence_length = helper.retrieve_seq_length(
            inputs if isinstance(inputs, tf.Tensor) else tf.stack(inputs))

    input_shape = helper.get_input_shape(inputs)

    with tf.variable_scope(scope, reuse=reuse) as scope:
        name = scope.name
        if dropout_fw:
            if type(dropout_fw) in [tuple, list]:
                input_keep_prob = dropout_fw[0]
                output_keep_prob = dropout_fw[1]
            elif isinstance(dropout_fw, float):
                input_keep_prob, output_keep_prob = dropout_fw, dropout_fw
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                "float)")
            rnncell_fw = DropoutWrapper(
                rnncell_fw, is_training, input_keep_prob, output_keep_prob)
        if dropout_bw:
            if type(dropout_bw) in [tuple, list]:
                input_keep_prob = dropout_bw[0]
                output_keep_prob = dropout_bw[1]
            elif isinstance(dropout_bw, float):
                input_keep_prob, output_keep_prob = dropout_bw, dropout_bw
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                "float)")
            rnncell_bw = DropoutWrapper(
                rnncell_bw, is_training, input_keep_prob, output_keep_prob)

        if type(inputs) not in [list, np.array]:
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            inputs = tf.transpose(inputs, (axes))
            inputs = tf.unstack(inputs)

        outputs, states_fw, states_bw = core_rnn_cell.static_bidirectional_rnn(
            rnncell_fw, rnncell_bw, inputs,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=sequence_length,
            dtype=tf.float32)

    if dynamic:
        if return_seq:
            o = outputs
        else:
            outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
            o = helper.advanced_indexing(outputs, sequence_length)
    else:
        o = outputs if return_seq else outputs[-1]

    sfw = states_fw
    sbw = states_bw

    return (o, sfw, sbw) if return_states else o
