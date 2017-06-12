from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple
import copy
from pydoc import locate

from .rnn_cell import LSTMCell

import six
rnn_cell = tf.contrib.rnn
EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


@six.add_metaclass(abc.ABCMeta)
class Encoder(GraphModule, Configurable):
    """Abstract encoder class. All encoders should inherit from this.

    Args:
      params: A dictionary of hyperparameters for the encoder.
      name: A variable scope for the encoder graph.
    """

    def __init__(self, params, mode, name):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)

    def _build(self, inputs, *args, **kwargs):
        return self.encode(inputs, *args, **kwargs)

    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        """
        Encodes an input sequence.

        Args:
          inputs: The inputs to encode. A float32 tensor of shape [B, T, ...].
          sequence_length: The length of each input. An int32 tensor of shape [T].

        Returns:
          An `EncoderOutput` tuple containing the outputs and final state.
        """
        raise NotImplementedError


class ConvEncoder(Encoder):
    """A deep convolutional encoder, as described in
    https://arxiv.org/abs/1611.02344. The encoder supports optional positions
    embeddings.

    Params:
      attention_cnn.units: Number of units in `cnn_a`. Same in each layer.
      attention_cnn.kernel_size: Kernel size for `cnn_a`.
      attention_cnn.layers: Number of layers in `cnn_a`.
      embedding_dropout_keep_prob: Dropout keep probability
        applied to the embeddings.
      output_cnn.units: Number of units in `cnn_c`. Same in each layer.
      output_cnn.kernel_size: Kernel size for `cnn_c`.
      output_cnn.layers: Number of layers in `cnn_c`.
      position_embeddings.enable: If true, add position embeddings to the
        inputs before pooling.
      position_embeddings.combiner_fn: Function used to combine the
        position embeddings with the inputs. For example, `tensorflow.add`.
      position_embeddings.num_positions: Size of the position embedding matrix.
        This should be set to the maximum sequence length of the inputs.
    """

    def __init__(self, params, mode, name="conv_encoder"):
        super(ConvEncoder, self).__init__(params, mode, name)
        self._combiner_fn = locate(
            self.params["position_embeddings.combiner_fn"])

    @staticmethod
    def default_params():
        return {
            "attention_cnn.units": 512,
            "attention_cnn.kernel_size": 3,
            "attention_cnn.layers": 15,
            "embedding_dropout_keep_prob": 0.8,
            "output_cnn.units": 256,
            "output_cnn.kernel_size": 3,
            "output_cnn.layers": 5,
            "position_embeddings.enable": True,
            "position_embeddings.combiner_fn": "tensorflow.multiply",
            "position_embeddings.num_positions": 100,
        }

    def encode(self, inputs, sequence_length):
        if self.params["position_embeddings.enable"]:
            positions_embed = _create_position_embedding(
                embedding_dim=inputs.get_shape().as_list()[-1],
                num_positions=self.params["position_embeddings.num_positions"],
                lengths=sequence_length,
                maxlen=tf.shape(inputs)[1])
            inputs = self._combiner_fn(inputs, positions_embed)

        # Apply dropout to embeddings
        inputs = tf.contrib.layers.dropout(
            inputs=inputs,
            keep_prob=self.params["embedding_dropout_keep_prob"],
            is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)

        with tf.variable_scope("cnn_a"):
            cnn_a_output = inputs
            for layer_idx in range(self.params["attention_cnn.layers"]):
                next_layer = tf.contrib.layers.conv2d(
                    inputs=cnn_a_output,
                    num_outputs=self.params["attention_cnn.units"],
                    kernel_size=self.params["attention_cnn.kernel_size"],
                    padding="SAME",
                    activation_fn=None)
                # Add a residual connection, except for the first layer
                if layer_idx > 0:
                    next_layer += cnn_a_output
                cnn_a_output = tf.tanh(next_layer)

        with tf.variable_scope("cnn_c"):
            cnn_c_output = inputs
            for layer_idx in range(self.params["output_cnn.layers"]):
                next_layer = tf.contrib.layers.conv2d(
                    inputs=cnn_c_output,
                    num_outputs=self.params["output_cnn.units"],
                    kernel_size=self.params["output_cnn.kernel_size"],
                    padding="SAME",
                    activation_fn=None)
                # Add a residual connection, except for the first layer
                if layer_idx > 0:
                    next_layer += cnn_c_output
                cnn_c_output = tf.tanh(next_layer)

        final_state = tf.reduce_mean(cnn_c_output, 1)

        return EncoderOutput(
            outputs=cnn_a_output,
            final_state=final_state,
            attention_values=cnn_c_output,
            attention_values_length=sequence_length)


class PoolingEncoder(Encoder):
    """An encoder that pools over embeddings, as described in
    https://arxiv.org/abs/1611.02344. The encoder supports optional positions
    embeddings and a configurable pooling window.

    Params:
      dropout_keep_prob: Dropout keep probability applied to the embeddings.
      pooling_fn: The 1-d pooling function to use, e.g.
        `tensorflow.layers.average_pooling1d`.
      pool_size: The pooling window, passed as `pool_size` to
        the pooling function.
      strides: The stride during pooling, passed as `strides`
        the pooling function.
      position_embeddings.enable: If true, add position embeddings to the
        inputs before pooling.
      position_embeddings.combiner_fn: Function used to combine the
        position embeddings with the inputs. For example, `tensorflow.add`.
      position_embeddings.num_positions: Size of the position embedding matrix.
        This should be set to the maximum sequence length of the inputs.
    """

    def __init__(self, params, mode, name="pooling_encoder"):
        super(PoolingEncoder, self).__init__(params, mode, name)
        self._pooling_fn = locate(self.params["pooling_fn"])
        self._combiner_fn = locate(
            self.params["position_embeddings.combiner_fn"])

    @staticmethod
    def default_params():
        return {
            "dropout_keep_prob": 0.8,
            "pooling_fn": "tensorflow.layers.average_pooling1d",
            "pool_size": 5,
            "strides": 1,
            "position_embeddings.enable": True,
            "position_embeddings.combiner_fn": "tensorflow.multiply",
            "position_embeddings.num_positions": 100,
        }

    def encode(self, inputs, sequence_length):
        if self.params["position_embeddings.enable"]:
            positions_embed = _create_position_embedding(
                embedding_dim=inputs.get_shape().as_list()[-1],
                num_positions=self.params["position_embeddings.num_positions"],
                lengths=sequence_length,
                maxlen=tf.shape(inputs)[1])
            inputs = self._combiner_fn(inputs, positions_embed)

        # Apply dropout
        inputs = tf.contrib.layers.dropout(
            inputs=inputs,
            keep_prob=self.params["dropout_keep_prob"],
            is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)

        outputs = self._pooling_fn(
            inputs=inputs,
            pool_size=self.params["pool_size"],
            strides=self.params["strides"],
            padding="SAME")

        # Final state is the average representation of the pooled embeddings
        final_state = tf.reduce_mean(outputs, 1)

        return EncoderOutput(
            outputs=outputs,
            final_state=final_state,
            attention_values=inputs,
            attention_values_length=sequence_length)


class UnidirectionalRNNEncoder(Encoder):
    """
    A unidirectional RNN encoder. Stacking should be performed as
    part of the cell.

    Args:
      cell: An instance of tf.contrib.rnn.RNNCell
      name: A name for the encoder
    """

    def __init__(self, params, mode, name="forward_rnn_encoder"):
        super(UnidirectionalRNNEncoder, self).__init__(params, mode, name)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def encode(self, inputs, sequence_length, **kwargs):
        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        cell = _get_rnn_cell(**self.params["rnn_cell"])
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)
        return EncoderOutput(
            outputs=outputs,
            final_state=state,
            attention_values=outputs,
            attention_values_length=sequence_length)


class BidirectionalRNNEncoder(Encoder):
    """
    A bidirectional RNN encoder. Uses the same cell for both the
    forward and backward RNN. Stacking should be performed as part of
    the cell.

    Args:
      cell: An instance of tf.contrib.rnn.RNNCell
      name: A name for the encoder
    """

    def __init__(self, params, mode, name="bidi_rnn_encoder"):
        super(BidirectionalRNNEncoder, self).__init__(params, mode, name)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def encode(self, inputs, sequence_length, **kwargs):
        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        cell_fw = _get_rnn_cell(**self.params["rnn_cell"])
        cell_bw = _get_rnn_cell(**self.params["rnn_cell"])
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)

        # Concatenate outputs and states of the forward and backward RNNs
        outputs_concat = tf.concat(outputs, 2)

        return EncoderOutput(
            outputs=outputs_concat,
            final_state=states,
            attention_values=outputs_concat,
            attention_values_length=sequence_length)


class StackBidirectionalRNNEncoder(Encoder):
    """
    A stacked bidirectional RNN encoder. Uses the same cell for both the
    forward and backward RNN. Stacking should be performed as part of
    the cell.

    Args:
      cell: An instance of tf.contrib.rnn.RNNCell
      name: A name for the encoder
    """

    def __init__(self, params, mode, name="stacked_bidi_rnn_encoder"):
        super(StackBidirectionalRNNEncoder, self).__init__(params, mode, name)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def encode(self, inputs, sequence_length, **kwargs):
        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        cell_fw = _get_rnn_cell(**self.params["rnn_cell"])
        cell_bw = _get_rnn_cell(**self.params["rnn_cell"])

        cells_fw = _unpack_cell(cell_fw)
        cells_bw = _unpack_cell(cell_bw)

        result = rnn_cell.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=inputs,
            dtype=tf.float32,
            sequence_length=sequence_length,
            **kwargs)
        outputs_concat, _output_state_fw, _output_state_bw = result
        final_state = (_output_state_fw, _output_state_bw)
        return EncoderOutput(
            outputs=outputs_concat,
            final_state=final_state,
            attention_values=outputs_concat,
            attention_values_length=sequence_length)


@six.add_metaclass(abc.ABCMeta)
class Configurable(object):
    """Interface for all classes that are configurable
    via a parameters dictionary.

    Args:
      params: A dictionary of parameters.
      mode: A value in tf.contrib.learn.ModeKeys
    """

    def __init__(self, params, mode):
        self._params = _parse_params(params, self.default_params())
        self._mode = mode
        self._print_params()

    def _print_params(self):
        """Logs parameter values"""
        classname = self.__class__.__name__
        tf.logging.info("Creating %s in mode=%s", classname, self._mode)
        tf.logging.info("\n%s", yaml.dump({classname: self._params}))

    @property
    def mode(self):
        """Returns a value in tf.contrib.learn.ModeKeys.
        """
        return self._mode

    @property
    def params(self):
        """Returns a dictionary of parsed parameters.
        """
        return self._params

    @abstractstaticmethod
    def default_params():
        """Returns a dictionary of default parameters. The default parameters
        are used to define the expected type of passed parameters. Missing
        parameter values are replaced with the defaults returned by this method.
        """
        raise NotImplementedError


class abstractstaticmethod(staticmethod):  # pylint: disable=C0111,C0103
    """Decorates a method as abstract and static"""
    __slots__ = ()

    def __init__(self, function):
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class GraphModule(object):
    """
    Convenience class that makes it easy to share variables.
    Each insance of this class creates its own set of variables, but
    each subsequent execution of an instance will re-use its variables.

    Graph components that define variables should inherit from this class
    and implement their logic in the `_build` method.
    """

    def __init__(self, name):
        """
        Initialize the module. Each subclass must call this constructor with a name.

        Args:
          name: Name of this module. Used for `tf.make_template`.
        """
        self.name = name
        self._template = tf.make_template(
            name, self._build, create_scope_now_=True)
        # Docstrings for the class should be the docstring for the _build method
        self.__doc__ = self._build.__doc__
        # pylint: disable=E1101
        self.__call__.__func__.__doc__ = self._build.__doc__

    def _build(self, *args, **kwargs):
        """Subclasses should implement their logic here.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # pylint: disable=missing-docstring
        return self._template(*args, **kwargs)

    def variable_scope(self):
        """Returns the proper variable scope for this module.
        """
        return tf.variable_scope(self._template.variable_scope)


def _parse_params(params, default_params):
    """Parses parameter values to the types defined by the default parameters.
    Default parameters are used for missing values.
    """
    # Cast parameters to correct types
    if params is None:
        params = {}
    result = copy.deepcopy(default_params)
    for key, value in params.items():
        # If param is unknown, drop it to stay compatible with past versions
        if key not in default_params:
            raise ValueError("%s is not a valid model parameter" % key)
        # Param is a dictionary
        if isinstance(value, dict):
            default_dict = default_params[key]
            if not isinstance(default_dict, dict):
                raise ValueError("%s should not be a dictionary", key)
            if default_dict:
                value = _parse_params(value, default_dict)
            else:
                # If the default is an empty dict we do not typecheck it
                # and assume it's done downstream
                pass
        if value is None:
            continue
        if default_params[key] is None:
            result[key] = value
        else:
            result[key] = type(default_params[key])(value)
    return result


def _unpack_cell(cell):
    """Unpack the cells because the stack_bidirectional_dynamic_rnn
    expects a list of cells, one per layer."""
    if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
        return cell._cells  # pylint: disable=W0212
    else:
        return [cell]


def _default_rnn_cell_params():
    """Creates default parameters used by multiple RNN encoders.
    """
    return {
        "cell_class": LSTMCell,
        "cell_params": {
            "num_units": 128
        },
        "dropout_input_keep_prob": 1.0,
        "dropout_output_keep_prob": 1.0,
        "num_layers": 1,
        "residual_connections": False,
        "residual_combiner": "add",
        "residual_dense": False
    }


def _toggle_dropout(cell_params, mode):
    """Disables dropout during eval/inference mode
    """
    cell_params = copy.deepcopy(cell_params)
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        cell_params["dropout_input_keep_prob"] = 1.0
        cell_params["dropout_output_keep_prob"] = 1.0
    return cell_params


def _position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 of
    End-To-End Memory Networks (https://arxiv.org/abs/1503.08895).

    Args:
      sentence_size: length of the sentence
      embedding_size: dimensionality of the embeddings

    Returns:
      A numpy array of shape [sentence_size, embedding_size] containing
      the fixed position encodings for each sentence position.
    """
    encoding = np.ones((sentence_size, embedding_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for k in range(1, le):
        for j in range(1, ls):
            encoding[j - 1, k - 1] = (1.0 - j / float(ls)) - (
                k / float(le)) * (1. - 2. * j / float(ls))
    return encoding


def _create_position_embedding(embedding_dim, num_positions, lengths, maxlen):
    """Creates position embeddings.

    Args:
      embedding_dim: Dimensionality of the embeddings. An integer.
      num_positions: The number of positions to be embedded. For example,
        if you have inputs of length up to 100, this should be 100. An integer.
      lengths: The lengths of the inputs to create position embeddings for.
        An int32 tensor of shape `[batch_size]`.
      maxlen: The maximum length of the input sequence to create position
        embeddings for. An int32 tensor.

    Returns:
      A tensor of shape `[batch_size, maxlen, embedding_dim]` that contains
      embeddings for each position. All elements past `lengths` are zero.
    """
    # Create constant position encodings
    position_encodings = tf.constant(
        _position_encoding(num_positions, embedding_dim),
        name="position_encoding")

    # Slice to size of current sequence
    pe_slice = position_encodings[:maxlen, :]
    # Replicate encodings for each element in the batch
    batch_size = tf.shape(lengths)[0]
    pe_batch = tf.tile([pe_slice], [batch_size, 1, 1])

    # Mask out positions that are padded
    positions_mask = tf.sequence_mask(
        lengths=lengths, maxlen=maxlen, dtype=tf.float32)
    positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)

    return positions_embed


def _get_rnn_cell(cell_class,
                  cell_params,
                  num_layers=1,
                  dropout_input_keep_prob=1.0,
                  dropout_output_keep_prob=1.0,
                  residual_connections=False,
                  residual_combiner="add",
                  residual_dense=False):
    """Creates a new RNN Cell

    Args:
      cell_class: Name of the cell class, e.g. "BasicLSTMCell".
      cell_params: A dictionary of parameters to pass to the cell constructor.
      num_layers: Number of layers. The cell will be wrapped with
        `tf.contrib.rnn.MultiRNNCell`
      dropout_input_keep_prob: Dropout keep probability applied
        to the input of cell *at each layer*
      dropout_output_keep_prob: Dropout keep probability applied
        to the output of cell *at each layer*
      residual_connections: If true, add residual connections
        between all cells

    Returns:
      An instance of `tf.contrib.rnn.RNNCell`.
    """

    cells = []
    for _ in range(num_layers):
        cell = cell_class(**cell_params)
        if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=dropout_input_keep_prob,
                output_keep_prob=dropout_output_keep_prob)
        cells.append(cell)

    if len(cells) > 1:
        final_cell = ExtendedMultiRNNCell(
            cells=cells,
            residual_connections=residual_connections,
            residual_combiner=residual_combiner,
            residual_dense=residual_dense)
    else:
        final_cell = cells[0]

    return final_cell
