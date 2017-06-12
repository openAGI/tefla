from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.framework import function  # pylint: disable=E0611
from tensorflow.python.util import nest

from .encoder import GraphModule, Configurable, _toggle_dropout, _default_rnn_cell_params, _get_rnn_cell
from ..utils.seq2seq_utils import CustomHelper
from . import beam_search


class DecoderOutput(
        namedtuple("DecoderOutput", ["logits", "predicted_ids", "cell_output"])):
    """Output of an RNN decoder.

    Note that we output both the logits and predictions because during
    dynamic decoding the predictions may not correspond to max(logits).
    For example, we may be sampling from the logits instead.
    """
    pass


class AttentionDecoderOutput(
        namedtuple("DecoderOutput", [
            "logits", "predicted_ids", "cell_output", "attention_scores",
            "attention_context"
        ])):
    """Augmented decoder output that also includes the attention scores.
    """
    pass


class FinalBeamDecoderOutput(
    namedtuple("FinalBeamDecoderOutput",
               ["predicted_ids", "beam_search_output"])):
    """Final outputs returned by the beam search after all decoding is finished.

    Args:
      predicted_ids: The final prediction. A tensor of shape
        `[T, 1, beam_width]`.
      beam_search_output: An instance of `BeamDecoderOutput` that describes
        the state of the beam search.
    """
    pass


@six.add_metaclass(abc.ABCMeta)
class RNNDecoder(Decoder, GraphModule, Configurable):
    """Base class for RNN decoders.

    Args:
      cell: An instance of `RNNCell`
      helper: An instance of `seq2seq_Helper` to assist decoding
      initial_state: A tensor or tuple of tensors used as the initial cell
        state.
      name: A name for this module
    """

    def __init__(self, params, mode, name):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)
        self.cell = _get_rnn_cell(**self.params["rnn_cell"])
        # Not initialized yet
        self.initial_state = None
        self.helper = None

    @abc.abstractmethod
    def initialize(self, name=None):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, name=None):
        raise NotImplementedError

    @property
    def batch_size(self):
        return tf.shape(nest.flatten([self.initial_state])[0])[0]

    def _setup(self, initial_state, helper):
        """Sets the initial state and helper for the decoder.
        """
        self.initial_state = initial_state
        self.helper = helper

    def finalize(self, outputs, final_state):
        """Applies final transformation to the decoder output once decoding is
        finished.
        """
        # pylint: disable=R0201
        return (outputs, final_state)

    @staticmethod
    def default_params():
        return {
            "max_decode_length": 100,
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def _build(self, initial_state, helper):
        if not self.initial_state:
            self._setup(initial_state, helper)

        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        maximum_iterations = None
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            maximum_iterations = self.params["max_decode_length"]

        outputs, final_state = dynamic_decode(
            decoder=self,
            output_time_major=True,
            impute_finished=False,
            maximum_iterations=maximum_iterations)
        return self.finalize(outputs, final_state)


class BasicDecoder(RNNDecoder):
    """Simple RNN decoder that performed a softmax operations on the cell output.
    """

    def __init__(self, params, mode, vocab_size, name="basic_decoder"):
        super(BasicDecoder, self).__init__(params, mode, name)
        self.vocab_size = vocab_size

    def compute_output(self, cell_output):
        """Computes the decoder outputs."""
        return tf.contrib.layers.fully_connected(
            inputs=cell_output, num_outputs=self.vocab_size, activation_fn=None)

    @property
    def output_size(self):
        return DecoderOutput(
            logits=self.vocab_size,
            predicted_ids=tf.TensorShape([]),
            cell_output=self.cell.output_size)

    @property
    def output_dtype(self):
        return DecoderOutput(
            logits=tf.float32, predicted_ids=tf.int32, cell_output=tf.float32)

    def initialize(self, name=None):
        finished, first_inputs = self.helper.initialize()
        return finished, first_inputs, self.initial_state

    def step(self, time_, inputs, state, name=None):
        cell_output, cell_state = self.cell(inputs, state)
        logits = self.compute_output(cell_output)
        sample_ids = self.helper.sample(
            time=time_, outputs=logits, state=cell_state)
        outputs = DecoderOutput(
            logits=logits, predicted_ids=sample_ids, cell_output=cell_output)
        finished, next_inputs, next_state = self.helper.next_inputs(
            time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)
        return (outputs, next_state, next_inputs, finished)


@function.Defun(
    tf.float32,
    tf.float32,
    tf.float32,
    func_name="att_sum_bahdanau",
    noinline=True)
def att_sum_bahdanau(v_att, keys, query):
    """Calculates a batch- and timweise dot product with a variable"""
    return tf.reduce_sum(v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])


@function.Defun(tf.float32, tf.float32, func_name="att_sum_dot", noinline=True)
def att_sum_dot(keys, query):
    """Calculates a batch- and timweise dot product"""
    return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])


@six.add_metaclass(abc.ABCMeta)
class AttentionLayer(GraphModule, Configurable):
    """
    Attention layer according to https://arxiv.org/abs/1409.0473.

    Params:
      num_units: Number of units used in the attention layer
    """

    def __init__(self, params, mode, name="attention"):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)

    @staticmethod
    def default_params():
        return {"num_units": 128}

    @abc.abstractmethod
    def score_fn(self, keys, query):
        """Computes the attention score"""
        raise NotImplementedError

    def _build(self, query, keys, values, values_length):
        """Computes attention scores and outputs.

        Args:
          query: The query used to calculate attention scores.
            In seq2seq this is typically the current state of the decoder.
            A tensor of shape `[B, ...]`
          keys: The keys used to calculate attention scores. In seq2seq, these
            are typically the outputs of the encoder and equivalent to `values`.
            A tensor of shape `[B, T, ...]` where each element in the `T`
            dimension corresponds to the key for that value.
          values: The elements to compute attention over. In seq2seq, this is
            typically the sequence of encoder outputs.
            A tensor of shape `[B, T, input_dim]`.
          values_length: An int32 tensor of shape `[B]` defining the sequence
            length of the attention values.

        Returns:
          A tuple `(scores, context)`.
          `scores` is vector of length `T` where each element is the
          normalized "score" of the corresponding `inputs` element.
          `context` is the final attention layer output corresponding to
          the weighted inputs.
          A tensor fo shape `[B, input_dim]`.
        """
        values_depth = values.get_shape().as_list()[-1]

        # Fully connected layers to transform both keys and query
        # into a tensor with `num_units` units
        att_keys = tf.contrib.layers.fully_connected(
            inputs=keys,
            num_outputs=self.params["num_units"],
            activation_fn=None,
            scope="att_keys")
        att_query = tf.contrib.layers.fully_connected(
            inputs=query,
            num_outputs=self.params["num_units"],
            activation_fn=None,
            scope="att_query")

        scores = self.score_fn(att_keys, att_query)

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(scores)[1]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

        # Normalize the scores
        scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the scores
        context = tf.expand_dims(scores_normalized, 2) * values
        context = tf.reduce_sum(context, 1, name="context")
        context.set_shape([None, values_depth])

        return (scores_normalized, context)


class AttentionLayerDot(AttentionLayer):
    """An attention layer that calculates attention scores using
    a dot product.
    """

    def score_fn(self, keys, query):
        return att_sum_dot(keys, query)


class AttentionLayerBahdanau(AttentionLayer):
    """An attention layer that calculates attention scores using
    a parameterized multiplication."""

    def score_fn(self, keys, query):
        v_att = tf.get_variable(
            "v_att", shape=[self.params["num_units"]], dtype=tf.float32)
        return att_sum_bahdanau(v_att, keys, query)


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
    """An RNN Decoder abstract interface object."""

    @property
    def batch_size(self):
        """The batch size of the inputs returned by `sample`."""
        raise NotImplementedError

    @property
    def output_size(self):
        """A (possibly nested tuple of...) integer[s] or `TensorShape` object[s]."""
        raise NotImplementedError

    @property
    def output_dtype(self):
        """A (possibly nested tuple of...) dtype[s]."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self, name=None):
        """Called before any decoding iterations.

        Args:
          name: Name scope for any created operations.

        Returns:
          `(finished, first_inputs, initial_state)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, time, inputs, state, name=None):
        """Called per step of decoding (but only once for dynamic decoding).

        Args:
          time: Scalar `int32` tensor.
          inputs: Input (possibly nested tuple of) tensor[s] for this time step.
          state: State (possibly nested tuple of) tensor[s] from previous time step.
          name: Name scope for any created operations.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        raise NotImplementedError


class AttentionDecoder(RNNDecoder):
    """An RNN Decoder that uses attention over an input sequence.

    Args:
      cell: An instance of ` tf.contrib.rnn.RNNCell`
      helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
      initial_state: A tensor or tuple of tensors used as the initial cell
        state.
      vocab_size: Output vocabulary size, i.e. number of units
        in the softmax layer
      attention_keys: The sequence used to calculate attention scores.
        A tensor of shape `[B, T, ...]`.
      attention_values: The sequence to attend over.
        A tensor of shape `[B, T, input_dim]`.
      attention_values_length: Sequence length of the attention values.
        An int32 Tensor of shape `[B]`.
      attention_fn: The attention function to use. This function map from
        `(state, inputs)` to `(attention_scores, attention_context)`.
        For an example, see `seq2seq.decoder.attention.AttentionLayer`.
      reverse_scores: Optional, an array of sequence length. If set,
        reverse the attention scores in the output. This is used for when
        a reversed source sequence is fed as an input but you want to
        return the scores in non-reversed order.
    """

    def __init__(self,
                 params,
                 mode,
                 vocab_size,
                 attention_keys,
                 attention_values,
                 attention_values_length,
                 attention_fn,
                 reverse_scores_lengths=None,
                 name="attention_decoder"):
        super(AttentionDecoder, self).__init__(params, mode, name)
        self.vocab_size = vocab_size
        self.attention_keys = attention_keys
        self.attention_values = attention_values
        self.attention_values_length = attention_values_length
        self.attention_fn = attention_fn
        self.reverse_scores_lengths = reverse_scores_lengths

    @property
    def output_size(self):
        return AttentionDecoderOutput(
            logits=self.vocab_size,
            predicted_ids=tf.TensorShape([]),
            cell_output=self.cell.output_size,
            attention_scores=tf.shape(self.attention_values)[1:-1],
            attention_context=self.attention_values.get_shape()[-1])

    @property
    def output_dtype(self):
        return AttentionDecoderOutput(
            logits=tf.float32,
            predicted_ids=tf.int32,
            cell_output=tf.float32,
            attention_scores=tf.float32,
            attention_context=tf.float32)

    def initialize(self, name=None):
        finished, first_inputs = self.helper.initialize()

        # Concat empty attention context
        attention_context = tf.zeros([
            tf.shape(first_inputs)[0],
            self.attention_values.get_shape().as_list()[-1]
        ])
        first_inputs = tf.concat([first_inputs, attention_context], 1)

        return finished, first_inputs, self.initial_state

    def compute_output(self, cell_output):
        """Computes the decoder outputs."""

        # Compute attention
        att_scores, attention_context = self.attention_fn(
            query=cell_output,
            keys=self.attention_keys,
            values=self.attention_values,
            values_length=self.attention_values_length)

        # TODO: Make this a parameter: We may or may not want this.
        # Transform attention context.
        # This makes the softmax smaller and allows us to synthesize information
        # between decoder state and attention context
        # see https://arxiv.org/abs/1508.04025v5
        softmax_input = tf.contrib.layers.fully_connected(
            inputs=tf.concat([cell_output, attention_context], 1),
            num_outputs=self.cell.output_size,
            activation_fn=tf.nn.tanh,
            scope="attention_mix")

        # Softmax computation
        logits = tf.contrib.layers.fully_connected(
            inputs=softmax_input,
            num_outputs=self.vocab_size,
            activation_fn=None,
            scope="logits")

        return softmax_input, logits, att_scores, attention_context

    def _setup(self, initial_state, helper):
        self.initial_state = initial_state

        def att_next_inputs(time, outputs, state, sample_ids, name=None):
            """Wraps the original decoder helper function to append the attention
            context.
            """
            finished, next_inputs, next_state = helper.next_inputs(
                time=time,
                outputs=outputs,
                state=state,
                sample_ids=sample_ids,
                name=name)
            next_inputs = tf.concat([next_inputs, outputs.attention_context], 1)
            return (finished, next_inputs, next_state)

        self.helper = CustomHelper(
            initialize_fn=helper.initialize,
            sample_fn=helper.sample,
            next_inputs_fn=att_next_inputs)

    def step(self, time_, inputs, state, name=None):
        cell_output, cell_state = self.cell(inputs, state)
        cell_output_new, logits, attention_scores, attention_context = \
            self.compute_output(cell_output)

        if self.reverse_scores_lengths is not None:
            attention_scores = tf.reverse_sequence(
                input=attention_scores,
                seq_lengths=self.reverse_scores_lengths,
                seq_dim=1,
                batch_dim=0)

        sample_ids = self.helper.sample(
            time=time_, outputs=logits, state=cell_state)

        outputs = AttentionDecoderOutput(
            logits=logits,
            predicted_ids=sample_ids,
            cell_output=cell_output_new,
            attention_scores=attention_scores,
            attention_context=attention_context)

        finished, next_inputs, next_state = self.helper.next_inputs(
            time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

        return (outputs, next_state, next_inputs, finished)


class BeamDecoderOutput(
        namedtuple("BeamDecoderOutput", [
            "logits", "predicted_ids", "log_probs", "scores", "beam_parent_ids",
            "original_outputs"
        ])):
    """Structure for the output of a beam search decoder. This class is used
    to define the output at each step as well as the final output of the decoder.
    If used as the final output, a time dimension `T` is inserted after the
    beam_size dimension.

    Args:
      logits: Logits at the current time step of shape `[beam_size, vocab_size]`
      predicted_ids: Chosen softmax predictions at the current time step.
        An int32 tensor of shape `[beam_size]`.
      log_probs: Total log probabilities of all beams at the current time step.
        A float32 tensor of shaep `[beam_size]`.
      scores: Total scores of all beams at the current time step. This differs
        from log probabilities in that the score may add additional processing
        such as length normalization. A float32 tensor of shape `[beam_size]`.
      beam_parent_ids: The indices of the beams that are being continued.
        An int32 tensor of shape `[beam_size]`.
    """
    pass


class BeamSearchDecoder(RNNDecoder):
    """The BeamSearchDecoder wraps another decoder to perform beam search instead
    of greedy selection. This decoder must be used with batch size of 1, which
    will result in an effective batch size of `beam_width`.

    Args:
      decoder: A instance of `RNNDecoder` to be used with beam search.
      config: A `BeamSearchConfig` that defines beam search decoding parameters.
    """

    def __init__(self, decoder, config):
        super(BeamSearchDecoder, self).__init__(decoder.params, decoder.mode,
                                                decoder.name)
        self.decoder = decoder
        self.config = config

    def __call__(self, *args, **kwargs):
        with self.decoder.variable_scope():
            return self._build(*args, **kwargs)

    @property
    def output_size(self):
        return BeamDecoderOutput(
            logits=self.decoder.vocab_size,
            predicted_ids=tf.TensorShape([]),
            log_probs=tf.TensorShape([]),
            scores=tf.TensorShape([]),
            beam_parent_ids=tf.TensorShape([]),
            original_outputs=self.decoder.output_size)

    @property
    def output_dtype(self):
        return BeamDecoderOutput(
            logits=tf.float32,
            predicted_ids=tf.int32,
            log_probs=tf.float32,
            scores=tf.float32,
            beam_parent_ids=tf.int32,
            original_outputs=self.decoder.output_dtype)

    @property
    def batch_size(self):
        return self.config.beam_width

    def initialize(self, name=None):
        finished, first_inputs, initial_state = self.decoder.initialize()

        # Create beam state
        beam_state = beam_search.create_initial_beam_state(config=self.config)
        return finished, first_inputs, (initial_state, beam_state)

    def finalize(self, outputs, final_state):
        # Gather according to beam search result
        predicted_ids = beam_search.gather_tree(outputs.predicted_ids,
                                                outputs.beam_parent_ids)

        # We're using a batch size of 1, so we add an extra dimension to
        # convert tensors to [1, beam_width, ...] shape. This way Tensorflow
        # doesn't confuse batch_size with beam_width
        outputs = nest.map_structure(lambda x: tf.expand_dims(x, 1), outputs)

        final_outputs = FinalBeamDecoderOutput(
            predicted_ids=tf.expand_dims(predicted_ids, 1),
            beam_search_output=outputs)

        return final_outputs, final_state

    def _build(self, initial_state, helper):
        # Tile initial state
        initial_state = nest.map_structure(
            lambda x: tf.tile(x, [self.batch_size, 1]), initial_state)
        self.decoder._setup(initial_state, helper)  # pylint: disable=W0212
        return super(BeamSearchDecoder, self)._build(self.decoder.initial_state,
                                                     self.decoder.helper)

    def step(self, time_, inputs, state, name=None):
        decoder_state, beam_state = state

        # Call the original decoder
        (decoder_output, decoder_state, _, _) = self.decoder.step(time_, inputs,
                                                                  decoder_state)

        # Perform a step of beam search
        bs_output, beam_state = beam_search.beam_search_step(
            time_=time_,
            logits=decoder_output.logits,
            beam_state=beam_state,
            config=self.config)

        # Shuffle everything according to beam search result
        decoder_state = nest.map_structure(
            lambda x: tf.gather(x, bs_output.beam_parent_ids), decoder_state)
        decoder_output = nest.map_structure(
            lambda x: tf.gather(x, bs_output.beam_parent_ids), decoder_output)

        next_state = (decoder_state, beam_state)

        outputs = BeamDecoderOutput(
            logits=tf.zeros([self.config.beam_width, self.config.vocab_size]),
            predicted_ids=bs_output.predicted_ids,
            log_probs=beam_state.log_probs,
            scores=bs_output.scores,
            beam_parent_ids=bs_output.beam_parent_ids,
            original_outputs=decoder_output)

        finished, next_inputs, next_state = self.decoder.helper.next_inputs(
            time=time_,
            outputs=decoder_output,
            state=next_state,
            sample_ids=bs_output.predicted_ids)
        next_inputs.set_shape([self.batch_size, None])

        return (outputs, next_state, next_inputs, finished)


def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
    """Perform dynamic decoding with `decoder`.

    Args:
      decoder: A `Decoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major).  If
        `True`, outputs are returned as time major tensors (this mode is faster).
        Otherwise, outputs are returned as batch major tensors (this adds extra
        time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: `int32` scalar, maximum allowed number of decoding
         steps.  Default is `None` (decode until the decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      scope: Optional variable scope to use.

    Returns:
      `(final_outputs, final_state)`.

    Raises:
      TypeError: if `decoder` is not an instance of `Decoder`.
      ValueError: if maximum_iterations is provided but is not a scalar.
    """
    if not isinstance(decoder, Decoder):
        raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                        type(decoder))

    with tf.variable_scope(scope or "decoder") as varscope:
        # Properly cache variable values inside the while_loop
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = tf.convert_to_tensor(
                maximum_iterations, dtype=tf.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state = decoder.initialize()

        zero_outputs = _create_zero_outputs(decoder.output_size,
                                            decoder.output_dtype,
                                            decoder.batch_size)

        if maximum_iterations is not None:
            initial_finished = tf.logical_or(
                initial_finished, 0 >= maximum_iterations)
        initial_time = tf.constant(0, dtype=tf.int32)

        def _shape(batch_size, from_shape):
            if not isinstance(from_shape, tf.TensorShape):
                return tf.TensorShape(None)
            else:
                batch_size = tf.contrib.util.constant_value(
                    tf.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tf.TensorShape([batch_size]).concatenate(from_shape)

        def _create_ta(s, d):
            return tf.TensorArray(
                dtype=d,
                size=0,
                dynamic_size=True,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                                decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                      finished):
            return tf.logical_not(tf.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished):
            """Internal while_loop body.

            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: 1-D bool tensor.

            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished)`.
            """
            (next_outputs, decoder_state, next_inputs,
             decoder_finished) = decoder.step(time, inputs, state)
            next_finished = tf.logical_or(decoder_finished, finished)
            if maximum_iterations is not None:
                next_finished = tf.logical_or(
                    next_finished, time + 1 >= maximum_iterations)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(
                    lambda out, zero: tf.where(finished, zero, out),
                    next_outputs,
                    zero_outputs)
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tf.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else tf.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(
                    _maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, emit)
            return (time + 1, outputs_ta, next_state, next_inputs, next_finished)

        res = tf.while_loop(
            condition,
            body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs,
                initial_finished
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]

        final_outputs = nest.map_structure(
            lambda ta: ta.stack(), final_outputs_ta)
        if not output_time_major:
            final_outputs = nest.map_structure(
                _transpose_batch_time, final_outputs)

    return final_outputs, final_state


def _transpose_batch_time(x):
    """Transpose the batch and time dimensions of a Tensor.

    Retains as much of the static shape information as possible.

    Args:
      x: A tensor of rank 2 or higher.

    Returns:
      x transposed along the first two dimensions.

    Raises:
      ValueError: if `x` is rank 1 or lower.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        raise ValueError(
            "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
            (x, x_static_shape))
    x_rank = tf.rank(x)
    x_t = tf.transpose(
        x, tf.concat(
            ([1, 0], tf.range(2, x_rank)), axis=0))
    x_t.set_shape(
        [x_static_shape[1].value, x_static_shape[0].value] + x_static_shape[2:])
    return x_t


def _create_zero_outputs(size, dtype, batch_size):
    """Create a zero outputs Tensor structure."""
    def _t(s):
        return (s if isinstance(s, tf.Tensor) else tf.constant(
            s.get_shape().as_list(),
            dtype=dtypes.int32,
            name="zero_suffix_shape"))

    def _create(s, d):
        return tf.zeros(
            tf.concat(
                ([batch_size], _t(s)), axis=0), dtype=d)

    return nest.map_structure(_create, size, dtype)
