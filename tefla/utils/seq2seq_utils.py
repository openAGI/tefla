"credit: github.com/google/seg2seq"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
import tensorflow as tf
from tensorflow.python.layers import base as layers_base
from tensorflow.python.util import nest

distributions = tf.contrib.distributions


__all__ = [
    "Helper",
    "TrainingHelper",
    "GreedyEmbeddingHelper",
    "CustomHelper",
    "ScheduledEmbeddingTrainingHelper",
    "ScheduledOutputTrainingHelper",
]


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
        tf.TensorShape([
            x_static_shape[1].value, x_static_shape[0].value
        ]).concatenate(x_static_shape[2:]))
    return x_t


def _unstack_ta(inp):
    return tf.TensorArray(
        dtype=inp.dtype, size=tf.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)


@six.add_metaclass(abc.ABCMeta)
class Helper(object):
    """Helper interface.  Helper instances are used by SamplingDecoder."""

    @abc.abstractproperty
    def batch_size(self):
        """Returns a scalar int32 tensor."""
        raise NotImplementedError("batch_size has not been implemented")

    @abc.abstractmethod
    def initialize(self, name=None):
        """Returns `(initial_finished, initial_inputs)`."""
        pass

    @abc.abstractmethod
    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        pass

    @abc.abstractmethod
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        pass


class CustomHelper(Helper):
    """Base abstract class that allows the user to customize sampling."""

    def __init__(self, initialize_fn, sample_fn, next_inputs_fn):
        """Initializer.

        Args:
          initialize_fn: callable that returns `(finished, next_inputs)`
            for the first iteration.
          sample_fn: callable that takes `(time, outputs, state)`
            and emits tensor `sample_ids`.
          next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
            and emits `(finished, next_inputs, next_state)`.
        """
        self._initialize_fn = initialize_fn
        self._sample_fn = sample_fn
        self._next_inputs_fn = next_inputs_fn
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    def initialize(self, name=None):
        with tf.name_scope(name, "%sInitialize" % type(self).__name__):
            (finished, next_inputs) = self._initialize_fn()
            if self._batch_size is None:
                self._batch_size = tf.size(finished)
        return (finished, next_inputs)

    def sample(self, time, outputs, state, name=None):
        with tf.name_scope(
                name, "%sSample" % type(self).__name__, (time, outputs, state)):
            return self._sample_fn(time=time, outputs=outputs, state=state)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(
                name, "%sNextInputs" % type(self).__name__, (time, outputs, state)):
            return self._next_inputs_fn(
                time=time, outputs=outputs, state=state, sample_ids=sample_ids)


class TrainingHelper(Helper):
    """A helper for use during training.  Only reads inputs.

    Returned sample_ids are the argmax of the RNN output logits.
    """

    def __init__(self, inputs, sequence_length, time_major=False, name=None):
        """Initializer.

        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          name: Name scope for any created operations.

        Raises:
          ValueError: if `sequence_length` is not a 1D tensor.
        """
        with tf.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
            inputs = tf.convert_to_tensor(inputs, name="inputs")
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)

            self._input_tas = nest.map_structure(_unstack_ta, inputs)
            self._sequence_length = tf.convert_to_tensor(
                sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be a vector, but received shape: %s" %
                    self._sequence_length.get_shape())

            self._zero_inputs = nest.map_structure(
                lambda inp: tf.zeros_like(inp[0, :]), inputs)

            self._batch_size = tf.size(sequence_length)

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        with tf.name_scope(name, "TrainingHelperInitialize"):
            finished = tf.equal(0, self._sequence_length)
            all_finished = tf.reduce_all(finished)
            next_inputs = tf.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
            return (finished, next_inputs)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "TrainingHelperSample", [time, outputs]):
            sample_ids = tf.cast(
                tf.argmax(outputs, axis=-1), tf.int32)
            return sample_ids

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """next_inputs_fn for TrainingHelper."""
        with tf.name_scope(name, "TrainingHelperNextInputs",
                           [time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = tf.reduce_all(finished)

            def read_from_ta(inp):
                return inp.read(next_time)
            next_inputs = tf.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(read_from_ta, self._input_tas))
            return (finished, next_inputs, state)


class ScheduledEmbeddingTrainingHelper(TrainingHelper):
    """A training helper that adds scheduled sampling.

    Returns -1s for sample_ids where no sampling took place; valid sample id
    values elsewhere.
    """

    def __init__(self, inputs, sequence_length, embedding, sampling_probability,
                 time_major=False, seed=None, scheduling_seed=None, name=None):
        """Initializer.

        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`.
          sampling_probability: A 0D `float32` tensor: the probability of sampling
            categorically from the output ids instead of reading directly from the
            inputs.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          scheduling_seed: The schedule decision rule sampling seed.
          name: Name scope for any created operations.

        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        with tf.name_scope(name, "ScheduledEmbeddingSamplingWrapper",
                           [embedding, sampling_probability]):
            if callable(embedding):
                self._embedding_fn = embedding
            else:
                self._embedding_fn = (
                    lambda ids: tf.nn.embedding_lookup(embedding, ids))
            self._sampling_probability = tf.convert_to_tensor(
                sampling_probability, name="sampling_probability")
            if self._sampling_probability.get_shape().ndims not in (0, 1):
                raise ValueError(
                    "sampling_probability must be either a scalar or a vector. "
                    "saw shape: %s" % (self._sampling_probability.get_shape()))
            self._seed = seed
            self._scheduling_seed = scheduling_seed
            super(ScheduledEmbeddingTrainingHelper, self).__init__(
                inputs=inputs,
                sequence_length=sequence_length,
                time_major=time_major,
                name=name)

    def initialize(self, name=None):
        return super(ScheduledEmbeddingTrainingHelper, self).initialize(name=name)

    def sample(self, time, outputs, state, name=None):
        with tf.name_scope(name, "ScheduledEmbeddingTrainingHelperSample",
                           [time, outputs, state]):
            # Return -1s where we did not sample, and sample_ids elsewhere
            select_sample_noise = tf.random_uniform(
                [self.batch_size], seed=self._scheduling_seed)
            select_sample = (self._sampling_probability > select_sample_noise)
            sample_id_sampler = distributions.Categorical(logits=outputs)
            return tf.where(
                select_sample,
                sample_id_sampler.sample(seed=self._seed),
                tf.tile([-1], [self.batch_size]))

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name, "ScheduledEmbeddingTrainingHelperSample",
                           [time, outputs, state, sample_ids]):
            (finished, base_next_inputs, state) = (
                super(ScheduledEmbeddingTrainingHelper, self).next_inputs(
                    time=time,
                    outputs=outputs,
                    state=state,
                    sample_ids=sample_ids,
                    name=name))

            def maybe_sample():
                """Perform scheduled sampling."""
                where_sampling = tf.cast(
                    tf.where(sample_ids > -1), tf.int32)
                where_not_sampling = tf.cast(
                    tf.where(sample_ids <= -1), tf.int32)
                where_sampling_flat = tf.reshape(where_sampling, [-1])
                where_not_sampling_flat = tf.reshape(
                    where_not_sampling, [-1])
                sample_ids_sampling = tf.gather(
                    sample_ids, where_sampling_flat)
                inputs_not_sampling = tf.gather(
                    base_next_inputs, where_not_sampling_flat)
                sampled_next_inputs = self._embedding_fn(sample_ids_sampling)
                base_shape = tf.shape(base_next_inputs)
                return (tf.scatter_nd(indices=where_sampling,
                                      updates=sampled_next_inputs,
                                      shape=base_shape)
                        + tf.scatter_nd(indices=where_not_sampling,
                                        updates=inputs_not_sampling,
                                        shape=base_shape))

            all_finished = tf.reduce_all(finished)
            next_inputs = tf.cond(
                all_finished, lambda: base_next_inputs, maybe_sample)
            return (finished, next_inputs, state)


class ScheduledOutputTrainingHelper(TrainingHelper):
    """A training helper that adds scheduled sampling directly to outputs.

    Returns False for sample_ids where no sampling took place; True elsewhere.
    """

    def __init__(self, inputs, sequence_length, sampling_probability,
                 time_major=False, seed=None, next_input_layer=None,
                 auxiliary_inputs=None, name=None):
        """Initializer.

        Args:
          inputs: A (structure) of input tensors.
          sequence_length: An int32 vector tensor.
          sampling_probability: A 0D `float32` tensor: the probability of sampling
            from the outputs instead of reading directly from the inputs.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          next_input_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Optional layer to apply to the RNN output to create
            the next input.
          auxiliary_inputs: An optional (structure of) auxiliary input tensors with
            a shape that matches `inputs` in all but (potentially) the final
            dimension. These tensors will be concatenated to the sampled output or
            the `inputs` when not sampling for use as the next input.
          name: Name scope for any created operations.

        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        with tf.name_scope(name, "ScheduledOutputTrainingHelper",
                           [inputs, auxiliary_inputs, sampling_probability]):
            self._sampling_probability = tf.convert_to_tensor(
                sampling_probability, name="sampling_probability")
            if self._sampling_probability.get_shape().ndims not in (0, 1):
                raise ValueError(
                    "sampling_probability must be either a scalar or a vector. "
                    "saw shape: %s" % (self._sampling_probability.get_shape()))

            if auxiliary_inputs is None:
                maybe_concatenated_inputs = inputs
            else:
                inputs = tf.convert_to_tensor(inputs, name="inputs")
                auxiliary_inputs = tf.convert_to_tensor(
                    auxiliary_inputs, name="auxiliary_inputs")
                maybe_concatenated_inputs = nest.map_structure(
                    lambda x, y: tf.concat((x, y), -1),
                    inputs, auxiliary_inputs)
                if not time_major:
                    auxiliary_inputs = nest.map_structure(
                        _transpose_batch_time, auxiliary_inputs)

            self._auxiliary_input_tas = (
                nest.map_structure(_unstack_ta, auxiliary_inputs)
                if auxiliary_inputs is not None else None)

            self._seed = seed

            if (next_input_layer is not None and not isinstance(next_input_layer,
                                                                layers_base._Layer)):  # pylint: disable=protected-access
                raise TypeError("next_input_layer must be a Layer, received: %s" %
                                type(next_input_layer))
            self._next_input_layer = next_input_layer

            super(ScheduledOutputTrainingHelper, self).__init__(
                inputs=maybe_concatenated_inputs,
                sequence_length=sequence_length,
                time_major=time_major,
                name=name)

    def initialize(self, name=None):
        return super(ScheduledOutputTrainingHelper, self).initialize(name=name)

    def sample(self, time, outputs, state, name=None):
        with tf.name_scope(name, "ScheduledOutputTrainingHelperSample",
                           [time, outputs, state]):
            sampler = distributions.Bernoulli(probs=self._sampling_probability)
            return tf.cast(
                sampler.sample(sample_shape=self.batch_size, seed=self._seed),
                tf.bool)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name, "ScheduledOutputTrainingHelperNextInputs",
                           [time, outputs, state, sample_ids]):
            (finished, base_next_inputs, state) = (
                super(ScheduledOutputTrainingHelper, self).next_inputs(
                    time=time,
                    outputs=outputs,
                    state=state,
                    sample_ids=sample_ids,
                    name=name))

            def maybe_sample():
                """Perform scheduled sampling."""

                def maybe_concatenate_auxiliary_inputs(outputs_, indices=None):
                    """Concatenate outputs with auxiliary inputs, if they exist."""
                    if self._auxiliary_input_tas is None:
                        return outputs_

                    next_time = time + 1
                    auxiliary_inputs = nest.map_structure(
                        lambda ta: ta.read(next_time), self._auxiliary_input_tas)
                    if indices is not None:
                        auxiliary_inputs = tf.gather_nd(
                            auxiliary_inputs, indices)
                    return nest.map_structure(
                        lambda x, y: tf.concat((x, y), -1),
                        outputs_, auxiliary_inputs)

                if self._next_input_layer is None:
                    return tf.where(
                        sample_ids, maybe_concatenate_auxiliary_inputs(outputs),
                        base_next_inputs)

                where_sampling = tf.cast(
                    tf.where(sample_ids), tf.int32)
                where_not_sampling = tf.cast(
                    tf.where(tf.logical_not(sample_ids)), tf.int32)
                outputs_sampling = tf.gather_nd(outputs, where_sampling)
                inputs_not_sampling = tf.gather_nd(base_next_inputs,
                                                   where_not_sampling)
                sampled_next_inputs = maybe_concatenate_auxiliary_inputs(
                    self._next_input_layer(outputs_sampling), where_sampling)

                base_shape = tf.shape(base_next_inputs)
                return (tf.scatter_nd(indices=where_sampling,
                                      updates=sampled_next_inputs,
                                      shape=base_shape)
                        + tf.scatter_nd(indices=where_not_sampling,
                                        updates=inputs_not_sampling,
                                        shape=base_shape))

            all_finished = tf.reduce_all(finished)
            next_inputs = tf.cond(
                all_finished, lambda: base_next_inputs, maybe_sample)
            return (finished, next_inputs, state)


class GreedyEmbeddingHelper(Helper):
    """A helper for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token):
        """Initializer.

        Args:
          embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`.
          start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
          end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
          ValueError: if `sequence_length` is not a 1D tensor.
        """
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))

        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = tf.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embedding_fn(self._start_tokens)

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        sample_ids = tf.cast(
            tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self._end_token)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)
