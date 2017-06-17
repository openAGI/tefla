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

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

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


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" %
              (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + \
                sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with tf.gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not tf.gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with tf.gfile.GFile(data_path, mode="rb") as data_file:
            with tf.gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                      tokenizer, normalize_digits)
                    tokens_file.write(
                        " ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                 to_vocabulary_size, tokenizer=None):
    """Preapre all necessary files that are required for the training.

      Args:
        data_dir: directory in which the data sets will be stored.
        from_train_path: path to the file that includes "from" training samples.
        to_train_path: path to the file that includes "to" training samples.
        from_dev_path: path to the file that includes "from" dev samples.
        to_dev_path: path to the file that includes "to" dev samples.
        from_vocabulary_size: size of the "from language" vocabulary to create and use.
        to_vocabulary_size: size of the "to language" vocabulary to create and use.
        tokenizer: a function to use to tokenize each data sentence;
          if None, basic_tokenizer will be used.

      Returns:
        A tuple of 6 elements:
          (1) path to the token-ids for "from language" training data-set,
          (2) path to the token-ids for "to language" training data-set,
          (3) path to the token-ids for "from language" development data-set,
          (4) path to the token-ids for "to language" development data-set,
          (5) path to the "from language" vocabulary file,
          (6) path to the "to language" vocabulary file.
      """
    # Create vocabularies of the appropriate sizes.
    to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
    from_vocab_path = os.path.join(
        data_dir, "vocab%d.from" % from_vocabulary_size)
    create_vocabulary(to_vocab_path, to_train_path,
                      to_vocabulary_size, tokenizer)
    create_vocabulary(from_vocab_path, from_train_path,
                      from_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
    from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
    data_to_token_ids(to_train_path, to_train_ids_path,
                      to_vocab_path, tokenizer)
    data_to_token_ids(from_train_path, from_train_ids_path,
                      from_vocab_path, tokenizer)

    # Create token ids for the development data.
    to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
    from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
    data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
    data_to_token_ids(from_dev_path, from_dev_ids_path,
                      from_vocab_path, tokenizer)

    return (from_train_ids_path, to_train_ids_path,
            from_dev_ids_path, to_dev_ids_path,
            from_vocab_path, to_vocab_path)
