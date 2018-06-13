"""Collection of input pipelines.

An input pipeline defines how to read and parse data. It produces a
tuple of (features, labels) that can be read by tf.learn estimators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import sys

import six
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder
from tensorflow.contrib.slim.python.slim.data import data_decoder
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.contrib.slim.python.slim.data import data_provider

from ..core.encoder import Configurable


def make_input_pipeline_from_def(def_dict, mode, **kwargs):
  """Creates an InputPipeline object from a dictionary definition.

  Args:
    def_dict: A dictionary defining the input pipeline.
      It must have "class" and "params" that correspond to the class
      name and constructor parameters of an InputPipeline, respectively.
    mode: A value in tf.contrib.learn.ModeKeys

  Returns:
    A new InputPipeline object
  """
  if "class" not in def_dict:
    raise ValueError("Input Pipeline definition must have a class property.")

  class_ = def_dict["class"]
  if not hasattr(sys.modules[__name__], class_):
    raise ValueError("Invalid Input Pipeline class: {}".format(class_))

  pipeline_class = getattr(sys.modules[__name__], class_)

  # Constructor arguments
  params = {}
  if "params" in def_dict:
    params.update(def_dict["params"])
  params.update(kwargs)

  return pipeline_class(params=params, mode=mode)


@six.add_metaclass(abc.ABCMeta)
class InputPipeline(Configurable):
  """Abstract InputPipeline class. All input pipelines must inherit from this.
  An InputPipeline defines how data is read, parsed, and separated into
  features and labels.

  Args:
    shuffle: If true, shuffle the data.
    num_epochs: Number of times to iterate through the dataset. If None,
      iterate forever.
  """

  def __init__(self, params, mode):
    Configurable.__init__(self, params, mode)

  @staticmethod
  def default_params():
    return {
        "shuffle": True,
        "num_epochs": None,
    }

  def make_data_provider(self, **kwargs):
    """Creates DataProvider instance for this input pipeline.

    Additional keyword arguments are passed to the DataProvider.
    """
    raise NotImplementedError("Not implemented.")

  @property
  def feature_keys(self):
    """Defines the features that this input pipeline provides.

    Returns a set of strings.
    """
    return set()

  @property
  def label_keys(self):
    """Defines the labels that this input pipeline provides.

    Returns a set of strings.
    """
    return set()

  @staticmethod
  def read_from_data_provider(data_provider):
    """Utility function to read all available items from a DataProvider."""
    item_values = data_provider.get(list(data_provider.list_items()))
    items_dict = dict(zip(data_provider.list_items(), item_values))
    return items_dict


class ParallelTextInputPipeline(InputPipeline):
  """An input pipeline that reads two parallel (line-by-line aligned) text
  files.

  Args:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "source_files": [],
        "target_files": [],
        "source_delimiter": " ",
        "target_delimiter": " ",
    })
    return params

  def make_data_provider(self, **kwargs):
    decoder_source = SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=self.params["source_delimiter"])

    dataset_source = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["source_files"],
        reader=tf.TextLineReader,
        decoder=decoder_source,
        num_samples=None,
        items_to_descriptions={})

    dataset_target = None
    if len(self.params["target_files"]) > 0:
      decoder_target = SplitTokensDecoder(
          tokens_feature_name="target_tokens",
          length_feature_name="target_len",
          prepend_token="SEQUENCE_START",
          append_token="SEQUENCE_END",
          delimiter=self.params["target_delimiter"])

      dataset_target = tf.contrib.slim.dataset.Dataset(
          data_sources=self.params["target_files"],
          reader=tf.TextLineReader,
          decoder=decoder_target,
          num_samples=None,
          items_to_descriptions={})

    return ParallelDataProvider(
        dataset1=dataset_source,
        dataset2=dataset_target,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])


class TFRecordInputPipeline(InputPipeline):
  """An input pipeline that reads a TFRecords containing both source and target
  sequences.

  Args:
    files: An array of file names to read from.
    source_field: The TFRecord feature field containing the source text.
    target_field: The TFRecord feature field containing the target text.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "files": [],
        "source_field": "source",
        "target_field": "target",
        "source_delimiter": " ",
        "target_delimiter": " ",
    })
    return params

  def make_data_provider(self, **kwargs):

    splitter_source = SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=self.params["source_delimiter"])

    splitter_target = SplitTokensDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=self.params["target_delimiter"])

    keys_to_features = {
        self.params["source_field"]: tf.FixedLenFeature((), tf.string),
        self.params["target_field"]: tf.FixedLenFeature((), tf.string, default_value="")
    }

    items_to_handlers = {}
    items_to_handlers["source_tokens"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["source_field"]],
        func=lambda dict: splitter_source.decode(dict[self.params["source_field"]],
                ["source_tokens"])[0])
    items_to_handlers["source_len"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["source_field"]],
        func=lambda dict: splitter_source.decode(dict[self.params["source_field"]],
        ["source_len"])[0])
    items_to_handlers["target_tokens"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["target_field"]],
        func=lambda dict: splitter_target.decode(dict[self.params["target_field"]],
                ["target_tokens"])[0])
    items_to_handlers["target_len"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["target_field"]],
        func=lambda dict: splitter_target.decode(dict[self.params["target_field"]],
                ["target_len"])[0]
    )

    decoder = tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["files"],
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})

    return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])


class ImageCaptioningInputPipeline(InputPipeline):
  """An input pipeline that reads a TFRecords containing both source and target
  sequences.

  Args:
    files: An array of file names to read from.
    source_field: The TFRecord feature field containing the source text.
    target_field: The TFRecord feature field containing the target text.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "files": [],
        "image_field": "image/data",
        "image_format": "jpg",
        "caption_ids_field": "image/caption_ids",
        "caption_tokens_field": "image/caption",
    })
    return params

  def make_data_provider(self, **kwargs):

    context_keys_to_features = {
        self.params["image_field"]:
        tf.FixedLenFeature([], dtype=tf.string),
        "image/format":
        tf.FixedLenFeature([], dtype=tf.string, default_value=self.params["image_format"]),
    }

    sequence_keys_to_features = {
        self.params["caption_ids_field"]: tf.FixedLenSequenceFeature([], dtype=tf.int64),
        self.params["caption_tokens_field"]: tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    items_to_handlers = {
        "image":
        tfexample_decoder.Image(
            image_key=self.params["image_field"], format_key="image/format", channels=3),
        "target_ids":
        tfexample_decoder.Tensor(self.params["caption_ids_field"]),
        "target_tokens":
        tfexample_decoder.Tensor(self.params["caption_tokens_field"]),
        "target_len":
        tfexample_decoder.ItemHandlerCallback(
            keys=[self.params["caption_tokens_field"]],
            func=lambda x: tf.size(x[self.params["caption_tokens_field"]]))
    }

    decoder = TFSEquenceExampleDecoder(context_keys_to_features, sequence_keys_to_features,
                                       items_to_handlers)

    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["files"],
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})

    return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["image"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_ids", "target_len"])


class SplitTokensDecoder(data_decoder.DataDecoder):
  """A DataProvider that splits a string tensor into individual tokens and
  returns the tokens and the length. Optionally prepends or appends special
  tokens.

  Args:
    delimiter: Delimiter to split on. Must be a single character.
    tokens_feature_name: A descriptive feature name for the token values
    length_feature_name: A descriptive feature name for the length value
  """

  def __init__(self,
               delimiter=" ",
               tokens_feature_name="tokens",
               length_feature_name="length",
               prepend_token=None,
               append_token=None):
    self.delimiter = delimiter
    self.tokens_feature_name = tokens_feature_name
    self.length_feature_name = length_feature_name
    self.prepend_token = prepend_token
    self.append_token = append_token

  def decode(self, data, items):
    decoded_items = {}

    # Split tokens
    tokens = tf.string_split([data], delimiter=self.delimiter).values

    # Optionally prepend a special token
    if self.prepend_token is not None:
      tokens = tf.concat([[self.prepend_token], tokens], 0)

    # Optionally append a special token
    if self.append_token is not None:
      tokens = tf.concat([tokens, [self.append_token]], 0)

    decoded_items[self.length_feature_name] = tf.size(tokens)
    decoded_items[self.tokens_feature_name] = tokens
    return [decoded_items[_] for _ in items]

  def list_items(self):
    return [self.tokens_feature_name, self.length_feature_name]


def make_parallel_data_provider(data_sources_source,
                                data_sources_target,
                                reader=tf.TextLineReader,
                                num_samples=None,
                                source_delimiter=" ",
                                target_delimiter=" ",
                                **kwargs):
  """Creates a DataProvider that reads parallel text data.

  Args:
    data_sources_source: A list of data sources for the source text files.
    data_sources_target: A list of data sources for the target text files.
      Can be None for inference mode.
    num_samples: Optional, number of records in the dataset
    delimiter: Split tokens in the data on this delimiter. Defaults to space.
    kwargs: Additional arguments (shuffle, num_epochs, etc) that are passed
      to the data provider

  Returns:
    A DataProvider instance
  """

  decoder_source = SplitTokensDecoder(
      tokens_feature_name="source_tokens",
      length_feature_name="source_len",
      append_token="SEQUENCE_END",
      delimiter=source_delimiter)

  dataset_source = tf.contrib.slim.dataset.Dataset(
      data_sources=data_sources_source,
      reader=reader,
      decoder=decoder_source,
      num_samples=num_samples,
      items_to_descriptions={})

  dataset_target = None
  if data_sources_target is not None:
    decoder_target = SplitTokensDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=target_delimiter)

    dataset_target = tf.contrib.slim.dataset.Dataset(
        data_sources=data_sources_target,
        reader=reader,
        decoder=decoder_target,
        num_samples=num_samples,
        items_to_descriptions={})

  return ParallelDataProvider(dataset1=dataset_source, dataset2=dataset_target, **kwargs)


class ParallelDataProvider(data_provider.DataProvider):
  """Creates a ParallelDataProvider. This data provider reads two datasets in
  parallel, keeping them aligned.

  Args:
    dataset1: The first dataset. An instance of the Dataset class.
    dataset2: The second dataset. An instance of the Dataset class.
      Can be None. If None, only `dataset1` is read.
    num_readers: The number of parallel readers to use.
    shuffle: Whether to shuffle the data sources and common queue when
      reading.
    num_epochs: The number of times each data source is read. If left as None,
      the data will be cycled through indefinitely.
    common_queue_capacity: The capacity of the common queue.
    common_queue_min: The minimum number of elements in the common queue after
      a dequeue.
    seed: The seed to use if shuffling.
  """

  def __init__(self,
               dataset1,
               dataset2,
               shuffle=True,
               num_epochs=None,
               common_queue_capacity=4096,
               common_queue_min=1024,
               seed=None):

    if seed is None:
      seed = np.random.randint(10e8)

    _, data_source = parallel_reader.parallel_read(
        dataset1.data_sources,
        reader_class=dataset1.reader,
        num_epochs=num_epochs,
        num_readers=1,
        shuffle=False,
        capacity=common_queue_capacity,
        min_after_dequeue=common_queue_min,
        seed=seed)

    data_target = ""
    if dataset2 is not None:
      _, data_target = parallel_reader.parallel_read(
          dataset2.data_sources,
          reader_class=dataset2.reader,
          num_epochs=num_epochs,
          num_readers=1,
          shuffle=False,
          capacity=common_queue_capacity,
          min_after_dequeue=common_queue_min,
          seed=seed)

    # Optionally shuffle the data
    if shuffle:
      shuffle_queue = tf.RandomShuffleQueue(
          capacity=common_queue_capacity,
          min_after_dequeue=common_queue_min,
          dtypes=[tf.string, tf.string],
          seed=seed)
      enqueue_ops = []
      enqueue_ops.append(shuffle_queue.enqueue([data_source, data_target]))
      tf.train.add_queue_runner(tf.train.QueueRunner(shuffle_queue, enqueue_ops))
      data_source, data_target = shuffle_queue.dequeue()

    # Decode source items
    items = dataset1.decoder.list_items()
    tensors = dataset1.decoder.decode(data_source, items)

    if dataset2 is not None:
      # Decode target items
      items2 = dataset2.decoder.list_items()
      tensors2 = dataset2.decoder.decode(data_target, items2)

      # Merge items and results
      items = items + items2
      tensors = tensors + tensors2

    super(ParallelDataProvider, self).__init__(
        items_to_tensors=dict(zip(items, tensors)), num_samples=dataset1.num_samples)


class TFSEquenceExampleDecoder(data_decoder.DataDecoder):
  """A decoder for TensorFlow Examples.

  Decoding Example proto buffers is comprised of two stages: (1) Example
  parsing and (2) tensor manipulation. In the first stage, the
  tf.parse_example function is called with a list of FixedLenFeatures
  and SparseLenFeatures. These instances tell TF how to parse the
  example. The output of this stage is a set of tensors. In the second
  stage, the resulting tensors are manipulated to provide the requested
  'item' tensors. To perform this decoding operation, an ExampleDecoder
  is given a list of ItemHandlers. Each ItemHandler indicates the set of
  features for stage 1 and contains the instructions for post_processing
  its tensors for stage 2.
  """

  def __init__(self, context_keys_to_features, sequence_keys_to_features, items_to_handlers):
    """Constructs the decoder.

    Args:
      keys_to_features: a dictionary from TF-Example keys to either
        tf.VarLenFeature or tf.FixedLenFeature instances. See tensorflow's
        parsing_ops.py.
      items_to_handlers: a dictionary from items (strings) to ItemHandler
        instances. Note that the ItemHandler's are provided the keys that they
        use to return the final item Tensors.
    """
    self._context_keys_to_features = context_keys_to_features
    self._sequence_keys_to_features = sequence_keys_to_features
    self._items_to_handlers = items_to_handlers

  def list_items(self):
    """See base class."""
    return list(self._items_to_handlers.keys())

  def decode(self, serialized_example, items=None):
    """Decodes the given serialized TF-example.

    Args:
      serialized_example: a serialized TF-example tensor.
      items: the list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.
    Returns:
      the decoded items, a list of tensor.
    """
    context, sequence = tf.parse_single_sequence_example(
        serialized_example, self._context_keys_to_features, self._sequence_keys_to_features)

    # Merge context and sequence features
    example = {}
    example.update(context)
    example.update(sequence)

    all_features = {}
    all_features.update(self._context_keys_to_features)
    all_features.update(self._sequence_keys_to_features)

    # Reshape non-sparse elements just once:
    for k, value in all_features.items():
      if isinstance(value, tf.FixedLenFeature):
        example[k] = tf.reshape(example[k], value.shape)

    if not items:
      items = self._items_to_handlers.keys()

    outputs = []
    for item in items:
      handler = self._items_to_handlers[item]
      keys_to_tensors = {key: example[key] for key in handler.keys}
      outputs.append(handler.tensors_to_item(keys_to_tensors))
    return outputs


def create_input_fn(pipeline,
                    batch_size,
                    bucket_boundaries=None,
                    allow_smaller_final_batch=False,
                    scope=None):
  """Creates an input function that can be used with tf.learn estimators. Note
  that you must pass "factory funcitons" for both the data provider and
  featurizer to ensure that everything will be created in  the same graph.

  Args:
    pipeline: An instance of `seq2seq.data.InputPipeline`.
    batch_size: Create batches of this size. A queue to hold a
      reasonable number of batches in memory is created.
    bucket_boundaries: int list, increasing non-negative numbers.
      If None, no bucket is performed.

  Returns:
    An input function that returns `(feature_batch, labels_batch)`
    tuples when called.
  """

  def input_fn():
    """Creates features and labels."""

    with tf.variable_scope(scope or "input_fn"):
      data_provider = pipeline.make_data_provider()
      features_and_labels = pipeline.read_from_data_provider(data_provider)

      if bucket_boundaries:
        _, batch = tf.contrib.training.bucket_by_sequence_length(
            input_length=features_and_labels["source_len"],
            bucket_boundaries=bucket_boundaries,
            tensors=features_and_labels,
            batch_size=batch_size,
            keep_input=features_and_labels["source_len"] >= 1,
            dynamic_pad=True,
            capacity=5000 + 16 * batch_size,
            allow_smaller_final_batch=allow_smaller_final_batch,
            name="bucket_queue")
      else:
        batch = tf.train.batch(
            tensors=features_and_labels,
            enqueue_many=False,
            batch_size=batch_size,
            dynamic_pad=True,
            capacity=5000 + 16 * batch_size,
            allow_smaller_final_batch=allow_smaller_final_batch,
            name="batch_queue")

      # Separate features and labels
      features_batch = {k: batch[k] for k in pipeline.feature_keys}
      if set(batch.keys()).intersection(pipeline.label_keys):
        labels_batch = {k: batch[k] for k in pipeline.label_keys}
      else:
        labels_batch = None

      return features_batch, labels_batch

  return input_fn
