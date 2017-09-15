from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import random
import tensorflow as tf


class TextDecoder(object):
    """A Decoder class to decode examples
      The dictionary features_keys for an image dataset can be:

     example_reading_spec{
      features_keys = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature(
            [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
      }}

      and for a simple algorithmic dataset with variable-length data it is:

     example_reading_spec{
      features_keys = {
        'inputs': tf.VarLenFeature(tf.int64),
        'targets': tf.VarLenFeature(tf.int64),
      }}

    Args:
        feature_keys: a dict, with features name and data types
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._feature_keys = dataset.example_reading_spec()[0]
        self._feature_names = self._feature_keys.keys()

    @property
    def feature_names(self):
        return self._feature_names

    def examples_reader(self, data_sources,
                        training,
                        capacity=32,
                        data_items_to_decoders=None,
                        data_items_to_decode=None):
        """Reads Examples from data_sources and decodes to Tensors.

        Args:
            data_sources: a list or tuple of sources from which the data will be read,
                for example [/path/to/train@128, /path/to/train2*, /tmp/.../train3*]
            training: a Boolean, whether to read for training or evaluation.
            capacity: integer, buffer capacity; set to 2 * max_batch_size or more.
            data_items_to_decoders: a dictionary mapping data items (that will be
                in the returned result) to decoders that will decode them using features
                defined in data_fields_to_features; see above for examples. By default
               (if this is None), we grab the tensor from every feature.
            data_items_to_decode: a subset of data items that will be decoded;
                by default (if this is None), we decode all items.

        Returns:
          A tf.contrib.data.Dataset of dict<feature name, Tensor>
        """

        def decode_record(record):
            """Serialized Example to dict of <feature name, Tensor>."""
            example_serialized = record
            item_decoders = data_items_to_decoders
            if item_decoders is None:
                item_decoders = {
                    field: Tensor(field)
                    for field in self._feature_keys
                }

            decode_items = data_items_to_decode
            if decode_items is None:
                decode_items = list(item_decoders.keys())

            decoded = self.decode(example_serialized,
                                  item_decoders, items=decode_items)
            return dict(zip(decode_items, decoded))
            # return decoded

        with tf.name_scope("examples_in"):
            data_files = self.dataset.get_data_files(data_sources)
            datasetreader = tf.contrib.data.TFRecordDataset(data_files)
            num_threads = min(4 if training else 1, len(data_files))
            datasetreader = datasetreader.map(
                decode_record, num_threads=num_threads)
            if training:
                datasetreader = datasetreader.shuffle(capacity)
            # Loop inifinitely if training, just once otherwise
            datasetreader = datasetreader.repeat(None if training else 1)
            return datasetreader

    def decode(self, serialized_example, item_decoders, items=None):
        """Decodes the given serialized TF-example.

        Args:
          serialized_example: a serialized TF-example tensor.
          items: the list of items to decode. These must be a subset of the item
            keys in self._items_to_handlers. If `items` is left as None, then all
            of the items in self._items_to_handlers are decoded.

        Returns:
          the decoded items, a list of tensor.
        """
        example = tf.parse_single_example(serialized_example,
                                          self._feature_keys)
        for k in sorted(self._feature_keys):
            v = self._feature_keys[k]
            if isinstance(v, tf.FixedLenFeature):
                example[k] = tf.reshape(example[k], v.shape)

        if not items:
            items = item_decoders.keys()

        outputs = []
        for item in items:
            handler = item_decoders[item]
            keys_to_tensors = {key: example[key] for key in handler.keys}
            outputs.append(handler.tensors_to_item(keys_to_tensors))
        return outputs


@six.add_metaclass(abc.ABCMeta)
class ItemHandler():
    """Specifies the item-to-Features mapping for tf.parse_example.
    An ItemHandler both specifies a list of Features used for parsing an Example
    proto as well as a function that post-processes the results of Example
    parsing.
    """

    def __init__(self, keys):
        """Constructs the handler with the name of the tf.Feature keys to use.
        See third_party/tensorflow/core/example/feature.proto
        Args:
          keys: the name of the TensorFlow Example Feature.
        """
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        self._keys = keys

    @property
    def keys(self):
        return self._keys

    @abc.abstractmethod
    def tensors_to_item(self, keys_to_tensors):
        """Maps the given dictionary of tensors to the requested item.
        Args:
          keys_to_tensors: a mapping of TF-Example keys to parsed tensors.
        Returns:
          the final tensor representing the item being handled.
        """
        pass


class Tensor(ItemHandler):
    """An ItemHandler that returns a parsed Tensor."""

    def __init__(self, tensor_key, shape_keys=None, shape=None, default_value=0):
        """Initializes the Tensor handler.
        Tensors are, by default, returned without any reshaping. However, there are
        two mechanisms which allow reshaping to occur at load time. If `shape_keys`
        is provided, both the `Tensor` corresponding to `tensor_key` and
        `shape_keys` is loaded and the former `Tensor` is reshaped with the values
        of the latter. Alternatively, if a fixed `shape` is provided, the `Tensor`
        corresponding to `tensor_key` is loaded and reshape appropriately.
        If neither `shape_keys` nor `shape` are provided, the `Tensor` will be
        returned without any reshaping.

        Args:
          tensor_key: the name of the `TFExample` feature to read the tensor from.
          shape_keys: Optional name or list of names of the TF-Example feature in
            which the tensor shape is stored. If a list, then each corresponds to
            one dimension of the shape.
          shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
            reshaped accordingly.
          default_value: The value used when the `tensor_key` is not found in a
            particular `TFExample`.

        Raises:
          ValueError: if both `shape_keys` and `shape` are specified.
        """
        if shape_keys and shape is not None:
            raise ValueError(
                'Cannot specify both shape_keys and shape parameters.')
        if shape_keys and not isinstance(shape_keys, list):
            shape_keys = [shape_keys]
        self._tensor_key = tensor_key
        self._shape_keys = shape_keys
        self._shape = shape
        self._default_value = default_value
        keys = [tensor_key]
        if shape_keys:
            keys.extend(shape_keys)
        super(Tensor, self).__init__(keys)

    def tensors_to_item(self, keys_to_tensors):
        tensor = keys_to_tensors[self._tensor_key]
        shape = self._shape
        if self._shape_keys:
            shape_dims = []
            for k in self._shape_keys:
                shape_dim = keys_to_tensors[k]
                if isinstance(shape_dim, tf.SparseTensor):
                    shape_dim = tf.sparse_tensor_to_dense(shape_dim)
                shape_dims.append(shape_dim)
            shape = tf.reshape(tf.stack(shape_dims), [-1])
        if isinstance(tensor, tf.SparseTensor):
            if shape is not None:
                tensor = tf.sparse_reshape(tensor, shape)
            tensor = tf.sparse_tensor_to_dense(
                tensor, self._default_value)
        else:
            if shape is not None:
                tensor = tf.reshape(tensor, shape)

        return tensor
