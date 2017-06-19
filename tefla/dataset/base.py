# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import os
import math
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Dataset(object):
    """A simple class for handling data sets,

    Args:
        name: a string, Name of the class instance
        decoder: object instance, tfrecords object decoding and image encoding and decoding
        data_dir: a string, path to the data folder
        num_classes: num of classes of the dataset
        num_examples_per_epoch: total number of examples per epoch
        items_to_description: a string descriving the items of the dataset

    """

    def __init__(self, name, decoder, data_dir=None, num_classes=10, num_examples_per_epoch=1, batch_size=1, items_to_descriptions=None, **kwargs):
        self.name = name
        self._decoder = decoder
        self.data_dir = data_dir
        self._num_classes = num_classes
        self._num_examples_per_epoch = num_examples_per_epoch
        self._batch_size = batch_size
        self.items_to_descriptions = items_to_descriptions
        self.__dict__.update(kwargs)

    @property
    def num_classes(self):
        """Returns the number of classes in the data set."""
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        """Set the number of classes in the data set."""
        self._num_classes = value

    @property
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        return self._num_examples_per_epoch

    @property
    def n_iters_per_epoch(self):
        return int(math.ceil(self._num_examples_per_epoch / float(self._batch_size)))

    @num_examples_per_epoch.setter
    def num_examples_per_epoch(self, value):
        """Set the number of examples in the data subset."""
        self._num_examples_per_epoch = value

    def data_files(self):
        """Returns a python list of all (sharded) data subset files.

        Returns:
            python list of all (sharded) data set files.

        Raises:
            ValueError: if there are not data_files matching the subset.
        """
        try:
            data_files = [f for f in os.listdir(self.data_dir)]
            data_files = [os.path.join(self.data_dir, f) for f in data_files]
            # return np.array(sorted(data_files))
            return data_files
        except Exception:
            raise ValueError('No files found for dataset %s at %s' %
                             (self.name, self.data_dir))

    @property
    def reader_class(self):
        """Return a reader for a single entry from the data set.

        Returns:
            Reader object that reads the data set.
        """
        return tf.TFRecordReader

    @property
    def decoder(self):
        """Return a decoder for a single entry from the data set.

        Returns:
            Decoder object that decodes the data samples.
        """
        return self._decoder

    @decoder.setter
    def decoder(self, decoder_object):
        """Set a decoder object for a single entry from the data set.
        """
        self._decoder = decoder_object
