# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os
import numpy as np
import tensorflow as tf


class Dataset(object):
    """A simple class for handling data sets."""
    __metaclass__ = ABCMeta

    def __init__(self, name, decoder, data_dir=None, items_to_descriptions=None, **kwargs):
        """Initialize dataset using a subset and the path to the data."""
        self.name = name
        self.decoer = decoder
        self.data_dir = data_dir
        self.items_to_descriptions = items_to_descriptions
        self.__dict__.update(kwargs)

    @abstractmethod
    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass

    @abstractmethod
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        pass

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
            return np.array(sorted(data_files))
        except Exception:
            raise ValueError('No files found for dataset %s at %s' % (self.name, self.data_dir))

    @property
    def reader_class(self):
        """Return a reader for a single entry from the data set.
        Returns:
            Reader object that reads the data set.
        """
        return tf.TFRecordReader()

    @property
    def decoder(self):
        """Return a decoder for a single entry from the data set.
        Returns:
            Decoder object that decodes the data samples.
        """
        return self.decoder
