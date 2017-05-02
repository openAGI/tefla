"""A dataset based on files in a training and validation directory, and corresponding label files."""
from __future__ import division, print_function, absolute_import

import logging

import numpy as np

from . import data_load_ops as data

logger = logging.getLogger('tefla')


class DataSet(object):

    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir
        training_images_dir = "%s/training_%d" % (data_dir, img_size)
        training_labels_file = "%s/training_labels.csv" % data_dir

        validation_images_dir = "%s/validation_%d" % (data_dir, img_size)
        validation_labels_file = "%s/validation_labels.csv" % data_dir

        self._training_files = data.get_image_files(training_images_dir)
        names = data.get_names(self._training_files)
        self._training_labels = data.get_labels(
            names, label_file=training_labels_file).astype(np.int32)

        self._validation_files = data.get_image_files(validation_images_dir)
        names = data.get_names(self._validation_files)
        self._validation_labels = data.get_labels(
            names, label_file=validation_labels_file).astype(np.int32)

    @property
    def training_X(self):
        return self._training_files

    @property
    def training_y(self):
        return self._training_labels

    @property
    def validation_X(self):
        return self._validation_files

    @property
    def validation_y(self):
        return self._validation_labels

    def num_training_files(self):
        return len(self._training_files)

    def num_validation_files(self):
        return len(self._validation_files)

    def num_classes(self):
        return len(np.unique(self._training_labels))

    def class_frequencies(self):
        return zip(np.unique(self._training_labels), np.bincount(self._training_labels))

    def balance_weights(self):
        total = self.num_training_files()
        class_counts = np.bincount(self._training_labels)
        return np.array([total / class_count for class_count in class_counts])

    def final_balance_weights(self):
        # Todo - should be easy to come up with a decent heuristic
        # manual final_balance_weights can always be used via the config
        pass

    def print_info(self):
        logger.info("Data: #Training images: %d" % self.num_training_files())
        logger.info("Data: #Classes: %d" % self.num_classes())
        logger.info("Data: Class frequencies: %s" % self.class_frequencies())
        logger.info("Data: Class balance weights: %s" % self.balance_weights())
        logger.info("Data: #Validation images: %d" %
                    self.num_validation_files())
