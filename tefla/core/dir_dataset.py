"""A dataset based on files in a training and validation directory, and
corresponding label files."""
from __future__ import division, print_function, absolute_import

import numpy as np

from . import data_load_ops as data
from . import logger


class DataSet(object):

  def __init__(self, data_dir, img_size, mode='classification', multilabel=False):
    self.data_dir = data_dir
    training_images_dir = "%s/training_%d" % (data_dir, img_size)
    training_labels_file = "%s/training_labels.csv" % data_dir

    validation_images_dir = "%s/validation_%d" % (data_dir, img_size)
    validation_labels_file = "%s/validation_labels.csv" % data_dir

    self._training_files = data.get_image_files(training_images_dir)
    names = data.get_names(self._training_files)
    self._training_labels = data.get_labels(
        names, label_file=training_labels_file, multilabel=multilabel).astype(np.int32)

    self._validation_files = data.get_image_files(validation_images_dir)
    names = data.get_names(self._validation_files)
    self._validation_labels = data.get_labels(
        names, label_file=validation_labels_file, multilabel=multilabel).astype(np.int32)

    # make sure no wrong labels exist
    if mode == 'classification':
      self._training_labels = self.check_labels(self._training_labels, 'training')
      self._validation_labels = self.check_labels(self._validation_labels, 'validation')

  def check_labels(self, labels, split):
    negative_ids = np.where(labels < 0)[0]
    if negative_ids.shape[0] > 0:
      logger.info('Possible mistakes in the {} dataset, found negative labels \
          Total: {}'.format(split, negative_ids.shape[0]))
      logger.info('Setting negative labels to zero, take action if ncessary')
      labels[negative_ids] = 0
    return labels

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
    if len(self._training_labels.shape) > 1:
      return self._training_labels.shape[1]
    return len(np.unique(self._training_labels))

  def class_frequencies(self):
    if len(self._training_labels.shape) > 1:
      return list(zip(list(range(self.num_classes())), np.sum(self._training_labels, axis=0)))
    else:
      return list(zip(np.unique(self._training_labels), np.bincount(self._training_labels)))

  def balance_weights(self):
    total = self.num_training_files()
    if len(self._training_labels.shape) > 1:
      class_counts = np.sum(self._training_labels, axis=0)
    else:
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
    logger.info("Data: #Validation images: %d" % self.num_validation_files())
