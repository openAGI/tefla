# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import

import functools
import logging

from .. import convert
from ..da import iterator

logger = logging.getLogger('tefla')


class Dataflow(object):

    def __init__(self, cnf, dataset=None, standardizer=None, crop_size=None, epoch=1, parallel=True):
        """
        Creates training iterator to access and augment the dataset in numpy format

        Args:
            cnf: configs dict with all training and augmentation params
            data_set: an instance of the dataset class
            standardizer: data samples standardization; either samplewise or aggregate
            crop_size: training time crop_size of the data samples
            epoch: the current epoch number; used for data balancing
            parallel: iterator type; either parallel or queued

        Usage:
            dataflow  = Dataflow(...)
            for i in number of range(num_epochs):
                dataflow.reset_iter(mode='training')
                while True:
                    try:
                        batch = next(it)
                    except Exception:
                        break
        """
        self.cnf = cnf if cnf is not None else {'batch_size_test': 1}
        self.dataset = dataset
        self.standardizer = standardizer
        self.crop_size = (crop_size, crop_size) if not isinstance(
            crop_size, (list, tuple)) else crop_size
        self.epoch = epoch
        self.parallel = parallel
        self.training_iter_object = self.create_training_iters()
        self.training_iter = iter(self.training_iter_object(
            self.dataset.training_X, self.dataset.training_y))
        self.validation_iter_object = self.create_validation_iters()
        self.validation_iter = iter(
            self.validation_iter_object(self.dataset.validation_X, self.dataset.validation_y))

    def reset_iterator(self, mode='training'):
        """
        Args:
            mode: a string, training/validation

        """
        if mode == 'trianing':
            self.training_iter = iter(self.training_iter_object(
                self.dataset.training_X, self.dataset.training_y))
        else:
            self.validation_iter = iter(
                self.validation_iter_object(self.dataset.validation_X, self.dataset.validation_y))

    def get_batch(self, mode='training'):
        """
        Args:
            mode: a string, training/validation

        Returns:
           a batch of data

        """
        if mode == 'training':
            return next(self.training_iter)
        else:
            return next(self.validation_iter)

    def create_training_iters(self):
        if self.parallel:
            training_iterator_maker = iterator.BalancingDAIterator
            logger.info('Using parallel iterators')
        else:
            training_iterator_maker = iterator.BalancingQueuedDAIterator
            logger.info('Using queued iterators')

        preprocessor = None
        training_iterator = training_iterator_maker(
            batch_size=self.cnf['batch_size_train'],
            shuffle=True,
            preprocessor=preprocessor,
            crop_size=self.crop_size,
            is_training=True,
            aug_params=self.cnf['aug_params'],
            balance_weights=self.dataset.balance_weights(),
            final_balance_weights=self.cnf['final_balance_weights'],
            balance_ratio=self.cnf['balance_ratio'],
            balance_epoch_count=self.epoch - 1,
            standardizer=self.standardizer,
            fill_mode='constant'
        )
        return training_iterator

    def create_validation_iters(self):
        if self.parallel:
            validation_iterator_maker = iterator.ParallelDAIterator
            logger.info('Using parallel iterators')
        else:
            validation_iterator_maker = iterator.ParallelDAIterator
            logger.info('Using queued iterators')

        preprocessor = None
        validation_iterator = validation_iterator_maker(
            batch_size=self.cnf['batch_size_test'],
            shuffle=False,
            preprocessor=preprocessor,
            crop_size=self.crop_size,
            is_training=False,
            standardizer=self.standardizer,
            fill_mode='constant'
        )

        return validation_iterator

    def convert_preprocessor(self, im_size):
        return functools.partial(convert.convert, target_size=im_size)

    def create_prediction_iter(self, preprocessor=None, sync=False):
        """
        Creates prediction iterator to access and augment the dataset

        Args:
            preprocessor: data processing or cropping function
            sync: a bool, if False, used parallel iterator
        """
        if sync:
            prediction_iterator_maker = iterator.DAIterator
        else:
            prediction_iterator_maker = iterator.ParallelDAIterator

        if preprocessor is None:
            preprocessor = self.convert_preprocessor(self.crop_size[0])

        prediction_iterator = prediction_iterator_maker(
            batch_size=self.cnf['batch_size_test'],
            shuffle=False,
            preprocessor=preprocessor,
            crop_size=self.crop_size,
            is_training=False,
            standardizer=self.standardizer,
            fill_mode='constant'
        )

        return prediction_iterator
