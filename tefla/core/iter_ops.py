from __future__ import division, print_function, absolute_import

import functools
import logging

from tefla import convert
from tefla.da import iterator

logger = logging.getLogger('tefla')


def create_training_iters(cnf, data_set, standardizer, epoch, parallel=True):
    if parallel:
        training_iterator_maker = iterator.BalancingDAIterator
        validation_iterator_maker = iterator.ParallelDAIterator
        logger.info('Using parallel iterators')
    else:
        training_iterator_maker = iterator.BalancingQueuedDAIterator
        validation_iterator_maker = iterator.ParallelDAIterator
        # validation_iterator_maker = iterator.QueuedDAIterator
        logger.info('Using queued iterators')

    crop_size = (cnf['w'], cnf['h'])
    preprocessor = None
    training_iterator = training_iterator_maker(
        batch_size=cnf['batch_size_train'],
        shuffle=True,
        preprocessor=preprocessor,
        crop_size=crop_size,
        is_training=True,
        aug_params=cnf['aug_params'],
        balance_weights=data_set.balance_weights(),
        final_balance_weights=cnf['final_balance_weights'],
        balance_ratio=cnf['balance_ratio'],
        balance_epoch_count=epoch - 1,
        standardizer=standardizer,
        fill_mode='constant'
        # save_to_dir=da_training_preview_dir
    )

    validation_iterator = validation_iterator_maker(
        batch_size=cnf['batch_size_test'],
        shuffle=False,
        preprocessor=preprocessor,
        crop_size=crop_size,
        is_training=False,
        standardizer=standardizer,
        fill_mode='constant'
    )

    return training_iterator, validation_iterator


def convert_preprocessor(im_size):
    return functools.partial(convert.convert, crop_size=im_size)


def create_prediction_iter(cnf, standardizer, preprocessor=None, sync=False):
    crop_size = (cnf['w'], cnf['h'])
    if sync:
        prediction_iterator_maker = iterator.DAIterator
    else:
        prediction_iterator_maker = iterator.ParallelDAIterator

    prediction_iterator = prediction_iterator_maker(
        batch_size=cnf['batch_size_test'],
        shuffle=False,
        preprocessor=preprocessor,
        crop_size=crop_size,
        is_training=False,
        standardizer=standardizer,
        fill_mode='constant'
    )

    return prediction_iterator
