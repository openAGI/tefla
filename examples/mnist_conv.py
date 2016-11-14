"""Trains a simple convolutional net on the MNIST dataset.
Gets to 99.5% validation set accuracy.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

np.random.seed(127)
import tensorflow as tf

tf.set_random_seed(127)

from tensorflow.examples.tutorials.mnist import input_data
from tefla.da.iterator import BatchIterator
from tefla.core.training import SupervisedTrainer
from tefla.utils import util
from tefla.core.layer_arg_ops import common_layer_args, make_args
from tefla.core.layers import input, conv2d, fully_connected, max_pool, softmax, batch_norm_tf, prelu, dropout
from tefla.core.mem_dataset import DataSet
from collections import OrderedDict
import logging

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

width = 28
height = 28

train_images = mnist[0].images.reshape(-1, height, width, 1)
train_labels = mnist[0].labels

validation_images = mnist[1].images.reshape(-1, height, width, 1)
validation_labels = mnist[1].labels

data_set = DataSet(train_images, train_labels, validation_images, validation_labels)


def model(is_training, reuse):
    outputs_collection = 'training_activations' if is_training else 'prediction_activations'
    common_args = common_layer_args(is_training, reuse, outputs_collection)
    conv_args = make_args(batch_norm=True, activation=prelu, **common_args)
    fc_args = make_args(activation=prelu, **common_args)
    logit_args = make_args(activation=None, **common_args)
    pool_args = make_args(**common_args)

    x = input((None, height, width, 1), **common_args)
    x = conv2d(x, 32, name='conv1_1', **conv_args)
    x = conv2d(x, 32, name='conv1_2', **conv_args)
    x = max_pool(x, name='pool1', **pool_args)
    x = dropout(x, p=0.25, name='dropout1', **common_args)
    x = fully_connected(x, n_output=128, name='fc1', **fc_args)
    x = dropout(x, p=0.5, name='dropout2', **common_args)
    logits = fully_connected(x, n_output=10, name="logits", **logit_args)
    predictions = softmax(logits, name='predictions', **common_args)

    end_points = OrderedDict(tf.get_collection(outputs_collection))
    return end_points


cnf = {
    'name': __name__.split('.')[-1],
    'classification': True,
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
    'l2_reg': 0.0000,
    'summary_dir': '/media/lalit/data/summary/mnist_conv',
    'schedule': {
        0: 0.01,
        30: 0.001,
        50: 'stop',
    },
}
util.init_logging('train.log', file_log_level=logging.INFO, console_log_level=logging.INFO)

training_iter = BatchIterator(32, False)
validation_iter = BatchIterator(32, False)
trainer = SupervisedTrainer(model, cnf, training_iter, validation_iter, classification=cnf['classification'])
trainer.fit(data_set, weights_from=None, start_epoch=1, verbose=1, summary_every=10)
