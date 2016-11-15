"""Trains a simple convolutional net on the MNIST dataset.
Gets to 99.5% validation set accuracy.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

np.random.seed(127)
import tensorflow as tf

tf.set_random_seed(127)

from tensorflow.examples.tutorials.mnist import input_data
from tefla.core.training import SupervisedTrainer
from tefla.utils import util
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import input, conv2d, fully_connected, max_pool, softmax, prelu, dropout
from tefla.core.mem_dataset import DataSet
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
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, **common_args)
    fc_args = make_args(activation=prelu, **common_args)
    logit_args = make_args(activation=None, **common_args)

    x = input((None, height, width, 1), **common_args)
    x = conv2d(x, 32, name='conv1_1', **conv_args)
    x = conv2d(x, 32, name='conv1_2', **conv_args)
    x = max_pool(x, name='pool1', **common_args)
    x = dropout(x, p=0.25, name='dropout1', **common_args)
    x = fully_connected(x, n_output=128, name='fc1', **fc_args)
    x = dropout(x, p=0.5, name='dropout2', **common_args)
    logits = fully_connected(x, n_output=10, name="logits", **logit_args)
    predictions = softmax(logits, name='predictions', **common_args)

    return end_points(is_training)


training_cnf = {
    'classification': True,
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
    'schedule': {
        0: 0.01,
        30: 0.001,
        50: 'stop',
    },
}
util.init_logging('train.log', file_log_level=logging.INFO, console_log_level=logging.INFO)

trainer = SupervisedTrainer(model, training_cnf, classification=training_cnf['classification'])
trainer.fit(data_set, weights_from=None, start_epoch=1, verbose=1, summary_every=10)
