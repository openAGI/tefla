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
from tefla.core.layers import input, conv2d, fully_connected, max_pool, softmax, batch_norm_lasagne, prelu
from collections import OrderedDict
import logging

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

train_images = mnist[0].images.reshape(-1, 1, 28, 28)
train_labels = mnist[0].labels

validation_images = mnist[1].images.reshape(-1, 1, 28, 28)
validation_labels = mnist[1].labels

test_images = mnist[2].images.reshape(-1, 1, 28, 28)
test_labels = mnist[2].labels


class DataSet(object):
    def __init__(self, training_files, training_labels, validation_files, validation_labels):
        self.training_files = training_files
        self.training_labels = training_labels
        self.validation_files = validation_files
        self.validation_labels = validation_labels

    def print_info(self):
        print(train_images.shape, train_labels.shape)
        print(validation_images.shape, validation_labels.shape)
        print(test_images.shape, test_labels.shape)


data_set = DataSet(train_images, train_labels, validation_images, validation_labels)

training_iter = BatchIterator(32, False)
validation_iter = BatchIterator(32, False)


def model(is_training, reuse):
    outputs_collection = 'training_activations' if is_training else 'prediction_activations'
    common_args = common_layer_args(is_training, reuse, outputs_collection)
    bn_args = make_args(updates_collections=None)
    conv_args = make_args(batch_norm=batch_norm_lasagne, batch_norm_args=bn_args, activation=prelu, **common_args)
    fc_args = make_args(activation=prelu, **common_args)
    logit_args = make_args(activation=None, **common_args)
    pool_args = make_args(**common_args)

    x = input((None, cnf['w'], cnf['h'], 1), **common_args)
    x = conv2d(x, 20, name='conv1_1', **conv_args)
    x = max_pool(x, name='pool1', **pool_args)
    x = fully_connected(x, n_output=100, name='fc1', **fc_args)
    logits = fully_connected(x, n_output=10, name="logits", **logit_args)
    predictions = softmax(logits, name='predictions', **common_args)

    end_points = OrderedDict(tf.get_collection(outputs_collection))
    return end_points


cnf = {
    'name': __name__.split('.')[-1],
    'w': 28,
    'h': 28,
    'classification': True,
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
    'l2_reg': 0.0000,
    'summary_dir': '/media/lalit/data/summary/mnist',
    'schedule': {
        0: 0.01,
        50: 0.001,
        90: 0.00019,
        110: 0.00019,
        120: 0.0001,
        220: 0.00001,
        251: 'stop',
    },
}
util.init_logging('train.log', file_log_level=logging.INFO, console_log_level=logging.INFO)

trainer = SupervisedTrainer(model, cnf, training_iter, validation_iter, classification=cnf['classification'])
trainer.fit(data_set, None, 1, verbose=1, summary_every=10)
