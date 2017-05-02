from __future__ import division, print_function, absolute_import

import numpy as np
import logging
from tensorflow.examples.tutorials.mnist import input_data
import tefla
from tefla.core.training import SupervisedTrainer
from tefla.core.lr_policy import StepDecayPolicy
from tefla.core.mem_dataset import DataSet
from tefla.utils import util
from fc_model import model

print('done importing')

import tensorflow as tf
np.random.seed(127)
tf.set_random_seed(127)


def train():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    width = 28
    height = 28

    train_images = mnist[0].images
    train_labels = mnist[0].labels

    validation_images = mnist[1].images
    validation_labels = mnist[1].labels

    data_set = DataSet(train_images, train_labels,
                       validation_images, validation_labels)
    training_cnf = {
        'classification': True,
        'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
        'num_epochs': 50,
        'lr_policy': StepDecayPolicy(
            schedule={
                0: 0.01,
                30: 0.001,
            }
        )
    }
    util.init_logging('train.log', file_log_level=logging.INFO,
                      console_log_level=logging.INFO)

    trainer = SupervisedTrainer(
        model, training_cnf, classification=training_cnf['classification'], is_summary=True)
    trainer.fit(data_set, weights_from=None,
                start_epoch=1, verbose=1)


if __name__ == '__main__':
    train()
