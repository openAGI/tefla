from __future__ import division, print_function, absolute_import

import click
import numpy as np
import os

np.random.seed(127)
import tensorflow as tf

tf.set_random_seed(127)

from tefla.core.learning_seg import SupervisedLearner
from tefla.da.standardizer import NoOpStandardizer
from tefla.da.standardizer import AggregateStandardizerTF
from tefla.utils import util
import logging


@click.command()
@click.option('--model', default=None, show_default=True,
              help='Relative path to model.')
@click.option('--training_cnf', default=None, show_default=True,
              help='Relative path to training config file.')
@click.option('--data_dir', default=None, show_default=True,
              help='Path to training directory.')
@click.option('--parallel', default=True, show_default=True,
              help='parallel or queued.')
@click.option('--start_epoch', default=1, show_default=True,
              help='Epoch number from which to resume training.')
@click.option('--num_classes', default=17, show_default=True,
              help='Number of classes for training.')
@click.option('--gpu_memory_fraction', default=0.92, show_default=True,
              help='Epoch number from which to resume training.')
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
@click.option('--weights_dir', default=None, show_default=True,
              help='Path to save weights file.')
@click.option('--resume_lr', default=0.01, show_default=True,
              help='Path to initial weights file.')
@click.option('--loss_type', default='cross_entropy', show_default=True,
              help='Loss fuction type.')
@click.option('--log_file_name', default='train_seg.log', show_default=True,
              help='Log file name.')
@click.option('--is_summary', default=False, show_default=True,
              help='Path to initial weights file.')
def main(model, training_cnf, data_dir, parallel, start_epoch, weights_from, weights_dir, resume_lr, num_classes, gpu_memory_fraction, is_summary, loss_type, log_file_name):
    with tf.Graph().as_default():
        model_def = util.load_module(model)
        model = model_def.model
        cnf = util.load_module(training_cnf).cnf

        if weights_from:
            weights_from = str(weights_from)

        trainer = SupervisedLearner(model, cnf, log_file_name=log_file_name, resume_lr=resume_lr, classification=cnf[
                                    'classification'], gpu_memory_fraction=gpu_memory_fraction, num_classes=num_classes, is_summary=is_summary, loss_type=loss_type, verbosity=1)
        trainer.fit(data_dir, weights_from=weights_from, weights_dir=weights_dir, start_epoch=start_epoch,
                    summary_every=399, keep_moving_averages=True)


if __name__ == '__main__':
    main()
