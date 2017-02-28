from __future__ import division, print_function, absolute_import

import click
import numpy as np
import os

np.random.seed(127)
import tensorflow as tf

tf.set_random_seed(127)

from tefla.core.learningv2 import SupervisedLearner
from tefla.da.standardizer import NoOpStandardizer
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
@click.option('--gpu_memory_fraction', default=0.92, show_default=True,
              help='Epoch number from which to resume training.')
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
@click.option('--resume_lr', default=0.01, show_default=True,
              help='Path to initial weights file.')
@click.option('--loss_type', default='cross_entropy', show_default=True,
              help='Loss fuction type.')
@click.option('--is_summary', default=False, show_default=True,
              help='Path to initial weights file.')
def main(model, training_cnf, data_dir, parallel, start_epoch, weights_from, resume_lr, gpu_memory_fraction, is_summary, loss_type):
    model_def = util.load_module(model)
    model = model_def.model
    cnf = util.load_module(training_cnf).cnf

    util.init_logging('train.log', file_log_level=logging.INFO,
                      console_log_level=logging.INFO)
    if weights_from:
        weights_from = str(weights_from)

    trainer = SupervisedLearner(model, cnf, resume_lr=resume_lr, classification=cnf[
                                'classification'], gpu_memory_fraction=gpu_memory_fraction, is_summary=is_summary, loss_type=loss_type, verbosity=1)
    data_dir_train = os.path.join(data_dir, 'train')
    data_dir_val = os.path.join(data_dir, 'val')
    trainer.fit(data_dir_train, data_dir_val, weights_from=weights_from, start_epoch=start_epoch, training_set_size=50000, val_set_size=10000,
                summary_every=399, keep_moving_averages=True)


if __name__ == '__main__':
    main()
