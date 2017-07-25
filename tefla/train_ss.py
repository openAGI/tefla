from __future__ import division, print_function, absolute_import

import click
import numpy as np

np.random.seed(127)
import tensorflow as tf

tf.set_random_seed(127)

from tefla.core.dir_dataset import DataSet
from tefla.core.iter_ops import create_training_iters
from tefla.core.learning_ss import SemiSupervisedTrainer
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
@click.option('--is_summary', default=False, show_default=True,
              help='Path to initial weights file.')
@click.option('--log_file_name', default='train_ss.log', show_default=True,
              help='Log file name.')
@click.option('--num_classes', default=6, show_default=True,
              help='Number of classes to train the model.')
def main(model, training_cnf, data_dir, parallel, start_epoch, weights_from, resume_lr, gpu_memory_fraction, is_summary, num_classes, log_file_name):
    model_def = util.load_module(model)
    model = model_def
    cnf = util.load_module(training_cnf).cnf

    if weights_from:
        weights_from = str(weights_from)

    data_set = DataSet(data_dir, model_def.image_size[0])
    standardizer = cnf.get('standardizer', NoOpStandardizer())

    training_iter, validation_iter = create_training_iters(
        cnf, data_set, standardizer, model_def.crop_size, start_epoch, parallel=parallel)
    trainer = SemiSupervisedTrainer(model, cnf, training_iterator=training_iter, validation_iterator=validation_iter, resume_lr=resume_lr, classification=cnf[
        'classification'], gpu_memory_fraction=gpu_memory_fraction, is_summary=is_summary, verbosity=1, log_file_name=log_file_name)
    trainer.fit(data_set, num_classes, weights_from,
                start_epoch, summary_every=399)


if __name__ == '__main__':
    main()
