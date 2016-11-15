from __future__ import division, print_function, absolute_import

import click
import numpy as np

np.random.seed(127)
import tensorflow as tf

tf.set_random_seed(127)

from tefla.core.dir_dataset import DataSet
from tefla.da.standardizer import AggregateStandardizer, SamplewiseStandardizer
from tefla.core.iter_ops import create_training_iters
from tefla.core.training import SupervisedTrainer
from tefla.utils import util
import logging


@click.command()
@click.option('--model', default=None, show_default=True,
              help='Relative path to model.')
@click.option('--training_cnf', default=None, show_default=True,
              help='Relative path to training config file.')
@click.option('--data_dir', default=None, show_default=True,
              help='Path to training directory.')
@click.option('--data_standardizer', default='samplewise', show_default=True,
              help='samplewise or aggregate standardizer.')
@click.option('--iterator_type', default='queued', show_default=True,
              help='parallel or queued.')
@click.option('--start_epoch', default=1, show_default=True,
              help='Epoch number from which to resume training.')
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
def main(model, training_cnf, data_dir, data_standardizer, iterator_type, start_epoch, weights_from):
    model_def = util.load_module(model)
    model = model_def.model
    cnf = util.load_module(training_cnf).cnf

    util.init_logging('train.log', file_log_level=logging.INFO, console_log_level=logging.INFO)
    if weights_from:
        weights_from = str(weights_from)

    data_set = DataSet(data_dir, model_def.image_size[0])

    if data_standardizer == 'samplewise':
        standardizer = SamplewiseStandardizer(clip=6)
    else:
        standardizer = AggregateStandardizer(
            cnf['mean'],
            cnf['std'],
            cnf['u'],
            cnf['ev'],
            cnf['sigma']
        )

    training_iter, validation_iter = create_training_iters(cnf, data_set, standardizer, model_def.crop_size,
                                                           start_epoch, iterator_type == 'parallel')
    trainer = SupervisedTrainer(model, cnf, training_iter, validation_iter, classification=cnf['classification'])
    trainer.fit(data_set, weights_from, start_epoch, verbose=1, summary_every=10)


if __name__ == '__main__':
    main()
