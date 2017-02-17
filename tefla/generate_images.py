import tensorflow as tf
import os

import click
import numpy as np
import cv2

from tefla.core.iter_ops import create_prediction_iter, convert_preprocessor
from tefla.core.prediction import QuasiPredictor
from tefla.da import data
from tefla.utils import util


@click.command()
@click.option('--model', default=None, show_default=True,
              help='Relative path to model.')
@click.option('--training_cnf', default=None, show_default=True,
              help='Relative path to training config file.')
@click.option('--predict_dir', help='Directory with Test Images')
@click.option('--weights_from', help='Path to initial weights file.')
@click.option('--dataset_name', default='dataset', help='Name of the dataset')
@click.option('--convert', is_flag=True,
              help='Convert/preprocess files before prediction.')
@click.option('--image_size', default=256, show_default=True,
              help='Image size for conversion.')
@click.option('--sync', is_flag=True,
              help='Do all processing on the calling thread.')
@click.option('--test_type', default='quasi', help='Specify test type, crop_10 or quasi')
@click.option('--gpu_memory_fraction', default=0.92, show_default=True,
              help='GPU memory fraction to use.')
def predict(model, training_cnf, predict_dir, weights_from, dataset_name, convert, image_size, sync,
            test_type, gpu_memory_fraction):
    model = util.load_module(model)
    cnf = util.load_module(training_cnf).cnf
    weights_from = str(weights_from)
    with tf.Graph().as_default():
        end_points_G = model.generator(
            [32, 100], True, None)
        inputs = tf.placeholder(tf.float32, shape=(
            None, model.image_size[0], model.image_size[0], 3), name="input")
        end_points_D = model.discriminator(
            inputs, True, None, num_classes=6, batch_size=32)
        saver = tf.train.Saver()
        print('Loading weights from: %s' % weights_from)
        if gpu_memory_fraction is not None:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        saver.restore(sess, weights_from)
        end_points_G_val = model.generator(
            [cnf['batch_size_test'], 100], False, True, batch_size=cnf['batch_size_test'])

        util.save_images('generated_images.png',
                         sess.run(end_points_G_val['softmax']), width=128, height=128)

        sess.close()


if __name__ == '__main__':
    predict()
