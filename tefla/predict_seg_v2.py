import os
import click
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy
import scipy.misc
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage import io
from skimage import transform
from skimage.util import img_as_float

from tefla.core.iter_ops import create_prediction_iter, convert_preprocessor
from tefla.core.prediction_v2 import SegmentPredictor
from tefla.da import data
from tefla.utils import util
from tefla.convert import convert

import tensorflow as tf


@click.command()
@click.option('--frozen_model', default=None, show_default=True,
              help='Relative path to model.')
@click.option('--training_cnf', default=None, show_default=True,
              help='Relative path to training config file.')
@click.option('--predict_dir', help='Directory with Test Images')
@click.option('--dataset_name', default='dataset', help='Name of the dataset')
@click.option('--convert1', is_flag=True,
              help='Convert/preprocess files before prediction.')
@click.option('--image_size', default=256, show_default=True,
              help='Image size for conversion.')
@click.option('--sync', is_flag=True,
              help='Do all processing on the calling thread.')
@click.option('--test_type', default='quasi', help='Specify test type, crop_10 or quasi')
@click.option('--gpu_memory_fraction', default=0.92, show_default=True,
              help='GPU memory fraction to use.')
def predict(frozen_model, training_cnf, predict_dir, dataset_name, convert1, image_size, sync,
            test_type, gpu_memory_fraction):
    cnf = util.load_module(training_cnf).cnf
    standardizer = cnf['standardizer']
    graph = util.load_frozen_graph(frozen_model)
    preprocessor = convert_preprocessor(448)
    predictor = SegmentPredictor(graph, standardizer, preprocessor)
    images = data.get_image_files(predict_dir)
    for image_filename in images:
        final_prediction_map = predictor.predict(image_filename)
    final_prediction_map = final_prediction_map.transpose(0, 2, 1)
    fig = plt.figure("segments")
    ax = fig.add_subplot(1, 1, 1)
    image = data.load_image(image_filename, preprocessor=preprocessor)
    img = image.transpose(2, 1, 0)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img.save('/tmp/test.png')
    image = io.imread('/tmp/test.png')
    ax.imshow(mark_boundaries(image,
                              final_prediction_map.squeeze()))
    plt.axis("off")
    plt.show()
    try:
        print(sum(sum(final_prediction_map.squeeze())))
        plt.imshow(final_prediction_map.squeeze())
    except Exception:
        print(sum(sum(final_prediction_map[1].squeeze())))
        plt.imshow(final_prediction_map[1].squeeze())
    plt.show()


if __name__ == '__main__':
    predict()
