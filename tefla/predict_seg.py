import os
import click
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import scipy
import scipy.misc
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage import io
from skimage import transform
from skimage.util import img_as_float

from tefla.core.iter_ops import create_prediction_iter, convert_preprocessor
from tefla.core.prediction_v2 import SegmentPredictor_v2 as SegmentPredictor
from tefla.da import data
from tefla.utils import util
from tefla.convert import convert

import tensorflow as tf


class SegParams(object):

    def __init__(self, name='SegParams'):
        self.name = name

    @staticmethod
    def feature_classes(self):
        classes = {1: 'Person',
                   2: 'Car',
                   3: 'Bike',
                   4: 'Bicycle',
                   5: 'Truck',
                   6: 'Taxi',
                   7: 'Bus',
                   8: 'Horse',
                   9: 'Cat',
                   10: 'Dog',
                   11: 'Bird',
                   12: 'Train',
                   13: 'Aeroplane',
                   14: 'Table', }
        return classes

    @staticmethod
    def feature_palette(self):
        palette = {
            1: (138 / 255.0, 46 / 255.0, 226 / 255.0),
            2: (173 / 255.0, 1, 47 / 255.0),
            3: (34 / 255.0, 139 / 255.0, 34 / 255.0),
            4: (233 / 255.0, 150 / 255.0, 122 / 255.0),
            5: (128 / 255.0,   0, 128 / 255.0),
            6: (0, 128 / 255.0, 128 / 255.0),
            7: (128 / 255.0, 128 / 255.0, 128 / 255.0),
            8: (64 / 255.0,   0,   0),
            9: (192 / 255.0,   0,   0),
            10: (64 / 255.0, 128 / 255.0,   0),
            11: (193 / 255.0, 1, 193 / 255.0),
            12: (64 / 255.0,   0, 128 / 255.0),
            13: (192 / 255.0,   0, 128 / 255.0),
            14: (64 / 255.0, 128 / 255.0, 128 / 255.0)}
        return palette


def plot_masks(cropped_image_path, prediction_map, output_image_path):
    fig = plt.figure("segments")
    ax = fig.add_subplot(1, 1, 1)
    image_draw = io.imread(cropped_image_path)
    segparams = SegParams()
    feature_mapping = segparams.feature_palette()
    classes = segparams.feature_classes()
    legend_patches = []
    for i in feature_mapping.keys():
        if i in prediction_map:
            temp_inds = np.where(prediction_map != i)
            temp_map = prediction_map.copy()
            temp_map[temp_inds] = 0
            image_draw = mark_boundaries(
                image_draw, temp_map, mode='inner', color=feature_mapping[i])  # outline_color=feature_mapping[i])
            legend_patches.append(mpatches.Patch(
                color=(feature_mapping[i][0], feature_mapping[i][1], feature_mapping[i][2], 1), label=classes[i]))
    ax.imshow(image_draw)
    lgd = ax.legend(handles=legend_patches,
                    loc="upper left", bbox_to_anchor=(1, 1))
    plt.axis("off")
    plt.savefig(output_image_path.strip('.jpg') + '_segmented.png', bbox_extra_artists=(
        lgd,), bbox_inches='tight')
    plt.show()


@click.command()
@click.option('--frozen_model', default=None, show_default=True,
              help='Relative path to model.')
@click.option('--training_cnf', default=None, show_default=True,
              help='Relative path to training config file.')
@click.option('--image_path', help='Directory with Test Images')
@click.option('--image_size', default=448, show_default=True,
              help='Image size for conversion.')
@click.option('--output_path', default='/tmp/test', help='Output Dir to save the segmented image')
@click.option('--gpu_memory_fraction', default=0.92, show_default=True,
              help='GPU memory fraction to use.')
def predict(frozen_model, training_cnf, image_path, image_size, output_path,
            gpu_memory_fraction):
    cnf = util.load_module(training_cnf).cnf
    standardizer = cnf['standardizer']
    graph = util.load_frozen_graph(frozen_model)
    preprocessor = convert_preprocessor(448)
    predictor = SegmentPredictor(graph, standardizer, preprocessor)
    final_prediction_map = predictor.predict(image_path)
    final_prediction_map = final_prediction_map.transpose(0, 2, 1).squeeze()
    image = data.load_image(image_path, preprocessor=preprocessor)
    img = image.transpose(2, 1, 0)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img.save('/tmp/test.png')
    image_filename = image_path.split('/')[-1]
    plot_masks('/tmp/test.png', final_prediction_map, output_path)

    """
    im1 = plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    im2 = plt.imshow(final_prediction_map, cmap=plt.cm.viridis,
                     alpha=.2, interpolation='bilinear')
    plt.show()
    try:
        print(sum(sum(final_prediction_map.squeeze())))
        plt.imshow(final_prediction_map.squeeze())
    except Exception:
        print(sum(sum(final_prediction_map[1].squeeze())))
        plt.imshow(final_prediction_map[1].squeeze())
    plt.show()
    """


if __name__ == '__main__':
    predict()
