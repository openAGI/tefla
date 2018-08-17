# -------------------------------------------------------------------#
# Tool to eval segmentation performance
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
import os
import click

from tefla.core.iter_ops import convert_preprocessor
from tefla.core.prediction_v2 import SegmentPredictor_v2 as SegmentPredictor
from tefla.utils import util
from tefla.core.metrics import IOUSeg as IOU

# pylint: disable=no-value-for-parameter


@click.command()
@click.option('--frozen_model', default=None, show_default=True, help='Relative path to model.')
@click.option(
    '--training_cnf', default=None, show_default=True, help='Relative path to training config file.')
@click.option('--predict_dir', help='Directory with Test Images and Labls (_final_mask.png)')
@click.option('--image_size', default=448, show_default=True, help='image size for conversion.')
@click.option('--num_classes', default=15, show_default=True, help='Number of classes.')
@click.option('--output_path', default='/tmp/test', help='Output Dir to save the segmented image')
@click.option(
    '--gpu_memory_fraction', default=0.92, show_default=True, help='GPU memory fraction to use.')
def predict(frozen_model, training_cnf, predict_dir, image_size, output_path, num_classes,
            gpu_memory_fraction):
  cnf = util.load_module(training_cnf).cnf
  standardizer = cnf['standardizer']
  graph = util.load_frozen_graph(frozen_model)
  preprocessor = convert_preprocessor(image_size)
  predictor = SegmentPredictor(graph, standardizer, preprocessor)
  # images = data.get_image_files(predict_dir)
  image_names = [
      filename.strip() for filename in os.listdir(predict_dir) if filename.endswith('.jpg')
  ]

  iou = IOU()
  per_class_iou = iou.per_class_iou(predictor, predict_dir, image_size)
  meaniou = iou.meaniou(predictor, predict_dir, image_size)
  print(per_class_iou)
  print('Mean IOU %5.5f' % meaniou)


if __name__ == '__main__':
  predict()
