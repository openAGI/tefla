import os

import click
import numpy as np

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
def predict(model, training_cnf, predict_dir, weights_from, dataset_name, convert, image_size, sync,
            test_type):
    model_def = util.load_module(model)
    model = model_def.model
    cnf = util.load_module(training_cnf).cnf
    weights_from = str(weights_from)
    images = data.get_image_files(predict_dir)

    standardizer = cnf.get('standardizer', None)

    preprocessor = convert_preprocessor(image_size) if convert else None
    prediction_iterator = create_prediction_iter(
        cnf, standardizer, model_def.crop_size, preprocessor, sync)

    if test_type == 'quasi':
        predictor = QuasiPredictor(
            model, cnf, weights_from, prediction_iterator, 20)
        predictions = predictor.predict(images)

    if not os.path.exists(os.path.join(predict_dir, '..', 'results')):
        os.mkdir(os.path.join(predict_dir, '..', 'results'))
    if not os.path.exists(os.path.join(predict_dir, '..', 'results', dataset_name)):
        os.mkdir(os.path.join(predict_dir, '..', 'results', dataset_name))

    names = data.get_names(images)
    image_prediction_prob = np.column_stack([names, predictions])
    headers = ['score%d' % (i + 1) for i in range(predictions.shape[1])]
    title = np.array(['image'] + headers)
    image_prediction_prob = np.vstack([title, image_prediction_prob])
    labels_file_prob = os.path.abspath(
        os.path.join(predict_dir, '..', 'results', dataset_name, 'predictions.csv'))
    np.savetxt(labels_file_prob, image_prediction_prob,
               delimiter=",", fmt="%s")


if __name__ == '__main__':
    predict()
