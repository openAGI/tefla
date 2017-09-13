from __future__ import division, print_function, absolute_import

import abc
import six
import time
from scipy.stats.mstats import gmean
import numpy as np
import tensorflow as tf
from ..da import tta
from ..da import data
from ..utils import util
from .special_layers import dense_crf


@six.add_metaclass(abc.ABCMeta)
class PredictSession(object):
    """
    base mixin class for prediction

    Args:
        graph: graph with weights and variables
        gpu_memory_fraction: fraction of gpu memory to use, if not cpu prediction
    """

    def __init__(self, graph, gpu_memory_fraction=None):
        self.graph = graph
        if gpu_memory_fraction is not None:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
            self.sess = tf.Session(graph=self.graph,
                                   config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto())

    def predict(self, X):
        with self.graph.as_default():
            return self._real_predict(X)

    @abc.abstractmethod
    def _real_predict(self, X):
        raise NotImplementedError


class OneCropPredictor(PredictSession):
    """One crop Predictor, it predict network out put from a single crop of an input image

    Args:
        graph: graph with weights and variables
        cnf: prediction configs
        prediction_iterator: iterator to access and augment the data for prediction
        gpu_memory_fraction: fraction of gpu memory to use, if not cpu prediction
    """

    def __init__(self, graph, prediction_iterator, input_tensor_name='model/inputs/input:0', predict_tensor_name='model/predictions/Softmax:0'):
        self.prediction_iterator = prediction_iterator
        self.inputs = graph.get_tensor_by_name(input_tensor_name)
        self.predictions = graph.get_tensor_by_name(predict_tensor_name)
        super(OneCropPredictor, self).__init__(graph)

    def _real_predict(self, X, xform=None, crop_bbox=None):
        tic = time.time()
        print('Making %d predictions' % len(X))
        data_predictions = []
        for X, y in self.prediction_iterator(X, xform=xform, crop_bbox=crop_bbox):
            predictions_e = self.sess.run(
                self.predictions, feed_dict={self.inputs: X})
            data_predictions.append(predictions_e)
        data_predictions = np.vstack(data_predictions)
        print('took %6.1f seconds' % (time.time() - tic))
        return data_predictions


class QuasiCropPredictor(PredictSession):
    """Quasi transform predictor

    Args:
        graph: graph with weights and variables
        cnf: prediction configs
        prediction_iterator: iterator to access and augment the data for prediction
        number_of_transform: number of determinastic augmentaions to be performed on the input data
            resulted predictions are averaged over the augmentated transformation prediction outputs
        gpu_memory_fraction: fraction of gpu memory to use, if not cpu prediction
    """

    def __init__(self, graph, cnf, prediction_iterator, number_of_transforms, input_tensor_name='model/inputs/input:0', predict_tensor_name='model/predictions/Softmax:0'):
        self.cnf = cnf
        self.prediction_iterator = prediction_iterator
        self.number_of_transforms = number_of_transforms
        self.predictor = OneCropPredictor(
            graph, prediction_iterator, input_tensor_name, predict_tensor_name)
        super(QuasiCropPredictor, self).__init__(graph)

    def _real_predict(self, X):
        standardizer = self.prediction_iterator.standardizer
        da_params = standardizer.da_processing_params()
        util.veryify_args(da_params, [
                          'sigma'], 'QuasiCropPredictor > standardizer does unknown da with param(s):')
        color_sigma = da_params.get('sigma', 0.0)
        tfs, color_vecs = tta.build_quasirandom_transforms(self.number_of_transforms, color_sigma=color_sigma,
                                                           **self.cnf['aug_params'])
        multiple_predictions = []
        for i, (xform, color_vec) in enumerate(zip(tfs, color_vecs), start=1):
            print('Quasi-random tta iteration: %d' % i)
            standardizer.set_tta_args(color_vec=color_vec)
            predictions = self.predictor._real_predict(X, xform=xform)
            multiple_predictions.append(predictions)
        return np.mean(multiple_predictions, axis=0)


class TenCropPredictor(PredictSession):
    """Multiples non Data augmented crops predictor

    Args:
        graph: graph with weights and variables
        cnf: prediction configs
        prediction_iterator: iterator to access and augment the data for prediction
        crop_size: crop size for network input
        im_size: original image size
        number_of_crops: total number of crops to extract from the input image
        gpu_memory_fraction: fraction of gpu memory to use, if not cpu prediction
        """

    def __init__(self, graph, prediction_iterator, crop_size, im_size, input_tensor_name='model/inputs/input:0', predict_tensor_name='model/predictions/Softmax:0'):
        self.prediction_iterator = prediction_iterator
        self.predictor = OneCropPredictor(
            graph, prediction_iterator, input_tensor_name, predict_tensor_name)
        self.crop_size = crop_size
        self.im_size = im_size
        self.prediction_iterator = prediction_iterator
        super(TenCropPredictor, self).__init__(graph)

    def _real_predict(self, X):
        crop_size = np.array(self.crop_size)
        im_size = np.array(self.im_size)
        bboxs = util.get_bbox_10crop(crop_size, im_size)
        multiple_predictions = []
        for i, bbox in enumerate(bboxs, start=1):
            print('Crop-deterministic iteration: %d' % i)
            predictions = self.predictor._real_predict(X, crop_bbox=bbox)
            multiple_predictions.append(predictions)
        return np.mean(multiple_predictions, axis=0)


class EnsemblePredictor(object):
    """Returns predcitions from multiples models

    Ensembled predictions from multiples models using ensemble type

    Args:
        predictors: predictor instances
    """

    def __init__(self, predictors):
        self.predictors = predictors

    def predict(self, X, ensemble_type='mean'):
        """
        Returns ensembled predictions for an input or batch of inputs

        Args:
            X: 4D tensor, inputs
            ensemble_type: operation to combine models probabilities
                    available type: ['mean', 'gmean', 'log_mean']
        """
        multiple_predictions = []
        for p in self.predictors:
            print('Ensembler - running predictions using: %s' % p)
            predictions = p.predict(X)
            multiple_predictions.append(predictions)
        multiple_predictions = np.array(multiple_predictions, dtype=np.float32)
        return _ensemble(ensemble_type, multiple_predictions)


def _ensemble(en_type, x):
    return {
        'mean': np.mean(x, axis=0),
        'gmean': gmean(x, axis=0),
        'log_mean': np.mean(np.log(x + (x == 0)), axis=0),
    }[en_type]


class SegmentPredictor(PredictSession):
    """One crop Predictor, it predict network out put from a single crop of an input image

    Args:
        graph: graph with weights and variables
        cnf: prediction configs
        standardizer: standardizer for the  input data for prediction
        gpu_memory_fraction: fraction of gpu memory to use, if not cpu prediction
    """

    def __init__(self, graph, standardizer, preprocessor, input_tensor_name='model/inputs/input:0', predict_tensor_name='model/ArgMax:0'):
        self.standardizer = standardizer
        self.preprocessor = preprocessor
        self.inputs = graph.get_tensor_by_name(input_tensor_name)
        self.predictions = graph.get_tensor_by_name(predict_tensor_name)
        super(SegmentPredictor, self).__init__(graph)

    def _real_predict(self, X, xform=None, crop_bbox=None):
        tic = time.time()
        X = data.load_image(X, preprocessor=self.preprocessor)
        X = self.standardizer(X, False)
        X = X.transpose(1, 2, 0)
        X = np.expand_dims(X, 0)
        predictions = self.sess.run(
            self.predictions, feed_dict={self.inputs: X})
        predictions = predictions.transpose(0, 2, 1)
        print('took %6.1f seconds' % (time.time() - tic))
        return predictions


class SegmentPredictor_v2(PredictSession):
    """One crop Predictor, it predict network out put from a single crop of an input image

    Args:
        graph: graph with weights and variables
        cnf: prediction configs
        standardizer: standardizer for the  input data for prediction
        gpu_memory_fraction: fraction of gpu memory to use, if not cpu prediction
    """

    def __init__(self, graph, standardizer, preprocessor, input_tensor_name='model/inputs/input:0', predict_tensor_name='model/final_map_logits/BiasAdd:0'):
        self.standardizer = standardizer
        self.preprocessor = preprocessor
        self.inputs = graph.get_tensor_by_name(input_tensor_name)
        self.predictions = graph.get_tensor_by_name(predict_tensor_name)
        super(SegmentPredictor_v2, self).__init__(graph)

    def _real_predict(self, X, xform=None, crop_bbox=None):
        tic = time.time()
        img_orig = data.load_image(X, preprocessor=self.preprocessor)
        img_orig = np.asarray(img_orig.transpose(1, 2, 0), dtype=np.uint8)
        X = data.load_image(X, preprocessor=self.preprocessor)
        X = self.standardizer(X, False)
        X = X.transpose(1, 2, 0)
        X = np.expand_dims(X, 0)
        raw_output_up = tf.nn.softmax(self.predictions)
        raw_output_up = tf.py_func(
            dense_crf, [raw_output_up, tf.expand_dims(img_orig, axis=0)], tf.float32)
        raw_output_up = tf.argmax(raw_output_up, dimension=3)
        predictions = self.sess.run(
            raw_output_up, {self.inputs: X})
        predictions = predictions.transpose(0, 2, 1)
        print('took %6.1f seconds' % (time.time() - tic))
        return predictions
