from __future__ import division, print_function, absolute_import

import time

import numpy as np
import tensorflow as tf

from tefla.da import tta


class Predictor(object):
    def __init__(self, model, cnf, weights_from, prediction_iterator):
        self.model = model
        self.cnf = cnf
        self.weights_from = weights_from
        self.prediction_iterator = prediction_iterator

        end_points_predict = model(is_training=False, reuse=None)
        self.inputs = end_points_predict['inputs']
        self.predictions = end_points_predict['predictions']

    def predict(self, X):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.weights_from)
            return self._real_predict(X, sess)

    def _real_predict(self, X, sess, xform=None):
        tic = time.time()
        print('Making %d predictions' % len(X))
        data_predictions = []
        for X, y in self.prediction_iterator(X, xform=xform):
            predictions_e = sess.run(self.predictions, feed_dict={self.inputs: X})
            data_predictions.append(predictions_e)
        data_predictions = np.vstack(data_predictions)
        print('took %6.1f seconds' % (time.time() - tic))
        return data_predictions


class QuasiPredictor(object):
    def __init__(self, model, cnf, weights_from, prediction_iterator, number_of_transforms):
        self.number_of_transforms = number_of_transforms
        self.cnf = cnf
        self.weights_from = weights_from
        self.prediction_iterator = prediction_iterator
        self.predictor = Predictor(model, cnf, weights_from, prediction_iterator)

    def predict(self, X):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.weights_from)
            return self._real_predict(X, sess)

    def _real_predict(self, X, sess, **unused):
        tfs, color_vecs = tta.build_quasirandom_transforms(self.number_of_transforms, color_sigma=self.cnf['sigma'],
                                                           **self.cnf['aug_params'])
        multiple_predictions = []
        for i, (xform, color_vec) in enumerate(zip(tfs, color_vecs), start=1):
            print('Quasi-random tta iteration: %d' % i)
            standardizer = self.prediction_iterator.standardizer
            if standardizer is not None:
                standardizer.update(color_vec=color_vec)
            predictions = self.predictor._real_predict(X, sess, xform=xform)
            multiple_predictions.append(predictions)
        return np.mean(multiple_predictions, axis=0)
