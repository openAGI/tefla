from functools import partial
import numpy as np
import tensorflow as tf
from tefla.core import losses
from tefla.utils import util


def MaximumMeanDiscrepancySlow(x, y, sigmas):
    num_samples = x.get_shape().as_list()[0]

    def AverageGaussianKernel(x, y, sigmas):
        result = 0
        for sigma in sigmas:
            dist = tf.reduce_sum(tf.square(x - y))
            result += tf.exp((-1.0 / (2.0 * sigma)) * dist)
        return result / num_samples**2

    total = 0

    for i in range(num_samples):
        for j in range(num_samples):
            total += AverageGaussianKernel(x[i, :], x[j, :], sigmas)
            total += AverageGaussianKernel(y[i, :], y[j, :], sigmas)
            total += -2 * AverageGaussianKernel(x[i, :], y[j, :], sigmas)

    return total


class LogQuaternionLossTest(tf.test.TestCase):

    def test_log_quaternion_loss_batch(self):
        with self.test_session():
            predictions = tf.random_uniform((10, 4), seed=1)
            predictions = tf.nn.l2_normalize(predictions, 1)
            labels = tf.random_uniform((10, 4), seed=1)
            labels = tf.nn.l2_normalize(labels, 1)
            params = {'batch_size': 10, 'use_logging': False}
            x = losses.log_quaternion_loss_batch(predictions, labels)
            self.assertTrue(((10,) == tf.shape(x).eval()).all())


class MaximumMeanDiscrepancyTest(tf.test.TestCase):

    def test_mmd_name(self):
        with self.test_session():
            x = tf.random_uniform((2, 3), seed=1)
            kernel = partial(util.gaussian_kernel_matrix,
                             sigmas=tf.constant([1.]))
            loss = losses.maximum_mean_discrepancy(x, x, kernel)

    def test_mmd_is_zero_when_inputs_are_same(self):
        with self.test_session():
            x = tf.random_uniform((2, 3), seed=1)
            kernel = partial(util.gaussian_kernel_matrix,
                             sigmas=tf.constant([1.]))
            self.assertEquals(
                0, losses.maximum_mean_discrepancy(x, x, kernel).eval())

    def test_fast_mmd_is_similar_to_slow_mmd(self):
        with self.test_session():
            x = tf.constant(np.random.normal(size=(2, 3)), tf.float32)
            y = tf.constant(np.random.rand(2, 3), tf.float32)

            cost_old = MaximumMeanDiscrepancySlow(x, y, [1.]).eval()
            kernel = partial(util.gaussian_kernel_matrix,
                             sigmas=tf.constant([1.]))
            cost_new = losses.maximum_mean_discrepancy(x, y, kernel).eval()

            self.assertAlmostEqual(cost_old, cost_new, delta=1e-5)

    def test_multiple_sigmas(self):
        with self.test_session():
            x = tf.constant(np.random.normal(size=(2, 3)), tf.float32)
            y = tf.constant(np.random.rand(2, 3), tf.float32)

            sigmas = tf.constant([2., 5., 10, 20, 30])
            kernel = partial(util.gaussian_kernel_matrix, sigmas=sigmas)
            cost_old = MaximumMeanDiscrepancySlow(
                x, y, [2., 5., 10, 20, 30]).eval()
            cost_new = losses.maximum_mean_discrepancy(
                x, y, kernel=kernel).eval()

            self.assertAlmostEqual(cost_old, cost_new, delta=1e-5)

    def test_mmd_is_zero_when_distributions_are_same(self):

        with self.test_session():
            x = tf.random_uniform((1000, 10), seed=1)
            y = tf.random_uniform((1000, 10), seed=3)

            kernel = partial(util.gaussian_kernel_matrix,
                             sigmas=tf.constant([100.]))
            loss = losses.maximum_mean_discrepancy(x, y, kernel=kernel).eval()

            self.assertAlmostEqual(0, loss, delta=1e-4)


class LogLossTest(tf.test.TestCase):

    def setUp(self):
        predictions = np.asarray([.9, .2, .2, .8, .4, .6]).reshape((2, 3))
        labels = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]).reshape((2, 3))

        self._np_predictions = predictions
        self._np_labels = labels

        epsilon = 1e-7
        self._expected_losses = np.multiply(
            labels, np.log(predictions + epsilon)) + np.multiply(
                1 - labels, np.log(1 - predictions + epsilon))

        self._predictions = tf.constant(predictions)
        self._labels = tf.constant(labels)

    def testValueErrorThrownWhenWeightIsNone(self):
        with self.test_session():
            with self.assertRaises(ValueError):
                losses.log_loss_tf(self._labels, self._labels, weights=None)

    def testAllCorrectNoLossWeight(self):
        loss = losses.log_loss_tf(self._labels, self._labels)
        with self.test_session():
            self.assertAlmostEqual(0.0, loss.eval(), 3)

    def testNonZeroLoss(self):
        loss = losses.log_loss_tf(self._predictions, self._labels)
        with self.test_session():
            self.assertAlmostEqual(-np.sum(self._expected_losses) /
                                   6.0, loss.eval(), 3)

    def testNonZeroLossWithWeight(self):
        weights = 2.3
        loss = losses.log_loss_tf(
            self._predictions, self._labels, weights=weights)
        with self.test_session():
            self.assertAlmostEqual(
                weights * -np.sum(self._expected_losses) / 6.0, loss.eval(), 3)


if __name__ == '__main__':
    tf.test.main()
