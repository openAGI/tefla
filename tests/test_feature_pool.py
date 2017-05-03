import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal

from tefla.core.layers import feature_max_pool_1d


def test_max_pool_1d_stride_2():
    x = np.array([[1, 3, 5, 4, 9, 2],
                  [4, 3, 5, 7, 9, 6]])

    pool_stride = 2
    expected_pooled_x = np.array([[3, 5, 9],
                                  [4, 7, 9]])

    pooled_x_s = feature_max_pool_1d(x, pool_stride)
    with tf.Session() as sess:
        pooled_x = sess.run(pooled_x_s)

    assert_array_equal(expected_pooled_x, pooled_x)


def test_max_pool_1d_stride_3():
    x = np.array([[1, 3, 5, 4, 9, 2],
                  [4, 3, 5, 7, 9, 6]])

    pool_stride = 3
    expected_pooled_x = np.array([[5, 9],
                                  [5, 9]])

    pooled_x_s = feature_max_pool_1d(x, pool_stride)
    with tf.Session() as sess:
        pooled_x = sess.run(pooled_x_s)

    assert_array_equal(expected_pooled_x, pooled_x)


def test_max_pool_1d_stride_2_batch_1():
    x = np.array([[1, 3, 5, 4, 9, 2]])

    pool_stride = 2
    expected_pooled_x = np.array([[3, 5, 9]])

    pooled_x_s = feature_max_pool_1d(x, pool_stride)
    with tf.Session() as sess:
        pooled_x = sess.run(pooled_x_s)

    assert_array_equal(expected_pooled_x, pooled_x)


def test_max_pool_1d_stride_2_batch_3():
    x = np.array([[1, 3, 5, 4, 9, 2],
                  [5, 3, 6, 8, 4, 2],
                  [4, 3, 5, 7, 9, 6]])

    pool_stride = 2
    expected_pooled_x = np.array([[3, 5, 9],
                                  [5, 8, 4],
                                  [4, 7, 9]])

    pooled_x_s = feature_max_pool_1d(x, pool_stride)
    with tf.Session() as sess:
        pooled_x = sess.run(pooled_x_s)

    assert_array_equal(expected_pooled_x, pooled_x)

if __name__ == '__main__':
    pytest.main([__file__])
