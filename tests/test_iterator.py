import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from tefla.da import iterator


def no_op_preprocessor(img):
    return img


def times_two_preprocessor(img):
    return img * 2


def test_batch_iter():
    data = np.arange(36).reshape(12, 3)
    bi = iterator.BatchIterator(4, False)
    data2 = np.vstack([items[0] for items in bi(data)])
    assert_array_equal(data, data2)


def test_batch_iter_with_shuffle():
    data = np.arange(36).reshape(12, 3)
    bi = iterator.BatchIterator(4, True)
    data2 = np.vstack([items[0] for items in bi(data)])
    assert_equal(np.any(np.not_equal(data, data2)), True)
    assert_array_equal(data, np.sort(data2, axis=0))


def test_queued_iter():
    data = np.arange(36).reshape(12, 3)
    bi = iterator.QueuedIterator(4, False)
    data2 = np.vstack([items[0] for items in bi(data)])
    assert_array_equal(data, data2)


def test_da_iter():
    data = np.arange(12 * 3 * 4 * 4).reshape(12, 3, 4, 4)
    dai = iterator.DAIterator(4, False, no_op_preprocessor, (4, 4), is_training=False)
    data2 = np.vstack([items[0] for items in dai(data)])
    assert_array_equal(data.transpose(0, 2, 3, 1), data2)


def test_da_iter_with_pre_processing():
    data = np.arange(12 * 3 * 4 * 4).reshape(12, 3, 4, 4)
    dai = iterator.DAIterator(4, False, times_two_preprocessor, (4, 4), is_training=False)
    data2 = np.vstack([items[0] for items in dai(data)])
    assert_array_equal(data.transpose(0, 2, 3, 1) * 2, data2)


def test_queued_da_iter():
    data = np.arange(12 * 3 * 4 * 4).reshape(12, 3, 4, 4)
    dai = iterator.QueuedDAIterator(4, False, no_op_preprocessor, (4, 4), is_training=False)
    data2 = np.vstack([items[0] for items in dai(data)])
    assert_array_equal(data.transpose(0, 2, 3, 1), data2)


def test_parallel_da_iter():
    data = np.arange(12 * 3 * 4 * 4).reshape(12, 3, 4, 4)
    dai = iterator.ParallelDAIterator(4, False, no_op_preprocessor, (4, 4), is_training=False)
    data2 = np.vstack([items[0] for items in dai(data)])
    assert_array_equal(data.transpose(0, 2, 3, 1), data2)


def test_parallel_da_iter_with_pre_processing():
    data = np.arange(12 * 3 * 4 * 4).reshape(12, 3, 4, 4)
    dai = iterator.ParallelDAIterator(4, False, times_two_preprocessor, (4, 4), is_training=False)
    data2 = np.vstack([items[0] for items in dai(data)])
    assert_array_equal(data.transpose(0, 2, 3, 1) * 2, data2)


def test_balancing_da_iter():
    data = np.arange(12 * 3 * 4 * 4).reshape(12, 3, 4, 4)
    dai = iterator.BalancingDAIterator(4, False, no_op_preprocessor, (4, 4), False, np.array([1., 1.]),
                                       np.array([1., 1.]), 1.)
    data2 = np.vstack([items[0] for items in dai(data)])
    assert_array_equal(data.transpose(0, 2, 3, 1), data2)


if __name__ == '__main__':
    pytest.main([__file__])
