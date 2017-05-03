import tensorflow as tf
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from tefla.core import metrics


@pytest.fixture(autouse=True)
def _reset_graph():
    tf.reset_default_graph()


def test_kappav2_op():
    kappav2 = metrics.KappaV2()
    labels = tf.placeholder(shape=(32,), name='labels', dtype=tf.int32)
    predictions = tf.placeholder(shape=(32,), name='predictions', dtype=tf.int32)
    kappa_metric = kappav2.metric(predictions, labels, num_classes=10, batch_size=32)
    label_v = np.random.randint(low=0, high=9, size=(32, ))
    pred_v = np.random.randint(low=0, high=9, size=(32, ))
    kappa = metrics.Kappa()
    kappa_metric_ = kappa.metric(pred_v, label_v, 10)
    with tf.Session() as sess:
        _kappa_metric = sess.run(kappa_metric, feed_dict={labels: label_v, predictions: pred_v})

    assert_array_almost_equal(_kappa_metric, kappa_metric_)


if __name__ == '__main__':
    pytest.main([__file__])
