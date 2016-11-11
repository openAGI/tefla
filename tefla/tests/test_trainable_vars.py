import pytest
import tensorflow as tf

from tefla.core.layers import fully_connected


@pytest.fixture(autouse=True)
def clean_graph():
    tf.reset_default_graph()


def test_trainable_true():
    x = tf.placeholder(tf.float32, [1, 10, 10, 3])
    x = fully_connected(x, 15, is_training=True, reuse=False, name='fc1', trainable=True)
    trainable_vars = [v.name for v in tf.trainable_variables()]
    assert 'fc1/W:0' in trainable_vars
    assert 'fc1/b:0' in trainable_vars


def test_trainable_false():
    x = tf.placeholder(tf.float32, [1, 10, 10, 3])
    x = fully_connected(x, 15, is_training=True, reuse=False, name='fc1', trainable=False)
    trainable_vars = [v.name for v in tf.trainable_variables()]
    assert 'fc1/W:0' not in trainable_vars
    assert 'fc1/b:0' not in trainable_vars


if __name__ == '__main__':
    pytest.main([__file__])
