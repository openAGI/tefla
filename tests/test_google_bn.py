import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_array_almost_equal
from tensorflow.python.ops import control_flow_ops

from tefla.core.layers import batch_norm_tf as batch_norm


@pytest.fixture(autouse=True)
def _reset_graph():
    tf.reset_default_graph()


def test_eval_moving_vars():
    height, width = 3, 3
    with tf.Session() as sess:
        image_shape = (10, height, width, 3)
        image_values = np.random.rand(*image_shape)
        images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
        output = batch_norm(images, is_training=False, reuse=None, decay=0.1, name='BatchNorm')
        assert len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) == 0
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        moving_mean = tf.contrib.framework.get_variables('BatchNorm/moving_mean')[0]
        moving_variance = tf.contrib.framework.get_variables('BatchNorm/moving_variance')[0]
        mean, variance = sess.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        assert_array_almost_equal(mean, [0] * 3)
        assert_array_almost_equal(variance, [1] * 3)
        # Simulate assigment from saver restore.
        expected_moving_mean = [0.1] * 3  # could be any number
        expected_moving_var = [0.5] * 3
        init_assigns = [tf.assign(moving_mean, expected_moving_mean), tf.assign(moving_variance, expected_moving_var)]
        sess.run(init_assigns)
        for _ in range(10):
            sess.run([output], {images: np.random.rand(*image_shape)})
        mean = moving_mean.eval()
        variance = moving_variance.eval()
        # Although we feed different images, the moving_mean and moving_variance
        # shouldn't change.
        assert_array_almost_equal(mean, expected_moving_mean)
        assert_array_almost_equal(variance, expected_moving_var)


def test_forced_update_moving_vars_and_output():
    with tf.Session() as sess:
        height, width = 3, 3
        image_shape = (10, height, width, 3)
        image_values = np.random.rand(*image_shape)
        images_mean = np.mean(image_values, axis=(0, 1, 2))
        images_var = np.var(image_values, axis=(0, 1, 2))
        images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
        decay = 0.8
        epsilon = 1e-5
        output_s = batch_norm(images, is_training=True, reuse=None, decay=decay, epsilon=epsilon,
                              updates_collections=None, name='BatchNorm')
        sess.run(tf.global_variables_initializer())

        moving_mean = tf.contrib.framework.get_variables('BatchNorm/moving_mean')[0]
        moving_variance = tf.contrib.framework.get_variables('BatchNorm/moving_variance')[0]
        mean, variance = sess.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        assert_array_almost_equal(mean, [0] * 3)
        assert_array_almost_equal(variance, [1] * 3)

        # Feed in the same batch of images 10 times
        n_times = 10
        expected_mean = np.array([0.] * 3)
        expected_var = np.array([1.] * 3)
        expected_output = (image_values - images_mean) / np.sqrt(images_var + epsilon)
        for _ in xrange(n_times):
            output = sess.run(output_s)
            mean, variance = sess.run([moving_mean, moving_variance])
            expected_mean = expected_mean * decay + images_mean * (1 - decay)
            expected_var = expected_var * decay + images_var * (1 - decay)
            assert_array_almost_equal(output, expected_output, decimal=4)
            assert_array_almost_equal(mean, expected_mean, decimal=4)
            assert_array_almost_equal(variance, expected_var, decimal=4)


def test_delayed_update_moving_vars_and_output():
    with tf.Session() as sess:
        height, width = 3, 3
        image_shape = (10, height, width, 3)
        image_values = np.random.rand(*image_shape)
        images_mean = np.mean(image_values, axis=(0, 1, 2))
        images_var = np.var(image_values, axis=(0, 1, 2))
        images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
        decay = 0.8
        epsilon = 1e-5
        output_s = batch_norm(images, is_training=True, reuse=None, decay=decay, epsilon=epsilon,
                              updates_collections=tf.GraphKeys.UPDATE_OPS, name='BatchNorm')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # updates_ops are added to UPDATE_OPS collection.
        assert len(update_ops) == 2
        with tf.control_dependencies(update_ops):
            barrier = tf.no_op(name='barrier')
        output_s = control_flow_ops.with_dependencies([barrier], output_s)
        sess.run(tf.global_variables_initializer())

        moving_mean = tf.contrib.framework.get_variables('BatchNorm/moving_mean')[0]
        moving_variance = tf.contrib.framework.get_variables('BatchNorm/moving_variance')[0]
        mean, variance = sess.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        assert_array_almost_equal(mean, [0] * 3)
        assert_array_almost_equal(variance, [1] * 3)

        # Feed in the same batch of images 10 times
        n_times = 10
        expected_mean = np.array([0.] * 3)
        expected_var = np.array([1.] * 3)
        expected_output = (image_values - images_mean) / np.sqrt(images_var + epsilon)
        for _ in range(n_times):
            output = sess.run(output_s)
            mean, variance = sess.run([moving_mean, moving_variance])
            expected_mean = expected_mean * decay + images_mean * (1 - decay)
            expected_var = expected_var * decay + images_var * (1 - decay)
            assert_array_almost_equal(output, expected_output, decimal=4)
            assert_array_almost_equal(mean, expected_mean, decimal=4)
            assert_array_almost_equal(variance, expected_var, decimal=4)


def test_delayed_update_moving_vars():
    with tf.Session() as sess:
        height, width = 3, 3
        image_shape = (10, height, width, 3)
        image_values = np.random.rand(*image_shape)
        expected_mean = np.mean(image_values, axis=(0, 1, 2))
        expected_var = np.var(image_values, axis=(0, 1, 2))
        images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
        decay = 0.1
        epsilon = 1e-5
        output = batch_norm(images, is_training=True, reuse=None, decay=decay, epsilon=epsilon,
                            updates_collections=tf.GraphKeys.UPDATE_OPS, name='BatchNorm')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # updates_ops are added to UPDATE_OPS collection.
        assert len(update_ops) == 2
        with tf.control_dependencies(update_ops):
            barrier = tf.no_op(name='barrier')
        output = control_flow_ops.with_dependencies([barrier], output)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        moving_mean = tf.contrib.framework.get_variables('BatchNorm/moving_mean')[0]
        moving_variance = tf.contrib.framework.get_variables('BatchNorm/moving_variance')[0]
        mean, variance = sess.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        assert_array_almost_equal(mean, [0] * 3)
        assert_array_almost_equal(variance, [1] * 3)
        for _ in range(10):
            sess.run([output])
        mean = moving_mean.eval()
        variance = moving_variance.eval()
        # After 10 updates with decay 0.1 moving_mean == expected_mean and
        # moving_variance == expected_var.
        assert_array_almost_equal(mean, expected_mean, decimal=4)
        assert_array_almost_equal(variance, expected_var, decimal=4)


if __name__ == '__main__':
    pytest.main([__file__])
