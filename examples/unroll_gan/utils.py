import numpy as np
import tensorflow as tf
import numbers


class SessionWrap(object):

    def __init__(self, session=None):
        self.session = session
        if session is None:
            self.release_session = True
        else:
            self.release_session = False

    def __enter__(self):
        if self.session is None:
            self.session = tf.Session()
        return self.session

    def __exit__(self, *args):
        if self.release_session:
            self.session.close()


def dense(input, output_dim, name=None, stddev=1., reuse=False, normalized=False, params=None):
    norm = tf.contrib.layers.variance_scaling_initializer(stddev)
    const = tf.constant_initializer(0.0)
    if params is not None:
        w, b = params
    else:
        with tf.variable_scope(name or 'linear') as scope:
            if reuse:
                scope.reuse_variables()
            w = tf.get_variable(
                'w', [input.get_shape()[1], output_dim], initializer=norm, dtype=tf.float32)
            b = tf.get_variable('b', [output_dim],
                                initializer=const, dtype=tf.float32)
    if normalized:
        w_n = w / tf.reduce_sum(tf.square(w), 1, keep_dims=True)
    else:
        w_n = w
    return tf.matmul(input, w_n) + b
