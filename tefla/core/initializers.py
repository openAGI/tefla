from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

# For external users
from tensorflow.contrib.layers import xavier_initializer


def he_normal(seed=None, scale=1.0, dtype=tf.float32):
    return variance_scaling_initializer(factor=2.0 * scale, mode='FAN_IN',
                                        uniform=False, seed=seed, dtype=dtype)


def he_uniform(seed=None, scale=1.0, dtype=tf.float32):
    return variance_scaling_initializer(factor=2.0 * scale, mode='FAN_IN',
                                        uniform=True, seed=seed, dtype=dtype)


# Code borrowed from Lasagne https://github.com/Lasagne/Lasagne under MIT license
def orthogonal(gain=1.0, dtype=np.float32):
    def _initializer(shape, dtype=dtype):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return np.array(gain * q, dtype=np.float32)

    return _initializer
