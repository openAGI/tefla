from __future__ import division, print_function, absolute_import
import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

# For external users
from tensorflow.contrib.layers import xavier_initializer


def he_normal(seed=None, scale=1.0, dtype=tf.float32):
    """
    He Normal initializer
    Kaiming He et al. (2015): Delving deep into rectifiers: Surpassing human-level
    performance on imagenet classification. arXiv preprint arXiv:1502.01852.

    Args:
        scale: float
               Scaling factor for the weights. Set this to ``1.0`` for linear and
               sigmoid units, to ``sqrt(2)`` for rectified linear units, and
               to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
               leakiness ``alpha``. Other transfer functions may need different factors.
    """
    return variance_scaling_initializer(factor=2.0 * scale, mode='FAN_IN',
                                        uniform=False, seed=seed, dtype=dtype)


def he_uniform(seed=None, scale=1.0, dtype=tf.float32):
    """
    He Uniform initializer

    Args:
        scale: float
               Scaling factor for the weights. Set this to ``1.0`` for linear and
               sigmoid units, to ``sqrt(2)`` for rectified linear units, and
               to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
               leakiness ``alpha``. Other transfer functions may need different factors.
    """
    return variance_scaling_initializer(factor=2.0 * scale, mode='FAN_IN',
                                        uniform=True, seed=seed, dtype=dtype)


def random_normal(seed=None, mean=0.0, stddev=1.0, dtype=tf.float32, name=None):
    """Random Normal initializer

    Args:
        mean: a `float`
        stddev: a `float`
    """
    return variance_scaling_initializer_v2(factor=2.0, mode='FAN_IN', uniform=False,
                                           seed=None, dtype=tf.float32, mean=mean, stddev=stddev, normal_type='random_normal', name=name)


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


def variance_scaling_initializer_v2(factor=2.0, mode='FAN_IN', uniform=False,
                                    seed=None, dtype=tf.float32, mean=0.0, stddev=1.0, normal_type=None, name=None):
    """Returns an initializer that generates tensors without scaling variance.
    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. This initializer use the following formula:
    ```python
      if mode='FAN_IN': # Count only number of input connections.
        n = fan_in
      elif mode='FAN_OUT': # Count only number of output connections.
        n = fan_out
      elif mode='FAN_AVG': # Average number of inputs and output connections.
        n = (fan_in + fan_out)/2.0
        truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
    ```
    * To get [Delving Deep into Rectifiers](
       http://arxiv.org/pdf/1502.01852v1.pdf), use (Default):<br/>
      `factor=2.0 mode='FAN_IN' uniform=False`
    * To get [Convolutional Architecture for Fast Feature Embedding](
       http://arxiv.org/abs/1408.5093), use:<br/>
      `factor=1.0 mode='FAN_IN' uniform=True`
    * To get [Understanding the difficulty of training deep feedforward neural
      networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
      use:<br/>
      `factor=1.0 mode='FAN_AVG' uniform=True.`
    * To get `xavier_initializer` use either:<br/>
      `factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
      `factor=1.0 mode='FAN_AVG' uniform=False`.
    Args:
      factor: Float.  A multiplicative factor.
      mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
      uniform: Whether to use uniform or normal distributed random initialization.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      dtype: The data type. Only floating point types are supported.
    Returns:
      An initializer that generates tensors with unit variance.
    Raises:
      ValueError: if `dtype` is not a floating point type.
      TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
    """
    if not dtype.is_floating:
        raise TypeError(
            'Cannot create initializer for non-floating point type.')
    if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
        raise TypeError('Unknow mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)
    if not uniform:
        if normal_type is None:
            normal_type = 'truncated_normal'
        else:
            normal_type = 'random_normal'

    def _initializer(shape, dtype=dtype, partition_info=None):
        """Initializer function."""
        if not dtype.is_floating:
            raise TypeError(
                'Cannot create initializer for non-floating point type.')
        # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
        # This is the right thing for matrix multiply and convolutions.
        if shape:
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == 'FAN_IN':
            # Count only number of input connections.
            n = fan_in
        elif mode == 'FAN_OUT':
            # Count only number of output connections.
            n = fan_out
        elif mode == 'FAN_AVG':
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        if uniform:
            # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            limit = math.sqrt(3.0 * factor / n)
            return tf.random_uniform(shape, -limit, limit,
                                     dtype, seed=seed)
        if not uniform and normal_type == 'truncated_normal':
            # To get stddev = math.sqrt(factor / n) need to adjust for
            # truncated.
            trunc_stddev = math.sqrt(1.3 * factor / n)
            return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                       seed=seed)
        else:
            return tf.random_normal(shape, mean=mean, stddev=stddev, dtype=tf.float32, seed=seed, name=name)

    return _initializer


def bilinear(f_shape):
    """Bilinear initialization for up sampling operation

    Args:
        f_shape: shape of the variable

    Returns:
        bilinear initializer

    """
    width = f_shape[0]
    heigh = f_shape[1]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return init


def random_orthonormal_initializer(shape, dtype=tf.float32, partition_info=None):
    """Variable initializer that produces a random orthonormal matrix

    Args:
        shape: shape of the variable

    Returns:
        random_orthogonal_matrix for initialization.
    """
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u
