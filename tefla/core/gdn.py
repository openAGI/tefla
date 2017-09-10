import tensorflow as tf
from tensorflow.python.ops import nn


def gdn(inputs,
        reuse,
        inverse=False,
        beta_min=1e-6,
        gamma_init=.1,
        reparam_offset=2 ** -18,
        trainable=True,
        name='gdn',
        dtype=tf.float32,
        **kwargs):
    """Generalized divisive normalization layer.
    Based on the papers:
      "Density Modeling of Images using a Generalized Normalization
      Transformation"
      Johannes Balle, Valero Laparra, Eero P. Simoncelli
      https://arxiv.org/abs/1511.06281
      "End-to-end Optimized Image Compression"
      Johannes Balle, Valero Laparra, Eero P. Simoncelli
      https://arxiv.org/abs/1611.01704
    Implements an activation function that is essentially a multivariate
    generalization of a particular sigmoid-type function:
    ```
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    ```
    where `i` and `j` run over channels. This implementation never sums across
    spatial dimensions. It is similar to local response normalization, but much
    more flexible, as `beta` and `gamma` are trainable parameters.
    Arguments:
      inverse: If `False` (default), compute GDN response. If `True`, compute IGDN
        response (one step of fixed point iteration to invert GDN; the division
        is replaced by multiplication).
      beta_min: Lower bound for beta, to prevent numerical error from causing
        square root of zero or negative values.
      gamma_init: The gamma matrix will be initialized as the identity matrix
        multiplied with this value. If set to zero, the layer is effectively
        initialized to the identity operation, since beta is initialized as one.
        A good default setting is somewhere between 0 and 0.5.
      reparam_offset: Offset added to the reparameterization of beta and gamma.
        The reparameterization of beta and gamma as their square roots lets the
        training slow down when their values are close to zero, which is desirable
        as small values in the denominator can lead to a situation where gradient
        noise on beta/gamma leads to extreme amounts of noise in the GDN
        activations. However, without the offset, we would get zero gradients if
        any elements of beta or gamma were exactly zero, and thus the training
        could get stuck. To prevent this, we add this small constant. The default
        value was empirically determined as a good starting point. Making it
        bigger potentially leads to more gradient noise on the activations, making
        it too small may lead to numerical precision issues.
      data_format: Format of input tensor. Currently supports `'channels_first'`
        and `'channels_last'`.
      trainable: Boolean, if `True`, also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require `reuse=True` in such
        cases.
    Properties:
      inverse: Boolean, whether GDN is computed (`True`) or IGDN (`False`).
      beta: The beta parameter as defined above (1D `Tensor`).
      gamma: The gamma parameter as defined above (2D `Tensor`).
    """

    input_shape = inputs.shape
    num_channels = input_shape[-1].value
    if num_channels is None:
        raise ValueError('The channel dimension of the inputs to `GDN` '
                         'must be defined.')
    input_rank = input_shape.ndims

    pedestal = tf.constant(
        reparam_offset ** 2, dtype=dtype)
    beta_bound = tf.constant(
        (beta_min + reparam_offset ** 2) ** .5, dtype=dtype)
    gamma_bound = tf.constant(
        reparam_offset, dtype=dtype)

    def beta_initializer(shape, dtype=None, partition_info=None):
        del partition_info  # unused
        return tf.sqrt(tf.ones(shape, dtype=dtype) + pedestal)

    def gamma_initializer(shape, dtype=None, partition_info=None):
        del partition_info  # unused
        assert len(shape) == 2
        assert shape[0] == shape[1]
        eye = tf.eye(shape[0], dtype=dtype)
        return tf.sqrt(gamma_init * eye + pedestal)
    with tf.variable_scope(name, reuse=reuse):
        beta = tf.get_variable('reparam_beta',
                               shape=[num_channels],
                               initializer=beta_initializer,
                               dtype=dtype,
                               trainable=trainable)
    beta = _lower_bound(beta, beta_bound)
    beta = tf.square(beta) - pedestal
    with tf.variable_scope(name, reuse=reuse):
        gamma = tf.get_variable('reparam_gamma',
                                shape=[num_channels, num_channels],
                                initializer=gamma_initializer,
                                dtype=dtype,
                                trainable=trainable)
    gamma = _lower_bound(gamma, gamma_bound)
    gamma = tf.square(gamma) - pedestal

    inputs = tf.convert_to_tensor(inputs, dtype=dtype)
    ndim = input_rank

    shape = gamma.get_shape().as_list()
    gamma = tf.reshape(gamma, (ndim - 2) * [1] + shape)

    norm_pool = nn.convolution(tf.square(inputs), gamma, 'VALID')
    norm_pool = nn.bias_add(norm_pool, beta, data_format='NHWC')
    norm_pool = tf.sqrt(norm_pool)

    if inverse:
        outputs = inputs * norm_pool
    else:
        outputs = inputs / norm_pool
    outputs.set_shape(inputs.get_shape())

    return outputs


def _lower_bound(inputs, bound, name=None):
    """Same as tf.maximum, but with helpful gradient for inputs < bound.
    The gradient is overwritten so that it is passed through if the input is not
    hitting the bound. If it is, only gradients that push `inputs` higher than
    the bound are passed through. No gradients are passed through to the bound.
    Args:
      inputs: input tensor
      bound: lower bound for the input tensor
      name: name for this op
    Returns:
      tf.maximum(inputs, bound)
    """
    with tf.name_scope(name, 'GDNLowerBound', [inputs, bound]) as scope:
        inputs = tf.convert_to_tensor(inputs, name='inputs')
        bound = tf.convert_to_tensor(bound, name='bound')
        with tf.get_default_graph().gradient_override_map(
                {'Maximum': 'GDNLowerBound'}):
            return tf.maximum(inputs, bound, name=scope)


@tf.RegisterGradient("GDNLowerBound")
def _lower_bound_grad(op, grad):
    """Gradient for `_lower_bound`.
    Args:
      op: the tensorflow op for which to calculate a gradient
      grad: gradient with respect to the output of the op
    Returns:
      gradients with respect to the inputs of the op
    """
    inputs = op.inputs[0]
    bound = op.inputs[1]
    pass_through_if = tf.logical_or(inputs >= bound, grad < 0)
    return [tf.cast(pass_through_if, grad.dtype) * grad, None]
