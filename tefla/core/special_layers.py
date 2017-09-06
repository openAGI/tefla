import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import pydensecrf.densecrf as dcrf
from .layers import conv2d, depthwise_conv2d, avg_pool_2d, max_pool, relu, crelu, batch_norm_tf as batch_norm
from ..utils import util
from . import initializers as initz


def spatialtransformer(U, theta, batch_size=64, downsample_factor=1.0, num_transform=1, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    It's based on lasagne implementation in [2]_, modified by Mrinal Haloi

    Args:
        U: float
            The output of a convolutional net should have the
            shape [batch_size, height, width, num_channels].
        theta: float
            The output of the localisation network should be [batch_size, num_transform, 6] or [batch_size, 6] if num_transform=1
            ```python
                `theta`` to :
                    identity = np.array([[1., 0., 0.],
                                     [0., 1., 0.]])
                    identity = identity.flatten()
                    theta = tf.Variable(initial_value=identity)
            ```
        downsample_factor: a float, determines output shape, downsample input shape by downsample_factor

    Returns:
        spatial transformed output of the network

    References
    .. [1]  "Spatial Transformer Networks", Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py


    """
    with tf.variable_scope(name):
        if num_transform > 1 and len(theta.get_shape().as_list()) == 3:
            _, num_transforms = map(int, theta.get_shape().as_list()[:2])
            indices = [[i] * num_transforms for i in range(batch_size)]
            U = tf.gather(U, tf.reshape(indices, [-1]))

        input_shape = U.get_shape().as_list()
        num_channels = input_shape[3]
        theta = tf.reshape(theta, (-1, 2, 3))
        theta = tf.cast(theta, tf.float32)
        if not isinstance(downsample_factor, float):
            downsample_factor = tf.cast(downsample_factor, tf.float32)

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = tf.cast(input_shape[1] / downsample_factor, tf.int32)
        out_width = tf.cast(input_shape[2] / downsample_factor, tf.int32)
        grid = _meshgrid(out_height, out_width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([batch_size]))
        grid = tf.reshape(grid, tf.stack([batch_size, 3, -1]))

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate(
            U, x_s_flat, y_s_flat, batch_size, downsample_factor)

        output = tf.reshape(input_transformed, tf.stack(
            [batch_size, out_height, out_width, num_channels]))
        return output


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(tf.expand_dims(
            tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, tf.int32)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])


def _interpolate(im, x, y, batch_size, downsample_factor):
    with tf.variable_scope('_interpolate'):
        input_shape = im.get_shape().as_list()
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        out_height = tf.cast(height / downsample_factor, tf.int32)
        out_width = tf.cast(width / downsample_factor, tf.int32)
        zero = tf.zeros([], dtype=tf.int32)
        max_y = tf.cast(height - 1, tf.int32)
        max_x = tf.cast(width - 1, tf.int32)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(tf.range(batch_size) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, tf.float32)
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output


def _meshgrid(height, width):
    with tf.variable_scope('_meshgrid'):
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])), tf.transpose(
            tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(
            tf.linspace(-1.0, 1.0, height), 1), tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
        return grid


def subsample(inputs, factor, name=None):
    """Subsamples the input along the spatial dimensions.

    Args:
        inputs: A `Tensor` of size [batch, height_in, width_in, channels].
        factor: The subsampling factor.
        name: Optional variable_scope.

    Returns:
        output: A `Tensor` of size [batch, height_out, width_out, channels] with the
            input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return max_pool(inputs, filter_size=(1, 1), stride=(factor, factor), name=name)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, name=None, **kwargs):
    """Strided 2-D convolution with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.

    Note that

       net = conv2d_same(inputs, num_outputs, 3, stride=stride)

    is equivalent to

       net = conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
       net = subsample(net, factor=stride)

    whereas

       net = conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

    Args:
        inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
        num_outputs: An integer, the number of output filters.
        kernel_size: An int with the kernel_size of the filters.
        stride: An integer, the output stride.
        rate: An integer, rate for atrous convolution.
        name: name.

    Returns:
        output: A 4-D tensor of size [batch, height_out, width_out, channels] with
            the convolution output.
    """
    if stride == 1:
        return conv2d(inputs, num_outputs, filter_size=(kernel_size, kernel_size), stride=(1, 1), dilaton=rate,
                      padding='SAME', name=name, **kwargs)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return conv2d(inputs, num_outputs, kernel_size, stride=stride,
                      dilation=rate, padding='VALID', name=name, **kwargs)


def bottleneck_v1(inputs, depth, depth_bottleneck, stride, rate=1, name=None, **kwargs):
    """Bottleneck residual unit variant with BN before convolutions.

    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
        inputs: A tensor of size [batch, height, width, channels].
        depth: The depth of the ResNet unit output.
        depth_bottleneck: The depth of the bottleneck layers.
        stride: The ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        outputs_collections: Collection to add the ResNet unit output.
        name: Optional variable_scope.

    Returns:
        The ResNet unit's output.
    """
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    with tf.variable_scope(name, 'bottleneck_v2', [inputs]):
        depth_in = util.last_dimension(inputs.get_shape(), min_rank=4)
        preact = batch_norm(inputs, activation_fn=tf.nn.relu,
                            name='preact', is_training=is_training, reuse=reuse)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = conv2d(preact, depth, is_training, reuse, filter_size=(1, 1), stride=(
                stride, stride), batch_norm=None, activation=None, name='shortcut')

        residual = conv2d(preact, depth_bottleneck, filter_size=(1, 1), stride=(1, 1),
                          name='conv1', **kwargs)
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               rate=rate, name='conv2', **kwargs)
        residual = conv2d(residual, depth, is_training, reuse, filter_size=(1, 1), stride=(1, 1),
                          batch_norm=None, activation=None, name='conv3')

        output = shortcut + residual

        return output


def bottleneck_v2(inputs, depth, depth_bottleneck, stride, rate=1, name=None, **kwargs):
    """Bottleneck residual unit variant with BN before convolutions.

    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
        inputs: A tensor of size [batch, height, width, channels].
        depth: The depth of the ResNet unit output.
        depth_bottleneck: The depth of the bottleneck layers.
        stride: The ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        outputs_collections: Collection to add the ResNet unit output.
        name: Optional variable_scope.

    Returns:
        The ResNet unit's output.
    """
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    with tf.variable_scope(name, 'bottleneck_v2', [inputs]):
        depth_in = util.last_dimension(inputs.get_shape(), min_rank=4)
        preact = batch_norm(inputs, activation_fn=tf.nn.relu,
                            name='preact', is_training=is_training, reuse=reuse)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = conv2d(preact, depth, is_training, reuse, filter_size=(1, 1), stride=(
                stride, stride), batch_norm=None, activation=None, name='shortcut')

        residual = conv2d(preact, depth_bottleneck, filter_size=(1, 1), stride=(1, 1),
                          name='conv1', **kwargs)
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               rate=rate, name='conv2', **kwargs)
        residual = conv2d(residual, depth, is_training, reuse, filter_size=(1, 1), stride=(1, 1),
                          batch_norm=None, activation=None, name='conv3')

        output = tf.nn.relu(shortcut + residual)

        return output


def memory_module(inputs, time, context, reuse, nwords, edim, mem_size, lindim, batch_size, nhop, share_list, regularizer=tf.nn.l2_loss, init_f=initz.random_normal(), trainable=True, name=None, **kwargs):
    with tf.variable_scope(name, 'memory_module', reuse=reuse):
        A_shape = B_shape = [nwords, edim]
        C_shape = [edim, edim]
        T_A_shape = T_B_shape = [mem_size, edim]
        A = tf.get_variable(
            name='A',
            shape=A_shape,
            initializer=init_f,
            regularizer=regularizer,
            trainable=trainable
        )
        B = tf.get_variable(
            name='B',
            shape=B_shape,
            initializer=init_f,
            regularizer=regularizer,
            trainable=trainable
        )
        C = tf.get_variable(
            name='C',
            shape=C_shape,
            initializer=init_f,
            regularizer=regularizer,
            trainable=trainable
        )
        T_A = tf.get_variable(
            name='T_A',
            shape=T_A_shape,
            initializer=init_f,
            regularizer=regularizer,
            trainable=trainable
        )
        T_B = tf.get_variable(
            name='T_B',
            shape=T_B_shape,
            initializer=init_f,
            regularizer=regularizer,
            trainable=trainable
        )

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(A, context)
        Ain_t = tf.nn.embedding_lookup(T_A, time)
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(B, context)
        Bin_t = tf.nn.embedding_lookup(T_B, time)
        Bin = tf.add(Bin_c, Bin_t)

        hid = []
        hid.append(inputs)

        for h in range(nhop):
            hid3dim = tf.reshape(hid[-1], [-1, 1, edim])
            Aout = tf.matmul(hid3dim, Ain, adj_y=True)
            Aout2dim = tf.reshape(Aout, [-1, mem_size])
            P = tf.nn.softmax(Aout2dim)

            probs3dim = tf.reshape(P, [-1, 1, mem_size])
            Bout = tf.matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, edim])

            Cout = tf.matmul(hid[-1], C)
            Dout = tf.add(Cout, Bout2dim)

            share_list[0].append(Cout)

            if lindim == edim:
                hid.append(Dout)
            elif lindim == 0:
                hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [batch_size, lindim])
                G = tf.slice(Dout, [0, lindim], [batch_size, edim - lindim])
                K = tf.nn.relu(G)
                hid.append(tf.concat([F, K], 1))
        return hid


def dense_crf(probs, img=None, n_classes=21, n_iters=10,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=2,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
        probs: class probabilities per pixel.
        img: if given, the pairwise bilateral potential on raw RGB values will be computed.
        n_iters: number of iterations of MAP inference.
        sxy_gaussian: standard deviations for the location component
            of the colour-independent term.
        compat_gaussian: label compatibilities for the colour-independent
            term (can be a number, a 1D array, or a 2D array).
        kernel_gaussian: kernel precision matrix for the colour-independent
            term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
        normalisation_gaussian: normalisation for the colour-independent term
            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
        sxy_bilateral: standard deviations for the location component of the colour-dependent term.
        compat_bilateral: label compatibilities for the colour-dependent
            term (can be a number, a 1D array, or a 2D array).
        srgb_bilateral: standard deviations for the colour component
            of the colour-dependent term.
        kernel_bilateral: kernel precision matrix for the colour-dependent term
            (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
        normalisation_bilateral: normalisation for the colour-dependent term
            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
        Refined predictions after MAP inference.
    """
    _, h, w, _ = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')

    d = dcrf.DenseCRF2D(h, w, n_classes)
    U = -np.log(probs)
    U = U.reshape((n_classes, -1))
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert(img.shape[1:3] == (
            h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape(
        (n_classes, h, w)).transpose(1, 2, 0)
    preds = np.expand_dims(preds, 0)
    return preds


class GradientReverseLayer(object):

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, gamma=1.0):
        grad_name = "GradientReverse%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _gradients_reverse(op, grad):
            return [tf.neg(grad) * gamma]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


gradient_reverse = GradientReverseLayer()


def resnext_block(inputs, nb_blocks, out_channels, is_training, reuse, cardinality,
                  downsample=False, downsample_strides=2, activation=relu, batch_norm=None, batch_norm_args=None, name="ResNeXtBlock", **kwargs):
    """ ResNeXt Block.
    resnext paper https://arxiv.org/pdf/1611.05431.pdf

    Args:
        inputs: `Tensor`. Inputs 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        out_channels: `int`. The number of convolutional filters of the
            layers surrounding the bottleneck layer.
        cardinality: `int`. Number of aggregated residual transformations.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `function` (returning a `Tensor`).
        batch_norm: `bool`. If True, apply batch normalization.
        use_ bias: `bool`. If True, a bias is used.
        w_init: `function`, Weights initialization.
        b_init: `tf.Tensor`. Bias initialization.
        w_regularizer: `function`. Add a regularizer to this
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
            override name.
        name: A name for this layer (optional). Default: 'ResNeXtBlock'.

    Returns:
        4-D Tensor [batch, new height, new width, out_channels].
    """
    resnext = inputs
    input_shape = util.get_input_shape(inputs)
    in_channels = input_shape[-1]

    card_values = [1, 2, 4, 8, 32]
    bottleneck_values = [64, 40, 24, 14, 4]
    bottleneck_size = bottleneck_values[card_values.index(cardinality)]
    group_width = [64, 80, 96, 112, 128]

    assert cardinality in card_values, "cardinality must be in [1, 2, 4, 8, 32]"

    with tf.variable_scope(name, reuse=reuse):
        for i in range(nb_blocks):
            identity = resnext
            if not downsample:
                downsample_strides = 1

            resnext = conv2d(resnext, bottleneck_size, is_training, reuse, filter_size=1,
                             stride=downsample_strides, batch_norm=batch_norm, batch_norm_args=batch_norm_args, activation=activation, padding='valid', name='conv1', **kwargs)

            resnext = depthwise_conv2d(
                resnext, cardinality, is_training, reuse, filter_size=3, batch_norm=batch_norm, batch_norm_args=batch_norm_args, activation=relu, stride=1, **kwargs)
            resnext = conv2d(resnext, out_channels, is_training, reuse, filter_size=1,
                             stride=1, batch_norm=None, activation=activation, padding='valid', name='conv2', **kwargs)

            if batch_norm is not None:
                batch_norm_args = batch_norm_args or {}
                resnext = batch_norm(resnext, is_training=is_training,
                                     reuse=reuse, trainable=trainable, **batch_norm_args)
            if downsample_strides > 1:
                identity = avg_pool_2d(
                    identity, filter_size=1, stride=downsample_strides)

            if in_channels != out_channels:
                ch = (out_channels - in_channels) // 2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

            resnext = resnext + identity
            resnext = activation(resnext)

        return resnext


def embedding(inputs, vocab_dim, embedding_dim, reuse, validate_indices=False,
              w_init=tf.random_uniform_initializer(-1., 1.), trainable=True, normalize=False, vocab_freqs=None, name="Embedding"):
    """ Embedding.
    Embedding layer for a sequence of integer ids or floats.

    Args:
        inputs: a 2-D `Tensor` [samples, ids].
        vocab_dim: list of `int`. Vocabulary size (number of ids).
        embedding_dim: list of `int`. Embedding size.
        validate_indices: `bool`. Whether or not to validate gather indices.
        w_init:  Weights initialization.
        trainable: `bool`. If True, weights will be trainable.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        name: A name for this layer (optional). Default: 'Embedding'.

    Returns:
        3-D Tensor [samples, embedded_ids, features].
    """

    input_shape = util.get_input_shape(inputs)
    assert len(input_shape) == 2, "Input Tensor shape must be 2-D"

    with tf.variable_scope(name, reuse=reuse):
        with tf.device('/cpu:0'):
            W = tf.get_variable("W", shape=[vocab_dim, embedding_dim],
                                initializer=w_init, trainable=trainable)
        if normalize:
            assert vocab_freqs is not None
            vocab_freqs = tf.constant(
                vocab_freqs, dtype=tf.float32, shape=(vocab_dim, 1))
            W = _normalize(W, vocab_freqs)

        output = tf.cast(inputs, tf.int32)
        output = tf.nn.embedding_lookup(W, output,
                                        validate_indices=validate_indices)

    shape = [-1] + output.get_shape().as_list()[1:3] + [1]
    # seq_length = util.retrieve_seq_length(tf.reshape(inputs, shape))

    return output


def _normalize(emb, vocab_freqs):
    weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev


def gated_layer(inputs, layer, num_units, is_training, reuse, name='gated_layer', **kwargs):
    """ Gated unit for language modelling

    Args:
        inputs: a 3-D/4-D `Tensor`, input [samples, timesteps, input_dim]
        layer: a `layer`, layer to pass the inputs
            e.g. `tefla.core.layers`
        num_units: a `int`, number of units for each layer
        is_training: a `boolean`, Training if its true
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        name: A name for this layer (optional). Default: 'gated_layer'.

    Returns:
        a 3-D/4-D `Tensor`, output of the gated unit

    """
    activation = kwargs.get('activation')
    with tf.name_scope(name):
        inputs = activation(inputs)
        inputs_shortcut = layer(inputs, num_units, is_training, reuse, filter_size=1,
                                name=name + '_shortcut', **kwargs)
        output1 = layer(inputs, num_units, is_training, reuse,
                        name=name + '_layer1', **kwargs)
        output2 = layer(inputs, num_units, is_training, reuse,
                        name=name + '_layer2', **kwargs)
        output = tf.multiply(output1, tf.sigmoid(output2))
        return output + inputs_shortcut


def glimpseSensor(img, normLoc, minRadius=4, depth=1, sensorBandwidth=12):
    """ Returns glimpses at the locations

    Args:
        img: a 4-D `Tensor`, [batch_size, width, height, channels]
        normloc: a `float`, [0, 1] normalized location
        minRadius: a `int`, min radius for zooming
        depth: a `int`, number of zooms
        sensorbandwidth: a `int`, output glimpse size, width/height

    Returns:
        a 5-D `tensor` of glimpses
    """
    input_shape = img.get_shape().as_list()
    img_size = input_shape[1]
    loc = tf.round(((normLoc + 1) / 2.0) * img_size)
    loc = tf.cast(loc, tf.int32)
    zooms = []
    for k in range(batch_size):
        imgZooms = []
        one_img = img[k, :, :, :]
        max_radius = minRadius * (2 ** (depth - 1))
        offset = 2 * max_radius
        one_img = tf.image.pad_to_bounding_box(one_img, offset, offset,
                                               max_radius * 4 + img_size, max_radius * 4 + img_size)

        for i in range(depth):
            r = int(minRadius * (2 ** (i)))
            d_raw = 2 * r
            d = tf.constant(d_raw, shape=[1])
            d = tf.tile(d, [2])
            loc_k = loc[k, :]
            adjusted_loc = offset + loc_k - r
            one_img2 = tf.reshape(one_img, (one_img.get_shape()[
                                  0].value, one_img.get_shape()[1].value, input_shape[3]))
            zoom = tf.slice(one_img2, adjusted_loc, d)
            zoom = tf.image.resize_bilinear(tf.reshape(
                zoom, (1, d_raw, d_raw, nput_shape[3])), (sensorBandwidth, sensorBandwidth))
            zoom = tf.reshape(
                zoom, (sensorBandwidth, sensorBandwidth, input_shape[3]))
            imgZooms.append(zoom)

        zooms.append(tf.stack(imgZooms))

    return tf.stack(zooms)


def pva_block_v1(x, num_units, name='pva_block_v1', **kwargs):
    """Adds a PVA block layer.
    convolution followed by crelu and scaling

    Args:
        x: A 4-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        num_units: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a int or tuple/list of 2 positive integers specifying the stride at which to
            compute output.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 4-D `Tensor` variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    trainable = kwargs.get('trainable')
    regularizer = kwargs.get('w_regularizer')
    activation = kwargs.get('activation')
    kwargs.update({'activation': None})
    with tf.variable_scope(name):
        output = conv2d(x, num_units, **kwargs)
        output = crelu(output)
        alpha = tf.get_variable('scale/alpha', shape=[2 * num_units, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0), trainable=trainable, regularizer=regularizer)
        beta = tf.get_variable('scale/beta', shape=[2 * num_units, ], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), trainable=trainable, regularizer=regularizer)
        output = tf.add(tf.multiply(output, alpha), beta)
        return activation(output)


def pva_block_v2(x, num_units, name='pva_block_v2', **kwargs):
    """Adds a PVA block v2 layer.
    first batch normalization followed by crelu and scaling, convolution is applied after scalling

    Args:
        x: A 4-D `Tensor` of with at least rank 2 and value for the last dimension,
            i.e. `[batch_size, in_height, in_width, depth]`,
        is_training: Bool, training or testing
        num_units: Integer or long, the number of output units in the layer.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        filter_size: a int or list/tuple of 2 positive integers specifying the spatial
            dimensions of of the filters.
        stride: a int or tuple/list of 2 positive integers specifying the stride at which to
            compute output.
        padding: one of `"VALID"` or `"SAME"`.
        activation: activation function, set to None to skip it and maintain
            a linear activation.
        batch_norm: normalization function to use. If
            `batch_norm` is `True` then google original implementation is used and
            if another function is provided then it is applied.
            default set to None for no normalizer function
        batch_norm_args: normalization function parameters.
        w_init: An initializer for the weights.
        w_regularizer: Optional regularizer for the weights.
        untie_biases: spatial dimensions wise baises
        b_init: An initializer for the biases. If None skip biases.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional name or scope for variable_scope/name_scope.
        use_bias: Whether to add bias or not

    Returns:
        The 4-D `Tensor` variable representing the result of the series of operations.
        e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

    Raises:
        ValueError: if `x` has rank less than 4 or if its last dimension is not set.
    """
    trainable = kwargs.get('trainable')
    reuse = kwargs.get('reuse')
    is_training = kwargs.get('is_training')
    regularizer = kwargs.get('w_regularizer')
    activation = kwargs.get('activation')
    kwargs.update({'activation': None})
    kwargs.update({'batch_norm': None})
    with tf.variable_scope(name, reuse=reuse):
        output = batch_norm(x, activation_fn=tf.nn.relu,
                            name='batch_norm', is_training=is_training, reuse=reuse)
        output = crelu(output)
        alpha = tf.get_variable('scale/alpha', shape=[2 * num_units, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0), trainable=trainable, regularizer=regularizer)
        beta = tf.get_variable('scale/beta', shape=[2 * num_units, ], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), trainable=trainable, regularizer=regularizer)
        output = tf.add(tf.multiply(output, alpha), beta)
        output = activation(output)
        output = conv2d(x, num_units, **kwargs)
        return output


def max_pool_2d_nxn_regions(inputs, output_size, mode='max'):
    """
    Performs a pooling operation that results in a fixed size:
    output_size x output_size.

    Used by spatial_pyramid_pool. Refer to appendix A in [1].

    Args:
        inputs: A 4D Tensor (B, H, W, C)
        output_size: The output size of the pooling operation.
        mode: The pooling mode {max, avg}

    Returns:
        A list of tensors, for each output bin.
        The list contains output_size * output_size elements, where
        each elment is a Tensor (N, C).

    """
    inputs_shape = tf.shape(inputs)
    h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
    w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)

    if mode == 'max':
        pooling_op = tf.reduce_max
    elif mode == 'avg':
        pooling_op = tf.reduce_mean
    else:
        msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode))

    result = []
    n = output_size
    for row in range(output_size):
        for col in range(output_size):
            # start_h = floor(row / n * h)
            start_h = tf.cast(
                tf.floor(tf.mul(tf.divide(row, n), tf.cast(h, tf.float32))), tf.int32)
            # end_h = ceil((row + 1) / n * h)
            end_h = tf.cast(
                tf.ceil(tf.mul(tf.divide((row + 1), n), tf.cast(h, tf.float32))), tf.int32)
            # start_w = floor(col / n * w)
            start_w = tf.cast(
                tf.floor(tf.mul(tf.divide(col, n), tf.cast(w, tf.float32))), tf.int32)
            # end_w = ceil((col + 1) / n * w)
            end_w = tf.cast(
                tf.ceil(tf.mul(tf.divide((col + 1), n), tf.cast(w, tf.float32))), tf.int32)
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = pooling_op(pooling_region, axis=(1, 2))
            result.append(pool_result)
    return result


def spatial_pyramid_pool(inputs, dimensions=[2, 1], mode='max', implementation='kaiming'):
    """
    Performs spatial pyramid pooling (SPP) over the input.
    It will turn a 2D input of arbitrary size into an output of fixed
    dimenson.
    Hence, the convlutional part of a DNN can be connected to a dense part
    with a fixed number of nodes even if the dimensions of the input
    image are unknown.

    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features.
    :math:`M_i` is given by :math:`n_i * n_i`, with :math:`n_i` as the number
    of pooling operations per dimension level :math:`i`.

    The length of the parameter dimensions is the level of the spatial pyramid.

    Args:
        inputs: A 4D Tensor (B, H, W, C).
        dimensions: The list of :math:`n_i`'s that define the output dimension
        of each pooling level :math:`i`. The length of dimensions is the level of
        the spatial pyramid.
        mode: Pooling mode 'max' or 'avg'.
        implementation: The implementation to use, either 'kaiming' or 'fast'.
        kamming is the original implementation from the paper, and supports variable
        sizes of input vectors, which fast does not support.

    Returns:
        A fixed length vector representing the inputs.

    Notes:
        SPP should be inserted between the convolutional part of a DNN and it's
        dense part. Convolutions can be used for arbitrary input dimensions, but
        the size of their output will depend on their input dimensions.
        Connecting the output of the convolutional to the dense part then
        usually demands us to fix the dimensons of the network's input.
        The spatial pyramid pooling layer, however, allows us to leave
        the network input dimensions arbitrary.
        The advantage over a global pooling layer is the added robustness
        against object deformations due to the pooling on different scales.

    """
    pool_list = []
    if implementation == 'kaiming':
        for pool_dim in dimensions:
            pool_list += max_pool_2d_nxn_regions(inputs, pool_dim, mode)
    else:
        shape = inputs.get_shape().as_list()
        for d in dimensions:
            h = shape[1]
            w = shape[2]
            ph = np.ceil(h * 1.0 / d).astype(np.int32)
            pw = np.ceil(w * 1.0 / d).astype(np.int32)
            sh = np.floor(h * 1.0 / d + 1).astype(np.int32)
            sw = np.floor(w * 1.0 / d + 1).astype(np.int32)
            pool_result = tf.nn.max_pool(inputs,
                                         ksize=[1, ph, pw, 1],
                                         strides=[1, sh, sw, 1],
                                         padding='SAME')
            pool_list.append(tf.reshape(
                pool_result, [tf.shape(inputs)[0], -1]))
    return tf.concat(1, pool_list)
