import tensorflow as tf


def spatialtransformer(U, theta, downsample_factor=1.0, num_transform=1, name='SpatialTransformer', **kwargs):
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
            batch_size, num_transforms = map(int, theta.get_shape().as_list()[:2])
            indices = [[i] * num_transforms for i in range(batch_size)]
            U = tf.gather(U, tf.reshape(indices, [-1]))

        input_shape = U.get_shape().as_list()
        batch_size = input_shape[0]
        num_channels = input_shape[3]
        theta = tf.reshape(theta, (-1, 2, 3))
        theta = tf.cast(theta, tf.float32)
        if not isinstance(downsample_factor, tf.float32):
            downsample_factor = tf.cast(downsample_factor, tf.float32)

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = tf.cast(input_shape[1] / downsample_factor, tf.float32)
        out_width = tf.cast(input_shape[2] / downsample_factor, tf.float32)
        grid = _meshgrid(out_height, out_width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.pack([batch_size]))
        grid = tf.reshape(grid, tf.pack([batch_size, 3, -1]))

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.batch_matmul(theta, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate(U, x_s_flat, y_s_flat, downsample_factor)

        output = tf.reshape(input_transformed, tf.pack([batch_size, out_height, out_width, num_channels]))
        return output


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, tf.int32)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])


def _interpolate(im, x, y, downsample_factor):
    with tf.variable_scope('_interpolate'):
        input_shape = im.get_shape().as_list()
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        out_height = tf.cast(height / downsample_factor, tf.float32)
        out_width = tf.cast(width / downsample_factor, tf.float32)
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
        im_flat = tf.reshape(im, tf.pack([-1, channels]))
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
        x_t = tf.matmul(tf.ones(shape=tf.pack([height, 1])), tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1), tf.ones(shape=tf.pack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(0, [x_t_flat, y_t_flat, ones])
        return grid
