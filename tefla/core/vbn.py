import tensorflow as tf


class Virtual_Batch_Norm(object):

    def __init__(self, image_size=108):
        self.image_size = image_size

    def vbn(self, tensor, name, disable_vbn=False, half=None):
        if disable_vbn:
            class Dummy(object):

                def __init__(self, tensor, ignored, half):
                    self.reference_output = tensor

                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbnl(self, tensor, name, disable_vbn=False, half=None):
        if disable_vbn:
            class Dummy(object):

                def __init__(self, tensor, ignored, half):
                    self.reference_output = tensor

                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBNL
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbnlp(self, tensor, name, disable_vbn=False, half=None):
        if disable_vbn:
            class Dummy(object):

                def __init__(self, tensor, ignored, half):
                    self.reference_output = tensor

                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBNLP
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbn1(self, tensor, name, disable_vbn=False):
        return self.vbn(tensor, name, disable_vbn=disable_vbn, half=1)

    def vbn2(self, tensor, name, disable_vbn=False):
        return self.vbn(tensor, name, disable_vbn=disable_vbn, half=2)


class VBNL(object):
    """
    Virtual Batch Normalization, Log scale for the scale parameter
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name):
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0], [
                                shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
            self.mean_sq = tf.reduce_mean(
                tf.square(half), [0, 1, 2], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name):
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [0, 1, 2], keep_dims=True)
            new_mean_sq = tf.reduce_mean(
                tf.square(x), [0, 1, 2], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, name='vbnl'):
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        with tf.variable_scope(name):
            self.gamma_driver = tf.get_variable(
                "gamma_driver", [shape[-1]], initializer=tf.random_normal_initializer(0., 0.02))
            gamma = tf.exp(self.gamma_driver)
            gamma = tf.reshape(gamma, [1, 1, 1, -1])
            self.beta = tf.get_variable(
                "beta", [shape[-1]], initializer=tf.constant_initializer(0.))
            beta = tf.reshape(self.beta, [1, 1, 1, -1])
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        out = out * gamma
        out = out + beta
        return out


class VBNLP(object):
    """
    Virtual Batch Normalization, Log scale for the scale parameter, per-Pixel normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name):
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0], [
                                shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(half), [0], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name):
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [0], keep_dims=True)
            new_mean_sq = tf.reduce_mean(tf.square(x), [0], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, name='vbnlp'):
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        with tf.variable_scope(name):
            self.gamma_driver = tf.get_variable(
                "gamma_driver", shape[1:], initializer=tf.random_normal_initializer(0., 0.02))
            gamma = tf.exp(self.gamma_driver)
            gamma = tf.expand_dims(gamma, 0)
            self.beta = tf.get_variable(
                "beta", shape[1:], initializer=tf.constant_initializer(0.))
            beta = tf.expand_dims(self.beta, 0)
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        out = out * gamma
        out = out + beta
        return out


class VBN(object):
    """
    Virtual Batch Normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name):
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0], [
                                shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
            self.mean_sq = tf.reduce_mean(
                tf.square(half), [0, 1, 2], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name):
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [1, 2], keep_dims=True)
            new_mean_sq = tf.reduce_mean(
                tf.square(x), [0, 1, 2], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq)
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, name='vbn'):
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        assert len(shape) == 4
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        with tf.variable_scope(name):
            self.gamma = tf.get_variable(
                "gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))
            gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
            self.beta = tf.get_variable(
                "beta", [shape[-1]], initializer=tf.constant_initializer(0.))
            beta = tf.reshape(self.beta, [1, 1, 1, -1])
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        out = out * gamma
        out = out + beta
        return out
