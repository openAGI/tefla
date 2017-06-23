from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf


class NoDAMixin(object):
    """Base DA mixin class"""

    def __init__(self):
        super(NoDAMixin, self).__init__()

    def da_processing_params(self):
        """DA processing params"""
        return {}

    def set_tta_args(self, **kwargs):
        """tta quasi transforms params set"""
        pass


class NoOpStandardizer(NoDAMixin):
    """No operation class"""

    def __call__(self, img, is_training):
        return img


class SamplewiseStandardizer(NoDAMixin):
    """Samplewise Standardizer

    Args:
        clip: max/min allowed value in the output image
            e.g.: 6
        channel_wise: perform standarization separately accross channels

    """

    def __init__(self, clip, channel_wise=False):
        self.clip = clip
        self.channel_wise = channel_wise
        super(SamplewiseStandardizer, self).__init__()

    def __call__(self, img, is_training):
        if self.channel_wise:
            img_mean = img.mean(axis=(1, 2))
            img_std = img.std(axis=(1, 2))
            np.subtract(img, img_mean.reshape(3, 1, 1), out=img)
            np.divide(img, (img_std + 1e-4).reshape(3, 1, 1), out=img)
        else:
            img_mean = img.mean()
            img_std = img.std()
            np.subtract(img, img_mean, out=img)
            np.divide(img, img_std + 1e-4, out=img)

        np.clip(img, -self.clip, self.clip, out=img)
        return img


class SamplewiseStandardizerTF(NoDAMixin):
    """Samplewise Standardizer

    Args:
        clip: max/min allowed value in the output image
            e.g.: 6
        channel_wise: perform standarization separately accross channels

    """

    def __init__(self, clip, channel_wise=False):
        self.clip = clip
        self.channel_wise = channel_wise
        super(SamplewiseStandardizerTF, self).__init__()

    def __call__(self, img, is_training):
        if self.channel_wise:
            img_mean, img_var = tf.nn.moments(img, axes=[0, 1])
            img = tf.div(tf.subtract(img, img_mean),
                         tf.sqrt(img_var) + tf.constant(1e-4))
        else:
            img_mean, img_var = tf.nn.moments(img, axes=[0, 1, 2])
            img = tf.div(tf.subtract(img, img_mean),
                         tf.sqrt(img_var) + tf.constant(1e-4))
        img = tf.clip_by_value(img, -self.clip, self.clip)
        return img


class AggregateStandardizer(object):
    """Aggregate Standardizer

    Creates a standardizer based on whole training dataset

    Args:
        mean: 1-D array, aggregate mean array
            e.g.: mean is calculated for each color channel, R, G, B
        std: 1-D array, aggregate standard deviation array
            e.g.: std is calculated for each color channel, R, G, B
        u: 2-D array, eigenvector for the color channel variation
        ev: 1-D array, eigenvalues
        sigma: float, noise factor
        color_vec: an optional color vector
    """

    def __init__(self, mean, std, u, ev, sigma=0.0, color_vec=None):
        self.mean = mean
        self.std = std
        self.u = u
        self.ev = ev
        self.sigma = sigma
        self.color_vec = color_vec

    def da_processing_params(self):
        return {'sigma': self.sigma}

    def set_tta_args(self, **kwargs):
        self.color_vec = kwargs['color_vec']

    def __call__(self, img, is_training):
        np.subtract(img, self.mean[:, np.newaxis, np.newaxis], out=img)
        np.divide(img, self.std[:, np.newaxis, np.newaxis], out=img)
        if is_training:
            img = self.augment_color(img, sigma=self.sigma)
        else:
            # tta (test time augmentation)
            img = self.augment_color(img, color_vec=self.color_vec)
        return img

    def augment_color(self, img, sigma=0.0, color_vec=None):
        """Augment color

        Args:
            img: input image
            sigma: a float, noise factor
            color_vec: an optional color vec

        """
        if color_vec is None:
            if not sigma > 0.0:
                color_vec = np.zeros(3, dtype=np.float32)
            else:
                color_vec = np.random.normal(0.0, sigma, 3)

        alpha = color_vec.astype(np.float32) * self.ev
        noise = np.dot(self.u, alpha.T)
        return img + noise[:, np.newaxis, np.newaxis]


class AggregateStandardizerTF(object):
    """Aggregate Standardizer

    Creates a standardizer based on whole training dataset

    Args:
        mean: 1-D array, aggregate mean array
            e.g.: mean is calculated for each color channel, R, G, B
        std: 1-D array, aggregate standard deviation array
            e.g.: std is calculated for each color channel, R, G, B
        u: 2-D array, eigenvector for the color channel variation
        ev: 1-D array, eigenvalues
        sigma: float, noise factor
        color_vec: an optional color vector
    """

    def __init__(self, mean, std, u, ev, sigma=0.0, color_vec=None):
        self.mean = tf.reshape(tf.to_float(mean), shape=(1, 1, 3))
        self.std = tf.reshape(tf.to_float(std), shape=(1, 1, 3))
        self.u = tf.reshape(tf.to_float(u), shape=(3, 3))
        self.ev = tf.reshape(tf.to_float(ev), shape=(3,))
        self.sigma = tf.to_float(sigma)
        self.color_vec = color_vec

    def da_processing_params(self):
        return {'sigma': self.sigma}

    def set_tta_args(self, **kwargs):
        self.color_vec = kwargs['color_vec']

    def __call__(self, img, is_training):
        img = tf.subtract(img, self.mean)
        img = tf.divide(img, self.std)
        if is_training:
            img = self.augment_color(img, sigma=self.sigma)
        else:
            # tta (test time augmentation)
            img = self.augment_color(img, color_vec=self.color_vec)
        return img

    def augment_color(self, img, sigma=0.0, color_vec=None):
        """Augment color

        Args:
            img: input image
            sigma: a float, noise factor
            color_vec: an optional color vec

        """
        if color_vec is None:
            # tf.zeros(shape=(3,), dtype=tf.float32))
            color_vec = tf.random_normal(shape=(3,), mean=0.0, stddev=sigma)

        alpha = tf.multiply(tf.to_float(color_vec), self.ev)
        # noise = tf.reshape(tf.tensordot(self.u, alpha, 1), shape=(1, 1, 3))
        noise = tf.reshape(
            tf.matmul(self.u, tf.reshape(alpha, (3, 1))), shape=(1, 1, 3))
        return tf.add(img, noise)
