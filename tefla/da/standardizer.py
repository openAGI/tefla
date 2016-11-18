from __future__ import division, print_function, absolute_import

import numpy as np


# Think of the classes in this module as da post-processors.
# The standardizer name is a historical artifact.
# A post-processor can be a standardizer, of course.

class NoDAMixin(object):
    def da_processing_params(self):
        return {}

    def set_tta_args(self, **kwargs):
        pass


class NoOpStandardizer(NoDAMixin):
    def __call__(self, img, is_training):
        return img


class SamplewiseStandardizer(NoDAMixin):
    def __init__(self, clip, channel_wise=False):
        self.clip = clip
        self.channel_wise = channel_wise

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


class AggregateStandardizer(object):
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
        if color_vec is None:
            if not sigma > 0.0:
                color_vec = np.zeros(3, dtype=np.float32)
            else:
                color_vec = np.random.normal(0.0, sigma, 3)

        alpha = color_vec.astype(np.float32) * self.ev
        noise = np.dot(self.u, alpha.T)
        return img + noise[:, np.newaxis, np.newaxis]
