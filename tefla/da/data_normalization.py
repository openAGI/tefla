# -------------------------------------------------------------------#
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import

import numpy as np
import pickle
import tensorflow as tf

_EPSILON = 1e-8


class DataNormalization(object):
    """ Input Data Normalization.

    Computes inputs data normalization params, It also performs samplewise
    and global standardization. It can be use to calculate the samplewise
    and global mean and std of the dataset.
    It can be use to compute ZCA whitening also.

    Args:
        name: an optional name of the ops
    """

    def __init__(self, name="DataNormalization"):
        self.session = None
        # Data Persistence
        with tf.name_scope(name) as scope:
            self.scope = scope
        self.global_mean = self.PersistentParameter(scope, name="mean")
        self.global_std = self.PersistentParameter(scope, name="std")
        self.global_pc = self.PersistentParameter(scope, name="pc")

    def restore_params(self, session):
        """ Restore the normalization params from the given session

        Args:
            session: The session use to perform the computation

        Returns:
            Returns True/False based on restore success
        """
        self.global_mean.is_restored(session)
        self.global_std.is_restored(session)
        self.global_pc.is_restored(session)

    def initialize(self, dataset, session, limit=None):
        """ Initialize preprocessing methods
        that pre-requires calculation over entire dataset.

        Args:
            dataset: A `ndarray`, its a ndarray representation of the whole dataset.
            session: The session use to perform the computation
            limit: Number of data sample to use, if None, computes on the whole dataset
        """
        # If a value is already provided, it has priority
        if self.global_mean.value is not None:
            self.global_mean.assign(self.global_mean.value, session)
        # Otherwise, if it has not been restored, compute it
        if not self.global_mean.is_restored(session):
            print("---------------------------------")
            print("Preprocessing... Calculating mean over all dataset "
                  "(this may take long)...")
            self.compute_global_mean(dataset, session, limit)
            print("Mean: " + str(self.global_mean.value) + " (To avoid "
                  "repetitive computation, add it to argument 'mean' of "
                  "`add_featurewise_zero_center`)")
        # If a value is already provided, it has priority
        if self.global_std.value is not None:
            self.global_std.assign(self.global_std.value, session)
        # Otherwise, if it has not been restored, compute it
        if not self.global_std.is_restored(session):
            print("---------------------------------")
            print("Preprocessing... Calculating std over all dataset "
                  "(this may take long)...")
            self.compute_global_std(dataset, session, limit)
            print("STD: " + str(self.global_std.value) + " (To avoid "
                  "repetitive computation, add it to argument 'std' of "
                  "`add_featurewise_stdnorm`)")
        # If a value is already provided, it has priority
        if self.global_pc.value is not None:
            self.global_pc.assign(self.global_pc.value, session)
        # Otherwise, if it has not been restored, compute it
        if not self.global_pc.is_restored(session):
            print("---------------------------------")
            print("Preprocessing... PCA over all dataset "
                  "(this may take long)...")
            self.compute_global_pc(dataset, session, limit)
            with open('PC.pkl', 'wb') as f:
                pickle.dump(self.global_pc.value, f)
            print("PC saved to 'PC.pkl' (To avoid repetitive computation, "
                  "load this pickle file and assign its value to 'pc' "
                  "argument of `add_zca_whitening`)")

    def zca_whitening(self, image):
        """ZCA wgitening

        Args:
            image: input image

        Returns:
            ZCA whitened image
        """
        flat = np.reshape(image, image.size)
        white = np.dot(flat, self.global_pc.value)
        s1, s2, s3 = image.shape[0], image.shape[1], image.shape[2]
        image = np.reshape(white, (s1, s2, s3))
        return image

    def normalize_image(self, batch):
        """Normalize image to [0,1] range

        Args:
            batch: a single image or batch of images
        """
        return np.array(batch) / 255.

    def crop_center(self, batch, shape):
        """Center crop of input images

        Args:
            batch: a single image or batch of images
            shape: output shape

        Returns:
            batch/single image with center crop as the value
        """
        oshape = np.shape(batch[0])
        nh = int((oshape[0] - shape[0]) * 0.5)
        nw = int((oshape[1] - shape[1]) * 0.5)
        new_batch = []
        for i in range(len(batch)):
            new_batch.append(batch[i][nh: nh + shape[0], nw: nw + shape[1]])
        return new_batch

    def samplewise_zero_center(self, image, per_channel=False):
        """Samplewise standardization

        Args:
            image: input image
            per_channel: whether to compute per image mean, default: False

        Returns:
            zero centered image
        """
        if not per_channel:
            im_zero_mean = image - np.mean(image)
        else:
            im_zero_mean = image - \
                np.mean(image, axis=(0, 1, 2), keepdims=True)
        return im_zero_mean

    def samplewise_stdnorm(self, image, per_channel=False):
        """Samplewise standardization

        Args:
            image: input image
            per_channel: whether to compute per image std, default: False

        Returns:
            zero centered image
        """
        if not per_channel:
            im_std = np.std(image)
            im_zero_std = image / (im_std + _EPSILON)
        else:
            im_std = np.std(image, axis=(0, 1, 2), keepdims=True)
            im_zero_std = image / (im_std + _EPSILON)
        return im_zero_std

    def compute_global_mean(self, dataset, session, limit=None):
        """ Compute mean of a dataset. A limit can be specified for faster
        computation, considering only 'limit' first elements.

        Args:
            dataset: A `ndarray`, its a ndarray representation of the whole dataset.
            session: The session use to perform the computation
            limit: Number of data sample to use, if None, computes on the whole dataset

        Returns:
            global dataset mean
        """
        _dataset = dataset
        mean = 0.
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        if isinstance(_dataset, np.ndarray) and not self.global_mean_pc:
            mean = np.mean(_dataset)
        else:
            # Iterate in case of non numpy data
            for i in range(len(dataset)):
                if not self.global_mean_pc:
                    mean += np.mean(dataset[i]) / len(dataset)
                else:
                    mean += (np.mean(dataset[i], axis=(0, 1),
                                     keepdims=True) / len(dataset))[0][0]
        self.global_mean.assign(mean, session)
        return mean

    def compute_global_std(self, dataset, session, limit=None):
        """ Compute std of a dataset. A limit can be specified for faster
        computation, considering only 'limit' first elements.
        Args:
            dataset: A `ndarray`, its a ndarray representation of the whole dataset.
            session: The session use to perform the computation
            limit: Number of data sample to use, if None, computes on the whole dataset

        Returns:
            global dataset std
        """
        _dataset = dataset
        std = 0.
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        if isinstance(_dataset, np.ndarray) and not self.global_std_pc:
            std = np.std(_dataset)
        else:
            for i in range(len(dataset)):
                if not self.global_std_pc:
                    std += np.std(dataset[i]) / len(dataset)
                else:
                    std += (np.std(dataset[i], axis=(0, 1),
                                   keepdims=True) / len(dataset))[0][0]
        self.global_std.assign(std, session)
        return std

    class PersistentParameter:
        """ Create a persistent variable that will be stored into the Graph.
        """

        def __init__(self, scope, name):
            self.is_required = False
            with tf.name_scope(scope):
                with tf.device('/cpu:0'):
                    # One variable contains the value
                    self.var = tf.Variable(0., trainable=False, name=name,
                                           validate_shape=False)
                    # Another one check if it has been restored or not
                    self.var_r = tf.Variable(False, trainable=False,
                                             name=name + "_r")
            # RAM saved vars for faster access
            self.restored = False
            self.value = None

        def is_restored(self, session):
            """ Check whether a param is restored from a session

            Args:
                session: session to perform ops

            Returns:
                a bool, the status of the op
            """
            if self.var_r.eval(session=session):
                self.value = self.var.eval(session=session)
                return True
            else:
                return False

        def assign(self, value, session):
            """ Assign a value to session variable

            Args:
                value: the value to add
                session: session to perform ops
            """
            session.run(tf.assign(self.var, value, validate_shape=False))
            self.value = value
            session.run(self.var_r.assign(True))
            self.restored = True
