"""
 Modified version of Original code https://github.com/ischlag/tensorflow-input-pipelines/blob/master/datasets/mnist.py

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class MnistData:
    """
    Downloads the MNIST dataset and creates an input pipeline ready to be fed into a model.

    - Reshapes flat images into 28 x 28
    - converts [0 1] to [-1 1]
    - shuffles the input
    - builds batches
    """
    NUM_THREADS = 8
    NUMBER_OF_CLASSES = 10
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    NUM_OF_CHANNELS = 1

    def __init__(self, one_hot_labels=True):
        """ Downloads the mnist data if necessary. """
        print("Loading MNIST data")
        self.mnist = input_data.read_data_sets(
            'data/MNIST', one_hot=one_hot_labels)

        self.TRAIN_SET_SIZE = self.mnist.train.images.shape[0]
        self.TEST_SET_SIZE = self.mnist.test.images.shape[0]
        self.VALIDATION_SET_SIZE = self.mnist.validation.images.shape[0]

    def train_batch(self, batch_size, shuffle=True, standardizer=None):
        return self.__build_generic_data_tensor(self.mnist.train.images,
                                                self.mnist.train.labels, batch_size,
                                                shuffle, standardizer=standardizer)

    def test_batch(self, batch_size, shuffle=False, standardizer=None):
        return self.__build_generic_data_tensor(self.mnist.test.images,
                                                self.mnist.test.labels, batch_size,
                                                shuffle, standardizer=standardizer)

    def validation_batch(self, batch_size, shuffle=False, standardizer=None):
        return self.__build_generic_data_tensor(self.mnist.validation.images,
                                                self.mnist.validation.labels, batch_size,
                                                shuffle, standardizer=standardizer)

    def __build_generic_data_tensor(self, raw_images, raw_targets, batch_size, shuffle, standardizer=None):
        """ Creates the input pipeline and performs some preprocessing. """

        images = tf.convert_to_tensor(raw_images)
        targets = tf.convert_to_tensor(raw_targets)

        set_size = raw_images.shape[0]

        images = tf.reshape(images, [set_size, 28, 28, 1])
        image, label = tf.train.slice_input_producer(
            [images, targets], shuffle=shuffle)

        if standardizer is not None:
            image = standardizer(image)

        images_batch, labels_batch = tf.train.batch(
            [image, label], batch_size=batch_size, num_threads=self.NUM_THREADS)

        return images_batch, labels_batch


def standardizer(image):
    return tf.image.per_image_standardization(image)
