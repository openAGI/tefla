"""
 Modified version of Original code https://github.com/ischlag/tensorflow-input-pipelines/blob/master/datasets/svhn.py

"""
import tensorflow as tf
import numpy as np
import threading

import svhn


class svhn_data(object):
    """
    Downloads the SVHN dataset and creates an input pipeline ready to be fed into a model.

    - Reshapes flat images into 32x32
    - converts [0 1] to [-1 1]
    - shuffles the input
    - builds batches
    """
    NUM_THREADS = 8
    NUMBER_OF_CLASSES = 10

    TRAIN_SET_SIZE = 73257
    TEST_SET_SIZE = 26032
    IMAGE_WIDTH = 32
    IMAGE_HEIGHT = 32
    NUM_OF_CHANNELS = 3

    def __init__(self, sess, feed_size=200, feed_queue_capacity=800, batch_queue_capacity=5, min_after_dequeue=4):
        """ Downloads the cifar100 data if necessary. """
        print("Loading SVHN data")
        self.feed_size = feed_size
        self.feed_queue_capacity = feed_queue_capacity
        self.batch_queue_capacity = batch_queue_capacity + 3 * 128
        self.min_after_dequeue = min_after_dequeue
        self.sess = sess
        svhn.download_data()

    def train_batch(self, batch_size, shuffle=False, standardizer=None):
        images, _, targets = svhn.load_training_data()
        return self.__build_generic_data_tensor(images,
                                                targets,
                                                shuffle, batch_size=batch_size,
                                                standardizer=standardizer)

    def test_batch(self, batch_size, shuffle=False, standardizer=None):
        images, _, targets = svhn.load_test_data()
        return self.__build_generic_data_tensor(images,
                                                targets,
                                                shuffle, batch_size=batch_size,
                                                standardizer=standardizer)

    def __build_generic_data_tensor(self, raw_images, raw_targets, shuffle, batch_size=32, standardizer=None):
        """
        Creates the input pipeline and performs some preprocessing.
        The full dataset needs to fit into memory for this version.
        """

        # load the data from numpy into our queue in blocks of feed_size
        # samples
        set_size, width, height, channels = raw_images.shape

        image_input = tf.placeholder(
            tf.float32, shape=[self.feed_size, width, height, channels])
        target_input = tf.placeholder(
            tf.float32, shape=[self.feed_size, self.NUMBER_OF_CLASSES])

        if self.shuffle:
            self.queue = tf.RandomShuffleQueue(
                capacity=self.feed_queue_capacity, min_after_dequeue=self.batch_queue_capacity, dtypes=[tf.float32, tf.float32])
        else:
            self.queue = tf.FIFOQueue(capacity=self.feed_queue_capacity, dtypes=[
                                      tf.float32, tf.float32], shapes=[[width, height, channels], [self.NUMBER_OF_CLASSES]])
        enqueue_op = self.queue.enqueue_many([image_input, target_input])
        image, target = self.queue.dequeue()

        if standardizer is not None:
            image = standardizer(image)

        images_batch, target_batch = tf.train.batch([image, target],
                                                    batch_size=batch_size,
                                                    capacity=self.batch_queue_capacity)

        def enqueue(sess):
            under = 0
            max = len(raw_images)
            while not self.coord.should_stop():
                upper = under + self.feed_size
                if upper <= max:
                    curr_data = raw_images[under:upper]
                    curr_target = raw_targets[under:upper]
                    under = upper
                else:
                    rest = upper - max
                    curr_data = np.concatenate(
                        (raw_images[under:max], raw_images[0:rest]))
                    curr_target = np.concatenate(
                        (raw_targets[under:max], raw_targets[0:rest]))
                    under = rest

                sess.run(enqueue_op, feed_dict={image_input: curr_data,
                                                target_input: curr_target})

        enqueue_thread = threading.Thread(target=enqueue, args=[self.sess])

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            coord=self.coord, sess=self.sess)

        enqueue_thread.isDaemon()
        enqueue_thread.start()

        return images_batch, target_batch

    def __del__(self):
        self.close()

    def close(self):
        self.queue.close(cancel_pending_enqueues=True)
        self.coord.request_stop()
        self.coord.join(self.threads)


def standardizer(image):
    return tf.image.per_image_standardization(image)
