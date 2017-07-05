# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.dataset.base import Dataset
from tefla.dataset.decoder import Decoder
from tefla.dataset.dataflow import Dataflow
from tefla.da.data_augmentation import inputs, distorted_inputsv2

data_dir = '/home/artelus_server/data/cifar_10/train'
num_readers = 8

features_keys = {
    'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
}

decoder = Decoder(features_keys)

dataset = Dataset('cifar10', decoder, data_dir)

dataflow = Dataflow(dataset, num_readers=num_readers,
                    shuffle=True, min_queue_examples=100, capacity=200)

# [image, label] = dataflow.get(['image', 'label'], [32, 32, 3])
image_batch, label_batch = dataflow.get_batch(32, tf.convert_to_tensor([0.1 for _ in xrange(0, 10)]), [
    32, 32, 3], init_probs=tf.convert_to_tensor([0.1 for _ in xrange(0, 10)]), threads_per_queue=4)
images, labels = distorted_inputsv2(dataflow, [32, 32, 3], [28, 28], target_probs=tf.convert_to_tensor([
                                    0.1 for _ in range(0, 10)]), init_probs=tf.convert_to_tensor([0.1 for _ in xrange(0, 10)]), batch_size=32, num_preprocess_threads=32, num_readers=8)


if __name__ == '__main__':
    sess = tf.Session()
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    # image, label = sess.run([image, label])
    images, labels = sess.run([images, labels])
    print(labels)
    print(images.shape)
    image_batch, label_batch = sess.run([image_batch, label_batch])
    print(label_batch)
    print(image_batch.shape)
    coord.request_stop()
    coord.join(stop_grace_period_secs=0.05)
    print(labels)
    # sess.close()
