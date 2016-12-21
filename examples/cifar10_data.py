from __future__ import absolute_import
# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.dataset.base import Dataset
from tefla.dataset.decoder import Decoder
from tefla.dataset.dataflow import Dataflow

data_dir='/home/artelus_server/data/cifar_10'
num_readers=8

features_keys = {
    'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
}

decoder = Decoder(features_keys)

dataset = Dataset('cifar10', decoder, data_dir)

dataflow = Dataflow(dataset, num_readers=num_readers, shuffle=True, min_queue_examples=100, capacity=200)

[image, label] = dataflow.get(['image', 'label'], [32, 32, 3])
[image_batch, label_batch] = dataflow.get_batch(32, [0.1 for _ in xrange(0, 10)], [32, 32, 3], init_probs=[0.1 for _ in xrange(0, 10)], threads_per_queue=4)


if __name__ == '__main__':
    sess = tf.Session()
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    image, label = sess.run([image, label])
    image_batch, label_batch = sess.run([image_batch, label_batch])
    coord.request_stop()
    coord.join(stop_grace_period_secs=0.05)
    print(image_batch.shape)
    sess.close()
