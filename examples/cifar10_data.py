from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tefla.dataset.base import Dataset
from tefla.dataset.decoder import Decoder
from tefla.dataset.dataflow import Dataflow

data_dir='/media/Data/'
num_readers=8

features_keys = {
    'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
}

decoder = Decoder(features_keys)

dataset = Dataset('cifar10', decoder, data_dir)

dataflow = Dataflow(dataset, num_readers=num_readers, shuffle=True, capacity=2048)

[image, label] = dataflow.get(['image', 'label'])
