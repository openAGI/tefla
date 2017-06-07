# -----------------------------------------------------#
# Original scripts from https://github.com/tensorflow/models/blob/master/slim/datasets/download_and_convert_cifar10.py
# Enhanced by Mrinal Haloi
# Enhancement Copyright 2016, Mrinal Haloi
# -----------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tefla.dataset.image_to_tfrecords import TFRecords

# The URL where the CIFAR data can be downloaded.
_DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# The number of training files.
_NUM_TRAIN_FILES = 5

# The height and width of each image.
_IMAGE_SIZE = 32

# The names of the classes.
_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


def _add_to_tfrecord(filename, tfrecord_writer, split_name=None, offset=0):
    """Loads data from the cifar10 pickle files and writes files to a TFRecord.

    Args:
        filename: The filename of the cifar10 pickle file.
        tfrecord_writer: The TFRecord writer to use for writing.
        offset: An offset into the absolute number of images previously written.

    Returns:
        The new offset.
    """
    with open(filename, 'r') as f:
        data = pickle.load(f)

    images = data['data']
    num_images = images.shape[0]

    images = images.reshape((num_images, 3, 32, 32))
    labels = data['labels']
    filenames = data['filenames']
    tfrecords = TFRecords()

    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_jpeg(image_placeholder)

        with tf.Session('') as sess:

            for j in range(num_images):
                sys.stdout.write(
                    '\r>> Reading file [%s] image %d/%d' % (filename, offset + j + 1, offset + num_images))
                sys.stdout.flush()

                image = np.squeeze(images[j]).transpose((1, 2, 0))
                label = labels[j]
                filename_image = filenames[j]

                jpg_string = sess.run(encoded_image, feed_dict={
                                      image_placeholder: image})

                example = tfrecords.convert_to_example(
                    filename_image, jpg_string, label, 'cifar', _IMAGE_SIZE, _IMAGE_SIZE)
                tfrecord_writer.write(example.SerializeToString())

        return offset + num_images


def _get_output_filename(dataset_dir, train, split_name):
    if train:
        return '%s/%s/cifar10_%s.tfrecord' % (dataset_dir, 'train', split_name)
    else:
        return '%s/%s/cifar10_%s.tfrecord' % (dataset_dir, 'val', split_name)


def _clean_up_temporary_files(dataset_dir):
    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

    tmp_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')
    tf.gfile.DeleteRecursively(tmp_dir)


def download_and_uncompress_tarball(tarball_url, dataset_dir):
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir, filename='labels.txt'):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def main(dataset_dir):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    tf.gfile.MkDir(os.path.join(dataset_dir, 'train'))
    tf.gfile.MkDir(os.path.join(dataset_dir, 'val'))
    if tf.gfile.Exists(dataset_dir + '/cifar-10-batches-py'):
        print('Dataset files already downloaded.')
    else:
        download_and_uncompress_tarball(_DATA_URL, dataset_dir)

    training_filename = _get_output_filename(dataset_dir, True,  'train')
    testing_filename = _get_output_filename(dataset_dir, False, 'val')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        offset = 0
        for i in range(_NUM_TRAIN_FILES):
            # 1-indexed.
            filename = os.path.join(
                dataset_dir, 'cifar-10-batches-py', 'data_batch_%d' % (i + 1))
            offset = _add_to_tfrecord(
                filename, tfrecord_writer, offset=offset, split_name='train')

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        filename = os.path.join(
            dataset_dir, 'cifar-10-batches-py', 'test_batch')
        _add_to_tfrecord(filename, tfrecord_writer, split_name='test')

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    write_label_file(labels_to_class_names, dataset_dir)

    _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Cifar10 dataset!')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        main(dataset_dir)
    else:
        print('Usage:python filename.py dataset_dir')
        exit(-1)
