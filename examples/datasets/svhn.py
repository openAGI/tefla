import tensorflow as tf
import dataset_utils
import scipy.io
import os
import sys
import numpy as np
from tefla.dataset.image_to_tfrecords import TFRecords

data_path = "/tmp/SVHN/"

if not os.path.exists(data_path):
    os.makedirs(data_path)

data_url = 'http://ufldl.stanford.edu/housenumbers/'
train_data = 'train_32x32.mat'
test_data = 'test_32x32.mat'
extra_data = 'extra_32x32.mat'

num_classes = 10


def download_data(data_path=data_path):
    """Download the SVHN data if it doesn't exist yet."""

    dataset_utils.maybe_download(
        url=data_url + train_data, download_dir=data_path)
    dataset_utils.maybe_download(
        url=data_url + test_data, download_dir=data_path)
    dataset_utils.maybe_download(
        url=data_url + extra_data, download_dir=data_path)


def load_training_data():
    """
    Load all the training-data for the SVHN data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    train_data = scipy.io.loadmat(
        data_path + 'train_32x32.mat', variable_names='X').get('X')
    train_labels = scipy.io.loadmat(
        data_path + 'train_32x32.mat', variable_names='y').get('y')

    images = train_data.transpose((3, 0, 1, 2)) / 255.0
    cls = train_labels[:, 0]
    cls[cls == 10] = 0

    return images, cls, dataset_utils.one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test-data for the SVHN data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    test_data = scipy.io.loadmat(
        data_path + 'test_32x32.mat', variable_names='X').get('X')
    test_labels = scipy.io.loadmat(
        data_path + 'test_32x32.mat', variable_names='y').get('y')

    images = test_data.transpose((3, 0, 1, 2)) / 255.0
    cls = test_labels[:, 0]
    cls[cls == 10] = 0

    return images, cls, dataset_utils.one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_extra_data():
    extra_data = scipy.io.loadmat(
        data_path + 'extra_32x32.mat', variable_names='X').get('X')
    extra_labels = scipy.io.loadmat(
        data_path + 'extra_32x32.mat', variable_names='y').get('y')

    images = extra_data.transpose((3, 0, 1, 2)) / 255.0
    cls = extra_labels[:, 0]
    cls[cls == 10] = 0

    return images, cls, dataset_utils.one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def _add_to_tfrecord(images, labels, tfrecord_writer, split_name=None, offset=0):
    """Loads data from the cifar10 pickle files and writes files to a TFRecord.

    Args:
        images: numpy array of images
        labels: numpy array of labels
        tfrecord_writer: The TFRecord writer to use for writing.
        offset: An offset into the absolute number of images previously written.

    Returns:
        The new offset.
    """

    num_images = images.shape[0]

    tfrecords = TFRecords()

    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_jpeg(image_placeholder)

        with tf.Session('') as sess:

            for j in range(num_images):
                sys.stdout.write('\r>> Reading file image %d/%d' % (offset + j + 1, offset + num_images))
                sys.stdout.flush()

                image = np.squeeze(images[j])
                width, height = image.shape[:2]
                label = labels[j]
                filename_image = 'svhn_image_' + str(j)

                jpg_string = sess.run(encoded_image, feed_dict={image_placeholder: image})

                example = tfrecords.convert_to_example(filename_image, jpg_string, label, 'svhn', width, height)
                tfrecord_writer.write(example.SerializeToString())

        return offset + num_images


def _get_output_filename(dataset_dir, split_name):
    return '%s/cifar10_%s.tfrecord' % (dataset_dir, split_name)


def main(dataset_dir):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    download_data(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    images, _, labels = load_training_data()
    # set_size, width, height, channels
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        _add_to_tfrecord(images, labels, tfrecord_writer, offset=0, split_name='train')

    # Next, process the testing data:
    test_images, _, test_labels = load_test_data()
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        _add_to_tfrecord(test_images, test_labels, tfrecord_writer, split_name='test')

    print('\nFinished converting the Cifar10 dataset!')
