"""Omniglot dataset download and preprocessing
"""

import cPickle as pickle
import logging
import os
import sys
import subprocess
import numpy as np
from scipy.misc import imresize
from scipy.misc import imrotate
from scipy.ndimage import imread
import tensorflow as tf


def get_data(data_dir):
    """Get data in form suitable for episodic training.

    Returns:
      Train and test data as dictionaries mapping
      label to list of examples.
    """
    with tf.gfile.GFile(os.path.join(data_dir, 'train_omni.pkl')) as f:
        processed_train_data = pickle.load(f)
    with tf.gfile.GFile(os.path.join(data_dir, 'test_omni.pkl')) as f:
        processed_test_data = pickle.load(f)

    train_data = {}
    test_data = {}

    for data, processed_data in zip([train_data, test_data],
                                    [processed_train_data, processed_test_data]):
        for image, label in zip(processed_data['images'],
                                processed_data['labels']):
            if label not in data:
                data[label] = []
            data[label].append(image.reshape([-1]).astype('float32'))

    intersection = set(train_data.keys()) & set(test_data.keys())
    assert not intersection, 'Train and test data intersect.'
    ok_num_examples = [len(ll) == 20 for _, ll in train_data.iteritems()]
    assert all(ok_num_examples), 'Bad number of examples in train data.'
    ok_num_examples = [len(ll) == 20 for _, ll in test_data.iteritems()]
    assert all(ok_num_examples), 'Bad number of examples in test data.'

    logging.info('Number of labels in train data: %d.', len(train_data))
    logging.info('Number of labels in test data: %d.', len(test_data))

    return train_data, test_data


def crawl_directory(directory, augment_with_rotations=False,
                    first_label=0):
    """Crawls data directory and returns stuff."""
    label_idx = first_label
    images = []
    labels = []
    info = []

    for root, _, files in os.walk(directory):
        logging.info('Reading files from %s', root)
        fileflag = 0
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            img = imread(full_file_name, flatten=True)
            for i, angle in enumerate([0, 90, 180, 270]):
                if not augment_with_rotations and i > 0:
                    break

                images.append(imrotate(img, angle))
                labels.append(label_idx + i)
                info.append(full_file_name)

            fileflag = 1

        if fileflag:
            label_idx += 4 if augment_with_rotations else 1

    return images, labels, info


def resize_images(images, new_width, new_height):
    """Resize images to new dimensions."""
    resized_images = np.zeros([images.shape[0], new_width, new_height],
                              dtype=np.float32)

    for i in range(images.shape[0]):
        resized_images[i, :, :] = imresize(images[i, :, :],
                                           [new_width, new_height],
                                           interp='bilinear',
                                           mode=None)
    return resized_images


def write_datafiles(directory, write_file,
                    resize=True, rotate=False, original_size=105,
                    new_width=28, new_height=28,
                    first_label=0):
    """Load and preprocess images from a directory and write them to a file.

    Args:
      directory: Directory of alphabet sub-directories.
      write_file: Filename to write to.
      resize: Whether to resize the images.
      rotate: Whether to augment the dataset with rotations.
      new_width: New resize width.
      new_height: New resize height.
      first_label: Label to start with.

    Returns:
      Number of new labels created.
    """

    # these are the default sizes for Omniglot:
    imgwidth = original_size
    imgheight = original_size

    logging.info('Reading the data.')
    images, labels, info = crawl_directory(directory,
                                           augment_with_rotations=rotate,
                                           first_label=first_label)

    images_np = np.zeros([len(images), imgwidth, imgheight], dtype=np.bool)
    labels_np = np.zeros([len(labels)], dtype=np.uint32)
    for i in xrange(len(images)):
        images_np[i, :, :] = images[i]
        labels_np[i] = labels[i]

    if resize:
        logging.info('Resizing images.')
        resized_images = resize_images(images_np, new_width, new_height)

        logging.info('Writing resized data in float32 format.')
        data = {'images': resized_images,
                'labels': labels_np,
                'info': info}
        with tf.gfile.GFile(write_file, 'w') as f:
            pickle.dump(data, f)
    else:
        logging.info('Writing original sized data in boolean format.')
        data = {'images': images_np,
                'labels': labels_np,
                'info': info}
        with tf.gfile.GFile(write_file, 'w') as f:
            pickle.dump(data, f)

    return len(np.unique(labels_np))


def maybe_download_data(repo_dir, repo_location, data_dir, train_dir, test_dir):
    """Download Omniglot repo if it does not exist."""
    if os.path.exists(repo_dir):
        logging.info('It appears that Git repo already exists.')
    else:
        logging.info('It appears that Git repo does not exist.')
        logging.info('Cloning now.')

        subprocess.check_output('git clone %s' % repo_location, shell=True)

    if os.path.exists(train_dir):
        logging.info('It appears that train data has already been unzipped.')
    else:
        logging.info('It appears that train data has not been unzipped.')
        logging.info('Unzipping now.')

        subprocess.check_output('unzip %s.zip -d %s' % (train_dir, data_dir),
                                shell=True)

    if os.path.exists(test_dir):
        logging.info('It appears that test data has already been unzipped.')
    else:
        logging.info('It appears that test data has not been unzipped.')
        logging.info('Unzipping now.')

        subprocess.check_output('unzip %s.zip -d %s' % (test_dir, data_dir),
                                shell=True)


def preprocess_omniglot(repo_dir, repo_location, data_dir, train_dir, test_dir, original_size=105, new_size=28, train_rotations=True, test_rotations=False):
    """Download and prepare raw Omniglot data.

    Downloads the data from GitHub if it does not exist.
    Then load the images, augment with rotations if desired.
    Resize the images and write them to a pickle file.
    """

    maybe_download_data(repo_dir, repo_location, data_dir, train_dir, test_dir)

    write_file = os.path.join(data_dir, 'train_omni.pkl')
    num_labels = write_datafiles(
        train_dir, write_file, resize=True, rotate=train_rotations, original_size=original_size,
        new_width=new_size, new_height=new_size)

    write_file = os.path.join(data_dir, 'test_omni.pkl')
    write_datafiles(
        test_dir, write_file, resize=True, rotate=test_rotations, original_size=original_size,
        new_width=new_size, new_height=new_size, first_label=num_labels)


def main(data_dir):
    logging.basicConfig(level=logging.INFO)
    main_dir = ''
    repo_location = 'https://github.com/brendenlake/omniglot.git'
    repo_dir = os.path.join(main_dir, 'omniglot')
    train_dir = os.path.join(repo_dir, 'python', 'images_background')
    test_dir = os.path.join(repo_dir, 'python', 'images_evaluation')
    preprocess_omniglot(repo_dir, repo_location, data_dir, train_dir, test_dir)


if __name__ == '__main__':
    main(sys.argv[1])
