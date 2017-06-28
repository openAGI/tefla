# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import threading
from datetime import datetime
import tensorflow as tf
import glob
import numpy as np
from PIL import Image
import random
from .decoder import ImageCoder


class TFRecords(object):
    """TFRecords

    Converts image data to tfrecords file.

    """

    def __init__(self, batch_size=128):
        self.batch_size = batch_size

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def is_jpg(self, filename):
        """Determine if a file contains a JPG format image.

        Args:
            filename: string, path of the image file.

        Returns:
            boolean indicating if the image is a JPG.
        """
        return self._is_jpg(filename)

    def _is_jpg(self, filename):
        data = open(filename, 'rb').read(11)
        if data[:4] != '\xff\xd8\xff\xe0':
            return False
        if data[6:] != 'JFIF\0':
            return False
        return True

    def is_png(self, filename):
        """Determine if a file contains a PNG format image.

        Args:
            filename: string, path of the image file.

        Returns:
            boolean indicating if the image is a PNG.
        """
        return self._is_png(filename)

    def _is_png(self, filename):
        return '.png' in filename

    def label_to_string(self, label):
        return{
            0: 'No DR',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
            4: 'Proliferative DR',
        }[label]

    def process_image(self, filename, coder):
        """Process a single image file.

        Args:
            filename: string, path to an image file e.g., '/path/to/example.JPG'.
            coder: instance of ImageCoder to provide TensorFlow image coding utils.

        Returns:
            image_buffer: string, JPEG encoding of RGB image.
            height: integer, image height in pixels.
            width: integer, image width in pixels.
        """
        # Read the image file.
        image_data = tf.gfile.FastGFile(filename, 'r').read()

        # Convert any PNG to JPEG's for consistency.
        if not self._is_jpg(filename):
            print('Converting PNG to JPEG for %s' % filename)
            image_data = coder.png_to_jpeg(image_data)

        # Decode the RGB JPEG.
        image = coder.decode_jpeg(image_data)

        # Check that image converted to RGB
        # image = tf.image.resize_images(image, cfg.TRAIN.im_height, cfg.TRAIN.im_width, method=3, align_corners=False)
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3

        return image_data, height, width

    def convert_to_example(self, filename, image_buffer, label, text, height, width, image_format='jpg', colorspace='RGB', channels=3):
        """Build an Example proto for an example.

        Args:
            filename: string, path to an image file, e.g., '/path/to/example.JPG'
            image_buffer: string, JPEG encoding of RGB image
            label: integer, identifier for the ground truth for the network
            text: string, unique human-readable, e.g. 'mild'
            height: integer, image height in pixels
            width: integer, image width in pixels
            image_format: string, image format, e.g.: '.jpg'
            colorspace: image colorspace
            channels: number of channels in the image, e.g. 3 for RGB

        Returns:
            Example proto
        """

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': self._int64_feature(height),
            'image/width': self._int64_feature(width),
            'image/colorspace': self._bytes_feature(colorspace),
            'image/channels': self._int64_feature(channels),
            'image/class/label': self._int64_feature(label),
            'image/class/text': self._bytes_feature(text),
            'image/format': self._bytes_feature(image_format),
            'image/filename': self._bytes_feature(bytes(os.path.basename(filename))),
            'image/encoded/image': self._bytes_feature(image_buffer)}))
        return example

    def process_image_files_batch(self, coder, thread_index, ranges, name, filenames, texts, labels, num_shards, train_dir):
        """Processes and saves list of images as TFRecord in 1 thread.

        Args:
            coder: instance of ImageCoder to provide TensorFlow image coding utils.
            thread_index: integer, unique batch to run index is within [0, len(ranges)).
            ranges: list of pairs of integers specifying ranges of each batches to
            analyze in parallel.
            name: string, unique identifier specifying the data set
            filenames: list of strings; each string is a path to an image file
            texts: list of strings; each string is human readable, e.g. 'mild'
            labels: list of integer; each integer identifies the ground truth
            num_shards: integer number of shards for this data set.
        """
        # Each thread produces N shards where N = int(num_shards / num_threads).
        # For instance, if num_shards = 128, and the num_threads = 2, then the first
        # thread would produce shards [0, 64).
        num_threads = len(ranges)
        assert not num_shards % num_threads
        num_shards_per_batch = int(num_shards / num_threads)

        shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][
                                   1], num_shards_per_batch + 1).astype(int)
        num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
        counter = 0
        for s in range(num_shards_per_batch):
            # Generate a sharded version of the file name, e.g.
            # 'train-00002-of-00010'
            shard = thread_index * num_shards_per_batch + s
            output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
            output_file = os.path.join(train_dir, output_filename)
            writer = tf.python_io.TFRecordWriter(output_file)
            print('processing')
            shard_counter = 0
            files_in_shard = np.arange(
                shard_ranges[s], shard_ranges[s + 1], dtype=int)
            for i in files_in_shard:
                filename = filenames[i] + '.jpg'
                label = labels[i]
                text = texts[i]
                image_buffer, height, width = self.process_image(
                    filename, coder)
                example = self.convert_to_example(
                    filename, image_buffer, label, text, height, width)
                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1
                print('Num of files %d' % (counter))
                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch.' % (
                        datetime.now(), thread_index, counter, num_files_in_thread))
                    sys.stdout.flush()

            print('%s [thread %d]: Wrote %d images to %s' %
                  (datetime.now(), thread_index, shard_counter, output_file))
            sys.stdout.flush()
            shard_counter = 0
        print('%s [thread %d]: Wrote %d images to %d shards.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    def process_image_files(self, name, filenames, texts, labels, num_shards, output_dir, num_threads=4):
        """Process and save list of images as TFRecord of Example protos.

        Args:
            name: string, unique identifier specifying the data set
            filenames: list of strings; each string is a path to an image file
            texts: list of strings; each string is human readable, e.g. 'dog'
            labels: list of integer; each integer identifies the ground truth
            num_shards: integer number of shards for this data set.
        """
        assert len(filenames) == len(texts)
        assert len(filenames) == len(labels)

        # Break all images into batches with a [ranges[i][0], ranges[i][1]].
        spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
        ranges = []
        threads = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])

        # Launch a thread for each batch.
        print('Launching %d threads for spacings: %s' % (num_threads, ranges))
        sys.stdout.flush()
        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()
        # Create a generic TensorFlow-based utility for converting all image
        # codings.
        coder = ImageCoder()
        threads = []
        for thread_index in range(len(ranges)):
            args = (coder, thread_index, ranges, name,
                    filenames, texts, labels, num_shards, output_dir)
            t = threading.Thread(
                target=self.process_image_files_batch, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print('%s: Finished writing all %d images in data set.' %
              (datetime.now(), len(filenames)))
        sys.stdout.flush()

    def find_image_files(self, data_dir, labels_file):
        """Build a list of all images files and labels in the data set.

        Args:
            data_dir: string, path to the root directory of images.
                Assumes that the image data set resides in JPEG files located in
                the following directory structure.
                data_dir/No DR/another-image.JPEG
                data_dir/No DR/my-image.jpg
                where 'No DR' is the label associated with these images.
            labels_file: string, path to the labels file.
                The list of valid labels are held in this file. Assumes that the file
                contains entries as such:
                No DR
                Mild
                Severe
                where each line corresponds to a label. We map each label contained in
                the file to an integer starting with the integer 0 corresponding to the
                label contained in the first line.

        Returns:
            filenames: list of strings; each string is a path to an image file.
            texts: list of strings; each string is the class, e.g. 'dog'
            labels: list of integer; each integer identifies the ground truth.
        """
        print('Determining list of input files and labels from %s.' % data_dir)
        unique_labels = [l.strip()
                         for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]

        labels = []
        filenames = []
        texts = []

        # Construct the list of JPEG files and labels.
        for text in unique_labels:
            file_label = text.split(',')
            jpeg_file_path = os.path.join(data_dir, file_label[0])
            filenames.append(jpeg_file_path)
            labels.append(int(file_label[1]))
            texts.append(self.label_to_string(int(file_label[1])))

        print('text len %d' % (len(texts)))
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        shuffled_index = range(len(filenames))
        random.seed(12345)
        random.shuffle(shuffled_index)

        filenames = [filenames[i] for i in shuffled_index]
        texts = [texts[i] for i in shuffled_index]
        labels = [labels[i] for i in shuffled_index]

        print('Found %d JPEG files across %d labels inside %s.' %
              (len(filenames), len(unique_labels), data_dir))
        return filenames, texts, labels

    def process_dataset(self, name, directory, output_directory, num_shards, labels_file):
        """Process a complete data set and save it as a TFRecord.

        Args:
            name: string, unique identifier specifying the data set.
            directory: string, root path to the data set.
            num_shards: integer number of shards for this data set.
            labels_file: string, path to the labels file.
        """
        filenames, texts, labels = self.find_image_files(directory, labels_file)
        self.process_image_files(name, filenames, texts, labels, num_shards, output_directory)

    def read_images_from(self, data_dir, imresize=[512, 512]):
        images = []
        jpeg_files_path = glob.glob(
            os.path.join(data_dir, '*.[jJ][pP][eE][gG]'))
        for filename in jpeg_files_path[:10]:
            im = Image.open(filename)
            im = im.resize((imresize[0], imresize[1]), Image.ANTIALIAS)
            im = np.asarray(im, np.uint8)
            images.append(im)

        # Use unint8 or you will be !!!
        images_only = [np.asarray(image, np.uint8) for image in images]
        images_only = np.array(images_only)

        print(images_only.shape)
        return images_only


if __name__ == '__main__':
    # Convert Images to tfRecords files
    im2r = TFRecords()
    im2r.process_dataset('retina_dr_train_256',
                         '/path/to/Data/train', 'path/to/output/dir', 16, '/path/to/Data/label.txt')
