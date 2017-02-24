# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf


class Reader(object):
    """TFrecords reader class

    Args:
        dataset: an instance of the dataset class
        reader_kwargs: extra arguments to be passed to the TFRecordReader
        shuffle: whether to shuffle the dataset
        num_readers:a int, num of readers to launch
        capacity: a int, capacity of the queue used
        num_epochs: a int, num of epochs for training or validation

    """

    def __init__(self, dataset, reader_kwargs=None, shuffle=True, num_readers=16, capacity=1, num_epochs=None,):
        reader_kwargs = reader_kwargs or {}
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.capacity = capacity
        self._readers = [self.dataset.reader_class(
            **reader_kwargs) for _ in range(num_readers)]

    @property
    def num_readers(self):
        """Returns the number of readers"""
        return len(self._readers)

    def single_reader(self, num_epochs=1, shuffle=False, capacity=1):
        """Single record reader

            Data will be read using single TFRecordReader, primarily used for validation

        Args:
            num_epochs: number of epoch
            shuffle: shuffle the dataset. False for validation
            capacity: queue capacity

        Returns:
            a single item from the tfrecord files
        """
        with tf.name_scope('single_reader'):
            data_files = self.dataset.data_files()
            filename_queue = tf.string_input_producer(
                data_files, num_epochs=num_epochs, shuffle=shuffle, capacity=capacity)
            # return key, value
            _, value = self._reader.read(filename_queue)
            return value

    def parallel_reader(self, min_queue_examples=1024):
        """Parallel record reader

            Primarily used for Training ops

        Args:
            min_queue_examples: min number of queue examples after dequeue

        Returns
            a single item from the tfrecord files
        """
        with tf.name_scope('parallel_reader'):
            data_files = self.dataset.data_files()
            filename_queue = tf.train.string_input_producer(
                data_files, num_epochs=self.num_epochs, shuffle=self.shuffle)
            if self.shuffle:
                examples_queue = tf.RandomShuffleQueue(
                    capacity=self.capacity, min_after_dequeue=min_queue_examples, dtypes=[tf.string])
            else:
                examples_queue = tf.FIFOQueue(
                    capacity=self.capacity, dtypes=[tf.string])

            enqueue_ops = []
            for _reader in self._readers:
                _, value = _reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))
            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            return examples_queue.dequeue()
