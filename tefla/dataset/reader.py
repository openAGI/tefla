# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf


class Reader(object):

    def __init__(self, dataset, reader_kwargs=None, shuffle=True, num_readers=16, capacity=1, num_epochs=None,):
        reader_kwargs = reader_kwargs or {}
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.capacity = capacity
        self._readers = [self.dataset.reader_class(**reader_kwargs) for _ in range(num_readers)]

    @property
    def num_readers(self):
        return len(self._readers)

    def single_reader(self):
        with tf.name_scope('single_reader'):
            data_files = self.dataset.data_files()
            filename_queue = tf.string_input_producer(data_files, num_epochs=1, shuffle=False, capacity=1)
            # return key, value
            _, value = self._reader.read(filename_queue)
            return value

    def parallel_reader(self, min_queue_examples=1024):
        with tf.name_scope('parallel_reader'):
            data_files = self.dataset.data_files()
            filename_queue = tf.train.string_input_producer(data_files, num_epochs=self.num_epochs, shuffle=self.shuffle)
            if self.shuffle:
                examples_queue = tf.RandomShuffleQueue(capacity=self.capacity, min_after_dequeue=min_queue_examples, dtypes=[tf.string])
            else:
                examples_queue = tf.FIFOQueue(capacity=self.capacity, dtypes=[tf.string])

            enqueue_ops = []
            for _reader in self._readers:
                _, value = _reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))
            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            return examples_queue.dequeue()
