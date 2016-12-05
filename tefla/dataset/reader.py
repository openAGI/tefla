import tensorflow as tf


class Reader(object):
    def __init__(self, reader_class, reader_kwargs, num_readers=8):
        self.reader_kwargs = reader_kwargs or {}
        self._readers = [reader_class(**reader_kwargs) for _ in range(num_readers)]

    @property
    def num_readers(self):
        return len(self._readers)

    def single_reader(self, data_files):
        with tf.name_scope('single_reader'):
            filename_queue = tf.string_input_producer(data_files, num_epochs=1, shuffle=False, capacity=1)
            # return key, value
            _, value = self._reader.read(filename_queue)
            return value

    def parallel_reader(self, reader, data_files, num_epochs, shuffle=True, capacity=2048, num_readers=16, min_queue_examples=1024, batch_size=32):
        with tf.name_scope('parallel_reader'):
            filename_queue = tf.train.string_input_producer(data_files, num_epochs=num_epochs, shuffle=shuffle)
            if shuffle:
                examples_queue = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_queue_examples, dtypes=[tf.string])
            else:
                examples_queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.string])

            enqueue_ops = []
            for _reader in self._readers:
                _, value = self._reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))
            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            return examples_queue.dequeue()
