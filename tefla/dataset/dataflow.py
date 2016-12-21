# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf
from tefla.dataset.reader import Reader

balanced_sample = tf.contrib.training.stratified_sample


class Dataflow(object):

    def __init__(self, dataset, num_readers=1, shuffle=True, num_epochs=None, min_queue_examples=1024, capacity=2048):
        self.min_queue_examples = min_queue_examples
        self.num_readers = num_readers
        self.shuffle = shuffle
        self.dataset=dataset
        self.reader = Reader(dataset, shuffle=shuffle, num_readers=num_readers, capacity=capacity, num_epochs=num_epochs)

    def get(self, items, image_size, resize_size=None):
        # image_size [width, height, channel]
        # resize_size [width, height]
        if self.num_readers == 1 and not self.shuffle:
            data = self.reader.single_reader()
        else:
            data = self.reader.parallel_reader(self.min_queue_examples)
        outputs = self.dataset.decoder.decode(data, image_size, resize_size=resize_size)
        valid_items = outputs.keys()
        self._validate_items(items, valid_items)
        return [outputs[item] for item in items]

    def get_batch(self, batch_size, target_probs, image_size, resize_size=None,  init_probs=None, enqueue_many=False, queue_capacity=2048, threads_per_queue=1, name='balancing_op'):
        image, label = self.get(['image', 'label'], image_size, resize_size)
        [data_batch], label_batch = balanced_sample([image], label, target_probs, batch_size, init_probs=init_probs,
            enqueue_many=enqueue_many, queue_capacity=queue_capacity, threads_per_queue=threads_per_queue, name=name)
        return data_batch, label_batch

    def _validate_items(self, items, valid_items):
        if not isinstance(items, (list, tuple)):
            raise ValueError('items must be a list or tuple')

        for item in items:
            if item not in valid_items:
                raise ValueError('Item [%s] is invalid. Valid entries include: %s' % (item, valid_items))
