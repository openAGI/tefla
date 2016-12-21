# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from tefla.dataset.reader import Reader


class Dataflow(object):

    def __init__(self, dataset, num_readers=1, shuffle=True, num_epochs=None, min_queue_examples=2024, capacity=1):
        self.min_queue_examples = min_queue_examples
        self.num_readers = num_readers
        self.shuffle = shuffle
        self.dataset=dataset
        self.reader = Reader(dataset, shuffle=shuffle, num_readers=num_readers, capacity=capacity, num_epochs=num_epochs)

    def get(self, items):
        if self.num_readers == 1 and not self.shuffle:
            data = self.reader.single_reader()
        else:
            data = self.reader.parallel_reader(self.min_queue_examples)
        outputs = self.dataset.decoder.decode(data)
        valid_items = outputs.keys()
        self._validate_items(items, valid_items)
        return [outputs[item] for item in items]

    def _validate_items(self, items, valid_items):
        if not isinstance(items, (list, tuple)):
            raise ValueError('items must be a list or tuple')

        for item in items:
            if item not in valid_items:
                raise ValueError('Item [%s] is invalid. Valid entries include: %s' % (item, valid_items))
