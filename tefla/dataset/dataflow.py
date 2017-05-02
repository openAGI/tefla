# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf
from .reader import Reader

balanced_sample = tf.contrib.training.stratified_sample


class Dataflow(object):
    """Dataflow handling class

    Args:
        dataset: an instance of the dataset class
        num_readers: num of readers to  read the dataset
        shuffle: a bool, shuffle the dataset
        num_epochs: total number of epoch for training or validation
        min_queue_examples: minimum number of items after dequeue
        capacity: total queue capacity
    """

    def __init__(self, dataset, num_readers=1, shuffle=True, num_epochs=None, min_queue_examples=1024, capacity=2048):
        self.min_queue_examples = min_queue_examples
        self.num_readers = num_readers
        self.shuffle = shuffle
        self.dataset = dataset
        self.reader = Reader(dataset, shuffle=shuffle, num_readers=num_readers,
                             capacity=capacity, num_epochs=num_epochs)

    def get(self, items, image_size, resize_size=None):
        """ Get a single example from the dataset

        Args:
            items: a list, with items to get from the dataset
                e.g.: ['image', 'label']
            image_size: a list with original image size
                e.g.: [width, height, channel]
            resize_size: if image resize required, provide a list of width and height
                e.g.: [width, height]

        """
        if self.num_readers == 1 and not self.shuffle:
            data = self.reader.single_reader()
        else:
            data = self.reader.parallel_reader(self.min_queue_examples)
        outputs = self.dataset.decoder.decode(
            data, image_size, resize_size=resize_size)
        valid_items = outputs.keys()
        self._validate_items(items, valid_items)
        return [outputs[item] for item in items]

    def get_batch(self, batch_size, target_probs, image_size, resize_size=None,  crop_size=[32, 32, 3], init_probs=None, enqueue_many=True, queue_capacity=2048, threads_per_queue=1, name='balancing_op'):
        """ Get a batch of examplea from the dataset

        Stochastically creates batches based on per-class probabilities.
        This method discards examples. Internally, it creates one queue to
        amortize the cost of disk reads, and one queue to hold the properly-proportioned batch.

        Args:
            batch_size: a int, batch_size
            target_probs: probabilities of class samples to be present in the batch
            image_size: a list with original image size
                e.g.: [width, height, channel]
            resize_size: if image resize required, provide a list of width and height
                e.g.: [width, height]
            init_probs: initial probs of data sample in the first batch
            enqueue_many: bool, if true, interpret input tensors as having a batch dimension.
            queue_capacity: Capacity of the large queue that holds input examples.
            threads_per_queue: Number of threads for the large queue that holds
                input examples and for the final queue with the proper class proportions.
            name: a optional scope/name of the op

        """
        if enqueue_many:
            image, label = self.batch_inputs(batch_size, True, image_size, crop_size,
                                             im_size=resize_size, bbox=None, image_preprocessing=None, num_preprocess_threads=4)
        else:
            image, label = self.get(['image', 'label'], image_size, resize_size)
        [data_batch], label_batch = balanced_sample([image], label, target_probs, batch_size, init_probs=init_probs,
                                                    enqueue_many=enqueue_many, queue_capacity=queue_capacity, threads_per_queue=threads_per_queue, name=name)
        return data_batch, label_batch

    # TODO need refinements
    def batch_inputs(self, batch_size, train, tfrecords_image_size, crop_size, im_size=None, bbox=None, image_preprocessing=None, num_preprocess_threads=16):
        """Contruct batches of training or evaluation examples from the image dataset.

        Args:
            dataset: instance of Dataset class specifying the dataset.
            See dataset.py for details.
            batch_size: integer
            train: boolean
            crop_size: training time image size. a int or tuple
            tfrecords_image_size: a list with original image size used to encode image in tfrecords
                e.g.: [width, height, channel]
            image_processing: a function to process image
            num_preprocess_threads: integer, total number of preprocessing threads

        Returns:
            images: 4-D float Tensor of a batch of images
            labels: 1-D integer Tensor of [batch_size].

        Raises:
            ValueError: if data is not found
        """
        with tf.name_scope('batch_processing'):
            if num_preprocess_threads % 4:
                raise ValueError('Please make num_preprocess_threads a multiple '
                                 'of 4 (%d % 4 != 0).', num_preprocess_threads)
            images_and_labels = []
            for thread_id in range(num_preprocess_threads):
                image, label = self.get(
                    ['image', 'label'], tfrecords_image_size, im_size)
                if image_preprocessing is not None:
                    image = image_preprocessing(
                        image, train, crop_size, im_size, thread_id, bbox)
                images_and_labels.append([image, label])
            images, label_index_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size,
                                                            capacity=2 * num_preprocess_threads * batch_size)

            # Reshape images into these desired dimensions.
            depth = 3
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size)

            images = tf.cast(images, tf.float32)
            images = tf.reshape(
                images, shape=[batch_size, crop_size[0], crop_size[1], depth])

            return images, tf.reshape(label_index_batch, [batch_size])

    def _validate_items(self, items, valid_items):
        if not isinstance(items, (list, tuple)):
            raise ValueError('items must be a list or tuple')

        for item in items:
            if item not in valid_items:
                raise ValueError(
                    'Item [%s] is invalid. Valid entries include: %s' % (item, valid_items))

    @property
    def n_iters_per_epoch(self):
        return self.dataset.n_iters_per_epoch
