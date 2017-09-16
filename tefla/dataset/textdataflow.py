from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random


import numpy as np

import six
from six.moves import zip
from six.moves import xrange
import tensorflow as tf


class TextDataflow(object):
    """Dataflow handling class

    Args:
        dataset: an instance of the dataset class
        num_readers: num of readers to  read the dataset
        shuffle: a bool, shuffle the dataset
        num_epochs: total number of epoch for training or validation
        min_queue_examples: minimum number of items after dequeue
        capacity: total queue capacity
    """

    def __init__(self, dataset, decoder, shuffle=True, num_threads=1, num_epochs=None, capacity=2048, max_length=None, min_bucket_length=8, length_bucket_step=1.1, shard_multiplier=1, length_multiplier=1):
        self.dataset = dataset
        self.decoder = decoder
        self.shuffle = shuffle
        self.capacity = capacity
        self.num_threads = num_threads
        self.max_length = max_length
        self.min_bucket_length = min_bucket_length
        self.length_bucket_step = length_bucket_step
        self.shard_multiplier = shard_multiplier
        self.length_multiplier = length_multiplier

    def get_batch(self, batch_size, preprocessing_fn=None, mode='training', drop_long_sequences=False):
        """Input pipeline, returns a dictionary of batched and padded tensors.

        Args:
          problem: Problem instance for which to build the input pipeline.
          data_file_pattern: file pattern for input files.
          capacity: int, data pipeline buffer capacity.
          mode: tf.contrib.learn.ModeKeys entry.
          batching_scheme: a dictionary containing
            "boundaries": a list of integers for the boundaries that will be
              used for bucketing; see bucket_by_sequence_length for more details.
            "batch_sizes": a list of batch sizes corresponding to the buckets
            "max_length": an integer.  We drop sequences which are longer.

        Returns:
          dict <feature name, batched and padded Tensor>
        """
        data_file_patterns = self.dataset.get_data_filepatterns(mode=mode)
        dataset_r = self.decoder.examples_reader(
            [data_file_patterns], bool(mode == 'training'), self.capacity)
        batching_scheme = self._batching_scheme(
            batch_size=batch_size,
            max_length=self.max_length,
            min_length_bucket=self.min_bucket_length,
            length_bucket_step=self.length_bucket_step,
            drop_long_sequences=drop_long_sequences,
            shard_multiplier=self.shard_multiplier,
            length_multiplier=self.length_multiplier)

        with tf.name_scope("input_pipeline"):
            if preprocessing_fn is not None:
                dataset_r = dataset_r.map(
                    lambda ex: preprocessing_fn(
                        ex, mode),
                    num_threads=self.num_threads)
            dataset_r = dataset_r.filter(
                lambda ex: self._example_too_big(ex, batching_scheme["max_length"]))

            dataset_r = self.bucket_by_sequence_length(dataset_r, self._example_length,
                                                       batching_scheme["boundaries"],
                                                       batching_scheme["batch_sizes"],
                                                       batching_scheme["window_size"])
            # We reshuffle the batches to prevent many long-sequence batches at once.
            if batching_scheme["shuffle_queue_size"] is not None:
                dataset_r = dataset_r.shuffle(
                    batching_scheme["shuffle_queue_size"])
            batched_examples = dataset_r.make_one_shot_iterator().get_next()
            return batched_examples

    def _example_length(self, example):
        length = 0
        # Length of the example is the maximum length of the feature lengths
        for v in example.values():
            # For images the sequence length is the size of the spatial dimensions.
            feature_length = (tf.shape(v)[0] if len(v.get_shape()) < 3 else
                              tf.shape(v)[0] * tf.shape(v)[1])
            length = tf.maximum(length, feature_length)
        return length

    def _example_too_big(self, example, max_length):
        return tf.less_equal(self._example_length(example), max_length)

    def bucket_by_sequence_length(self, dataset, example_length_fn, bucket_boundaries,
                                  bucket_batch_sizes, window_size):
        """Bucket entries in dataset by length.

        Args:
          dataset: Dataset of dict<feature name, Tensor>.
          example_length_fn: function from example to int, determines the length of
            the example, which will determine the bucket it goes into.
          bucket_boundaries: list<int>, boundaries of the buckets.
          bucket_batch_sizes: list<int>, batch size per bucket.
          window_size: an integer divisible by all elements of bucket_batch_sizes

        Returns:
          Dataset of padded and batched examples.
        """
        with tf.name_scope("bucket_by_seq_length"):

            def example_to_bucket_id(example):
                """Return int64 id of the length bucket for this example."""
                seq_length = example_length_fn(example)

                boundaries = list(bucket_boundaries)
                buckets_min = [np.iinfo(np.int32).min] + boundaries
                buckets_max = boundaries + [np.iinfo(np.int32).max]
                conditions_c = tf.logical_and(
                    tf.less_equal(buckets_min, seq_length),
                    tf.less(seq_length, buckets_max))
                bucket_id = tf.reduce_min(tf.where(conditions_c))

                return bucket_id

            def batching_fn(bucket_id, grouped_dataset):
                batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
                batch_size = batch_sizes[bucket_id]

                # Pad each dimension of each feature so that they match.
                padded_shapes = dict(
                    [(name, [None] * len(shape))
                     for name, shape in grouped_dataset.output_shapes.items()])
                return grouped_dataset.padded_batch(batch_size, padded_shapes)

            dataset = dataset.group_by_window(example_to_bucket_id, batching_fn,
                                              window_size)
            return dataset

    def _bucket_boundaries(self, max_length, min_length=8, length_bucket_step=1.1):
        """A default set of length-bucket boundaries."""
        assert min_length <= max_length
        assert length_bucket_step > 1.0
        x = min_length
        boundaries = []
        while x < max_length:
            boundaries.append(x)
            x = max(x + 1, int(x * length_bucket_step))
        return boundaries

    def _batching_scheme(self, batch_size,
                         max_length,
                         min_length_bucket,
                         length_bucket_step,
                         drop_long_sequences=False,
                         shard_multiplier=1,
                         length_multiplier=1):
        """A batching scheme based on model hyperparameters.

        Every batch containins a number of sequences divisible by `shard_multiplier`.

        Args:
          batch_size: int, total number of tokens in a batch.
          max_length: int, sequences longer than this will be skipped. Defaults to
            batch_size.
          min_length_bucket: int
          length_bucket_step: float greater than 1.0
          drop_long_sequences: bool, if True, then sequences longer than
            `max_length` are dropped.  This prevents generating batches with
            more than the usual number of tokens, which can cause out-of-memory
            errors.
          shard_multiplier: an integer increasing the batch_size to suit splitting
            across datashards.
          length_multiplier: an integer multiplier that is used to increase the
            batch sizes and sequence length tolerance.

        Returns:
           A dictionary with parameters that can be passed to input_pipeline:
             * boundaries: list of bucket boundaries
             * batch_sizes: list of batch sizes for each length bucket
             * max_length: int, maximum length of an example
        """
        max_length = max_length or batch_size
        boundaries = self._bucket_boundaries(
            max_length, min_length_bucket, length_bucket_step)
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length *= length_multiplier
        batch_sizes = [
            max(1, batch_size // length) for length in boundaries + [max_length]
        ]
        max_batch_size = max(batch_sizes)
        # Since the Datasets API only allows a single constant for window_size,
        # and it needs divide all bucket_batch_sizes, we pick a highly-compoisite
        # window size and then round down all batch sizes to divisors of that window
        # size, so that a window can always be divided evenly into batches.
        # TODO: remove this when Dataset API improves.
        highly_composite_numbers = [
            1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
            2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
            83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
            720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
            7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
            36756720, 43243200, 61261200, 73513440, 110270160]
        window_size = max([
            i for i in highly_composite_numbers if i <= 3 * max_batch_size])
        divisors = [i for i in xrange(
            1, window_size + 1) if window_size % i == 0]
        batch_sizes = [max([d for d in divisors if d <= bs])
                       for bs in batch_sizes]
        window_size *= shard_multiplier
        batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
        max_batches_per_window = window_size // min(batch_sizes)
        shuffle_queue_size = max_batches_per_window * 3
        ret = {
            "boundaries": boundaries,
            "batch_sizes": batch_sizes,
            "max_length": (max_length if drop_long_sequences else 10**9),
            "shuffle_queue_size": shuffle_queue_size,
            "window_size": window_size,
        }
        return ret

    def constant_batching_scheme(self, constant_batch_size_in_sequences):
        """A batching scheme with constant batch size.

        Args:
          constant_batch_size_in_sequences: an integer

        Returns:
           a dictionary
        """
        boundaries = self._bucket_boundaries(1024)
        batch_sizes = [constant_batch_size_in_sequences] * (1 + len(boundaries))
        return {
            "boundaries": boundaries,
            "batch_sizes": batch_sizes,
            "max_length": 10**9,
            "shuffle_queue_size": None,
            "window_size": constant_batch_size_in_sequences,
        }
