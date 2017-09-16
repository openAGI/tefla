from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import random


import numpy as np
from six.moves import xrange
import tensorflow as tf
from tefla.dataset.textdataset import TextDataset
from tefla.dataset.textdecoder import TextDecoder
from tefla.dataset.textdataflow import TextDataflow


class TestDataset(TextDataset):

    def __init__(self, data_dir, vocab_name='testvocab', dataset_name='test'):
        super(TestDataset, self).__init__(data_dir, vocab_name, dataset_name)

    def generator(self, data_dir, tmp_dir, is_training):
        for i in xrange(30):
            yield {"inputs": [i] * (i + 1), "targets": [i], "floats": [i + 0.5]}

    @property
    def num_shards(self):
        return 1

    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64),
            "floats": tf.VarLenFeature(tf.float32),
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)


def generate_test_data(data_dir, tmp_dir):
    testdata = TestDataset(data_dir)
    print('generrating data')
    testdata.generate_data(tmp_dir)
    filepatterns = testdata.get_data_filepatterns(mode='both')
    assert tf.gfile.Glob(filepatterns[0])
    return filepatterns, testdata


class DataReaderTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.set_random_seed(1)
        cls.filepatterns, testdata = generate_test_data(
            '/tmp', '/tmp')
        cls.decoder = TextDecoder(testdata)
        cls.dataflow = TextDataflow(testdata, cls.decoder)

    def testBatchingSchemeMaxLength(self):
        scheme = self.dataflow._batching_scheme(
            batch_size=20, max_length=None,
            min_length_bucket=8, length_bucket_step=1.1,
            drop_long_sequences=False)
        self.assertGreater(scheme["max_length"], 10000)

        scheme = self.dataflow._batching_scheme(
            batch_size=20, max_length=None,
            min_length_bucket=8, length_bucket_step=1.1,
            drop_long_sequences=True)
        self.assertEqual(scheme["max_length"], 20)

        scheme = self.dataflow._batching_scheme(
            batch_size=20, max_length=15,
            min_length_bucket=8, length_bucket_step=1.1,
            drop_long_sequences=True)
        self.assertEqual(scheme["max_length"], 15)

        scheme = self.dataflow._batching_scheme(
            batch_size=20, max_length=15,
            min_length_bucket=8, length_bucket_step=1.1,
            drop_long_sequences=False)
        self.assertGreater(scheme["max_length"], 10000)

    def testBatchingSchemeBuckets(self):
        scheme = self.dataflow._batching_scheme(
            batch_size=128,
            max_length=0,
            min_length_bucket=8,
            length_bucket_step=1.1)
        boundaries, batch_sizes = scheme["boundaries"], scheme["batch_sizes"]
        self.assertEqual(len(boundaries), len(batch_sizes) - 1)
        expected_boundaries = [
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28,
            30, 33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124]
        self.assertEqual(expected_boundaries, boundaries)
        expected_batch_sizes = [
            16, 12, 12, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 4, 4, 4, 4, 4, 3, 3, 3,
            3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(expected_batch_sizes, batch_sizes)

    def testBucketBySeqLength(self):

        def example_len(ex):
            return tf.shape(ex["inputs"])[0]

        boundaries = [10, 20, 30]
        batch_sizes = [10, 8, 4, 2]
        window_size = 40

        dataset_r = self.decoder.examples_reader(
            self.filepatterns[0],
            False, 32)
        dataset = self.dataflow.bucket_by_sequence_length(
            dataset_r, example_len,
            boundaries, batch_sizes, window_size)
        batch = dataset.make_one_shot_iterator().get_next()
        input_vals = []
        obs_batch_sizes = []
        with tf.train.MonitoredSession() as sess:
            # Until OutOfRangeError
            for i in xrange(60):
                batch_val = sess.run(batch)
                batch_inputs = batch_val["inputs"]
                batch_size, max_len = batch_inputs.shape
                obs_batch_sizes.append(batch_size)
                for inputs in batch_inputs:
                    input_val = inputs[0]
                    input_vals.append(input_val)
                    repeat = input_val + 1
                    self.assertAllEqual([input_val] * repeat + [0] * (max_len - repeat),
                                        inputs)

        self.assertEqual(list(range(30)), sorted(input_vals))
        self.assertTrue(len(set(obs_batch_sizes)) > 1)


if __name__ == '__main__':
    tf.test.main()
