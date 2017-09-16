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

    def testBasicExampleReading(self):
        datasetreader = self.decoder.examples_reader(
            [self.filepatterns[0]], True, 32)
        examples = datasetreader.make_one_shot_iterator().get_next()
        with tf.train.MonitoredSession() as sess:
            for _ in xrange(10):
                ex_val = sess.run(examples)
                inputs, targets, floats = (ex_val["inputs"], ex_val["targets"],
                                           ex_val["floats"])
                self.assertEqual(np.int64, inputs.dtype)
                self.assertEqual(np.int64, targets.dtype)
                self.assertEqual(np.float32, floats.dtype)
                for field in [inputs, targets, floats]:
                    self.assertGreater(len(field), 0)

    def testTrainEvalBehavior(self):
        train_datasetreader = self.decoder.examples_reader(
            [self.filepatterns[0]], True, 16)
        train_examples = train_datasetreader.make_one_shot_iterator().get_next()
        eval_datasetreader = self.decoder.examples_reader(
            [self.filepatterns[1]], False, 16)
        eval_examples = eval_datasetreader.make_one_shot_iterator().get_next()

        eval_idxs = []
        with tf.train.MonitoredSession() as sess:

            # Eval should not be shuffled and only run through once
            for i in xrange(30):
                self.assertEqual(i, sess.run(eval_examples)["inputs"][0])
                eval_idxs.append(i)

            with self.assertRaises(tf.errors.OutOfRangeError):
                sess.run(eval_examples)
                # Should never run because above line should error
                eval_idxs.append(30)

            # Ensuring that the above exception handler actually ran and we didn't
            # exit the MonitoredSession context.
            eval_idxs.append(-1)

        self.assertAllEqual(list(range(30)) + [-1], eval_idxs)


if __name__ == '__main__':
    tf.test.main()
