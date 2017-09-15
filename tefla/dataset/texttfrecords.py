from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import os
import random
import six
from six.moves import xrange

import tensorflow as tf

UNSHUFFLED_SUFFIX = "-unshuffled"


class TextTFRecord(object):

    def to_example(self, dictionary):
        """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
        features = {}
        for (k, v) in six.iteritems(dictionary):
            if not v:
                raise ValueError("Empty generated field: %s", str((k, v)))
            if isinstance(v[0], six.integer_types):
                features[k] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=v))
            elif isinstance(v[0], float):
                features[k] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=v))
            elif isinstance(v[0], six.string_types):
                if not six.PY2:  # Convert in python 3.
                    v = [bytes(x, "utf-8") for x in v]
                features[k] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=v))
            elif isinstance(v[0], bytes):
                features[k] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=v))
            else:
                raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                                 (k, str(v[0]), str(type(v[0]))))
        return tf.train.Example(features=tf.train.Features(feature=features))

    def generate_files_distributed(self, generator,
                                   output_name,
                                   output_dir,
                                   num_shards=1,
                                   max_cases=None,
                                   task_id=0):
        """generate_files but with a single writer writing to shard task_id."""
        assert task_id < num_shards
        output_filename = sharded_name(output_name, task_id, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        tf.logging.info("Writing to file %s", output_file)
        writer = tf.python_io.TFRecordWriter(output_file)

        counter = 0
        for case in generator:
            if counter % 100000 == 0:
                tf.logging.info("Generating case %d for %s." %
                                (counter, output_name))
            counter += 1
            if max_cases and counter > max_cases:
                break
            sequence_example = self.to_example(case)
            writer.write(sequence_example.SerializeToString())

        writer.close()
        return output_file

    def generate_files(self, generator, output_filenames, max_cases=None):
        """Generate cases from a generator and save as TFRecord files.

        Generated cases are transformed to tf.Example protos and saved as TFRecords
        in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

        Args:
          generator: a generator yielding (string -> int/float/str list) dictionaries.
          output_filenames: List of output file paths.
          max_cases: maximum number of cases to get from the generator;
            if None (default), we use the generator until StopIteration is raised.
        """
        num_shards = len(output_filenames)
        writers = [tf.python_io.TFRecordWriter(
            fname) for fname in output_filenames]
        counter, shard = 0, 0
        for case in generator:
            if counter > 0 and counter % 100000 == 0:
                tf.logging.info("Generating case %d." % counter)
            counter += 1
            if max_cases and counter > max_cases:
                break
            sequence_example = self.to_example(case)
            writers[shard].write(sequence_example.SerializeToString())
            shard = (shard + 1) % num_shards

        for writer in writers:
            writer.close()

    def read_records(self, filename):
        reader = tf.python_io.tf_record_iterator(filename)
        records = []
        for record in reader:
            records.append(record)
            if len(records) % 100000 == 0:
                tf.logging.info("read: %d", len(records))
        return records

    def write_records(self, records, out_filename):
        writer = tf.python_io.TFRecordWriter(out_filename)
        for count, record in enumerate(records):
            writer.write(record)
            if count > 0 and count % 100000 == 0:
                tf.logging.info("write: %d", count)
        writer.close()

    def generate_dataset_and_shuffle(self, train_gen,
                                     train_paths,
                                     dev_gen,
                                     dev_paths,
                                     shuffle=True):
        self.generate_files(train_gen, train_paths)
        self.generate_files(dev_gen, dev_paths)
        if shuffle:
            self.shuffle_dataset(train_paths)

    def shuffle_dataset(self, filenames):
        tf.logging.info("Shuffling data...")
        for fname in filenames:
            records = self.read_records(fname)
            random.shuffle(records)
            out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
            self.write_records(records, out_fname)
            tf.gfile.Remove(fname)
