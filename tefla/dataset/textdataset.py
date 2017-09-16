from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import os

import tensorflow as tf
from . import text_encoder
from .texttfrecords import TextTFRecord

UNSHUFFLED_SUFFIX = "-unshuffled"


@six.add_metaclass(abc.ABCMeta)
class TextDataset():

    def __init__(self, data_dir, vocab_name, dataset_name):
        self._vocab_name = vocab_name
        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self.tfrecords = TextTFRecord()

    @property
    def is_character_level(self):
        raise NotImplementedError()

    @property
    def has_inputs(self):
        return True

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def input_space_id(self):
        raise NotImplementedError()

    @property
    def target_space_id(self):
        raise NotImplementedError()

    @property
    def num_shards(self):
        raise NotImplementedError()

    @property
    def num_dev_shards(self):
        return 1

    @property
    def vocab_name(self):
        return self._vocab_name

    @property
    def vocab_file(self):
        return "%s.%d" % (self.vocab_name, self.targeted_vocab_size)

    @property
    def dataset_name(self):
        return self_dataset_name

    @property
    def use_subword_tokenizer(self):
        raise NotImplementedError()

    @property
    def targeted_vocab_size(self):
        raise NotImplementedError()

    @property
    def use_train_shards_for_dev(self):
        return False

    @abc.abstractmethod
    def generator(self, tmp_dir, train, *args, **kwargs):
        """Generator for lm1b sentences.

        Args:
          tmp_dir: a string.
          train: a boolean.
          characters: a boolean

        Yields:
          A dictionary {"inputs": [0], "targets": [<subword ids>]}
        """
        raise NotImplementedError()

    def feature_encoders(self):
        return {
            "inputs": text_encoder.TextEncoder(),
            "targets": text_encoder.TextEncoder()
        }

    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64)
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def generate_data(self, tmp_dir, task_id=-1):
        train_paths = self.training_filepaths(
            self.num_shards)
        dev_paths = self.dev_filepaths(
            self.num_dev_shards)
        if self.use_train_shards_for_dev:
            all_paths = train_paths + dev_paths
            self.tfrecords.generate_files(
                self.generator(self._data_dir, tmp_dir, True), all_paths)
            self.tfrecords.shuffle_dataset(train_paths)
        else:
            self.tfrecords.generate_dataset_and_shuffle(
                self.generator(self._data_dir, tmp_dir, True), train_paths,
                self.generator(self._data_dir, tmp_dir, False), dev_paths)

    def feature_encoders(self):
        if self.is_character_level:
            encoder = text_encoder.ByteTextEncoder()
        elif self.use_subword_tokenizer:
            vocab_filename = os.path.join(self._data_dir, self.vocab_file)
            encoder = text_encoder.SubwordTextEncoder(vocab_filename)
        else:
            vocab_filename = os.path.join(self._data_dir, self.vocab_file)
            encoder = text_encoder.TokenTextEncoder(vocab_filename)
        if self.has_inputs:
            return {"inputs": encoder, "targets": encoder}
        return {"targets": encoder}

    def training_filepaths(self, num_shards):
        return self.train_data_filenames(num_shards)

    def dev_filepaths(self, num_shards):
        return self.dev_data_filenames(num_shards)

    def test_filepaths(self, num_shards):
        return self.test_data_filenames(num_shards)

    def _data_filenames(self, output_name, output_dir, num_shards):
        return [
            os.path.join(output_dir, fname)
            for fname in self.shard_filepath(output_name, num_shards)
        ]

    def train_data_filenames(self, num_shards):
        return self._data_filenames(self._dataset_name + UNSHUFFLED_SUFFIX + "-train", self._data_dir, num_shards)

    def dev_data_filenames(self, num_shards):
        return self._data_filenames(self._dataset_name + "-dev", self._data_dir, num_shards)

    def test_data_filenames(self, num_shards):
        return self._data_filenames(self.dataset_name + "-test", self._data_dir, num_shards)

    def combined_data_filenames(self, output_dir, num_training_shards):
        return (self.train_data_filenames(output_dir, num_training_shards) +
                self.dev_data_filenames(output_dir, 1) + self.test_data_filenames(
                    output_dir, 1))

    def sharded_name(self, base_name, shard, total_shards):
        return "%s-%.5d-of-%.5d" % (base_name, shard, total_shards)

    def shard_filepath(self, fname, num_shards):
        return [
            self.sharded_name(fname, shard, num_shards) for shard in xrange(num_shards)
        ]

    def get_data_filepatterns(self, mode='training'):
        datasets = []
        data_dir = os.path.join(self._data_dir, self._dataset_name)
        if mode == 'training':
            datasets.append("%s-train*" % data_dir)
        elif mode == 'eval':
            datasets.append("%s-dev*" % data_dir)
        else:
            datasets.append("%s-train*" % data_dir)
            datasets.append("%s-dev*" % data_dir)
        return datasets

    def get_data_files(self, data_sources):
        """Get data_files from data_sources.

        Args:
            data_sources: a list/tuple of files or the location of the data, i.e.
            /path/to/train@128, /path/to/train* or /tmp/.../train*

        Returns:
            a list of data_files.

        Raises:
            ValueError: if not data files are not found
        """
        if isinstance(data_sources, (list, tuple)):
            data_files = []
            for source in data_sources:
                data_files += self.get_data_files(source)
        else:
            if '*' in data_sources or '?' in data_sources or '[' in data_sources:
                data_files = tf.gfile.Glob(data_sources)
            else:
                data_files = [data_sources]
        if not data_files:
            raise ValueError('No data files found in %s' % (data_sources,))
        return data_files


class SpaceID(object):
    """Input and target space ids. Add more as needed."""
    # Generic / unknown output space (default)
    GENERIC = 0
    # Image labels
    IMAGE_LABEL = 1
    # English characters
    EN_CHR = 2
    # English tokens
    EN_TOK = 3
    # English bpe tokens
    EN_BPE_TOK = 4
    # French characters
    FR_CHR = 5
    # French tokens
    FR_TOK = 6
    # German characters
    DE_CHR = 7
    # German tokens
    DE_TOK = 8
    # German bpe tokens
    DE_BPE_TOK = 9
    # Digit cipher lexicon 0
    DIGIT_0 = 10
    # Digit cipher lexicon 1
    DIGIT_1 = 11
    # Audio waveform domain
    AUDIO_WAV = 12
    # Audio spectral domain
    AUDIO_SPECTRAL = 13
    # Parse characters
    PARSE_CHR = 14
    # Parse tokens
    PARSE_TOK = 15
    # Chinese tokens
    ZH_TOK = 16
    # Icelandic characters
    ICE_CHAR = 17
    # Icelandic tokens
    ICE_TOK = 18
    # Icelandic parse tokens
    ICE_PARSE_TOK = 19
    # Macedonian tokens
    MK_TOK = 20
    # Czech tokens
    CS_TOK = 21
    # Czech characters
    CS_CHR = 22
    # Genetic bases (ACTG)
    DNA = 23
    # Real numbers
    REAL = 24
    # Images
    IMAGE = 25
    # Peptide
    PEPTIDE = 26
    # Python
    PY_TOK = 27
    # C++
    CPP_TOK = 28
