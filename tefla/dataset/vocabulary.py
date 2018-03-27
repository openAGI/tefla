from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tensorflow import gfile
from ..core import logger

SpecialVocab = collections.namedtuple("SpecialVocab", ["UNK", "SEQUENCE_START", "SEQUENCE_END"])


class Vocabulary(object):
  """Vocabulary class for an image-to-text model."""

  def __init__(self, vocab_file, start_word="<S>", end_word="</S>", unk_word="<UNK>"):
    """Initializes the vocabulary.

    Args:
        vocab_file: File containing the vocabulary, where the words are the first
          whitespace-separated token on each line (other tokens are ignored) and
          the word ids are the corresponding line numbers.
        start_word: Special word denoting sentence start.
        end_word: Special word denoting sentence end.
        unk_word: Special word denoting unknown words.
    """
    if not tf.gfile.Exists(vocab_file):
      logger.fatal("Vocab file %s not found." % vocab_file)
    logger.info("Initializing vocabulary from file: %s" % vocab_file)

    with tf.gfile.GFile(vocab_file, mode="r") as f:
      reverse_vocab = list(f.readlines())
    reverse_vocab = [line.split()[0] for line in reverse_vocab]
    assert start_word in reverse_vocab
    assert end_word in reverse_vocab
    if unk_word not in reverse_vocab:
      reverse_vocab.append(unk_word)
    vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

    logger.info("Created vocabulary with %d words" % len(vocab))

    self.vocab = vocab  # vocab[word] = id
    self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

    # Save special word ids.
    self.start_id = vocab[start_word]
    self.end_id = vocab[end_word]
    self.unk_id = vocab[unk_word]

  def word_to_id(self, word):
    """Returns the integer word id of a word string.

    Args:
        word: `str`, input word

    Returns:
        a `int`; integer id of the word with respect to the vocabulary
    """
    if word in self.vocab:
      return self.vocab[word]
    else:
      return self.unk_id

  def id_to_word(self, word_id):
    """Returns the word string of an integer word id.

    Args:
        word: `int`, input word_id

    Returns:
        a `str`; word string of the word id with respect to the vocabulary
    """
    if word_id >= len(self.reverse_vocab):
      return self.reverse_vocab[self.unk_id]
    else:
      return self.reverse_vocab[word_id]


class VocabInfo(collections.namedtuple("VocbabInfo", ["path", "vocab_size", "special_vocab"])):
  """Convenience structure for vocabulary information."""

  @property
  def total_size(self):
    """Returns size the the base vocabulary plus the size of extra
    vocabulary."""
    return self.vocab_size + len(self.special_vocab)


def get_vocab_info(vocab_path):
  """Creates a `VocabInfo` instance that contains the vocabulary size and the
  special vocabulary for the given file.

  Args:
    vocab_path: Path to a vocabulary file with one word per line.

  Returns:
    A VocabInfo tuple.
  """
  with gfile.GFile(vocab_path) as file:
    vocab_size = sum(1 for _ in file)
  special_vocab = get_special_vocab(vocab_size)
  return VocabInfo(vocab_path, vocab_size, special_vocab)


def get_special_vocab(vocabulary_size):
  """Returns the `SpecialVocab` instance for a given vocabulary size."""
  return SpecialVocab(*range(vocabulary_size, vocabulary_size + 3))


def create_vocabulary_lookup_table(filename, default_value=None):
  """Creates a lookup table for a vocabulary file.

  Args:
    filename: Path to a vocabulary file containg one word per line.
      Each word is mapped to its line number.
    default_value: UNK tokens will be mapped to this id.
      If None, UNK tokens will be mapped to [vocab_size]

    Returns:
      A tuple (vocab_to_id_table, id_to_vocab_table,
      word_to_count_table, vocab_size). The vocab size does not include
      the UNK token.
  """
  if not gfile.Exists(filename):
    raise ValueError("File does not exist: {}".format(filename))

  # Load vocabulary into memory
  with gfile.GFile(filename) as file:
    vocab = list(line.strip("\n") for line in file)
  vocab_size = len(vocab)

  has_counts = len(vocab[0].split("\t")) == 2
  if has_counts:
    vocab, counts = zip(*[_.split("\t") for _ in vocab])
    counts = [float(_) for _ in counts]
    vocab = list(vocab)
  else:
    counts = [-1. for _ in vocab]

  # Add special vocabulary items
  special_vocab = get_special_vocab(vocab_size)
  vocab += list(special_vocab._fields)
  vocab_size += len(special_vocab)
  counts += [-1. for _ in list(special_vocab._fields)]

  if default_value is None:
    default_value = special_vocab.UNK

  logger.info("Creating vocabulary lookup table of size %d" % vocab_size)

  vocab_tensor = tf.constant(vocab)
  count_tensor = tf.constant(counts, dtype=tf.float32)
  vocab_idx_tensor = tf.range(vocab_size, dtype=tf.int64)

  # Create ID -> word mapping
  id_to_vocab_init = tf.contrib.lookup.KeyValueTensorInitializer(vocab_idx_tensor, vocab_tensor,
                                                                 tf.int64, tf.string)
  id_to_vocab_table = tf.contrib.lookup.HashTable(id_to_vocab_init, "UNK")

  # Create word -> id mapping
  vocab_to_id_init = tf.contrib.lookup.KeyValueTensorInitializer(vocab_tensor, vocab_idx_tensor,
                                                                 tf.string, tf.int64)
  vocab_to_id_table = tf.contrib.lookup.HashTable(vocab_to_id_init, default_value)

  # Create word -> count mapping
  word_to_count_init = tf.contrib.lookup.KeyValueTensorInitializer(vocab_tensor, count_tensor,
                                                                   tf.string, tf.float32)
  word_to_count_table = tf.contrib.lookup.HashTable(word_to_count_init, -1)

  return vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_size
