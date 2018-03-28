# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import tempfile

from tefla.dataset import vocabulary


def create_temporary_vocab_file(words, counts=None):
  """
    Creates a temporary vocabulary file.

    Args:
      words: List of words in the vocabulary

    Returns:
      A temporary file object with one word per line
    """
  vocab_file = tempfile.NamedTemporaryFile()
  if counts is None:
    for token in words:
      vocab_file.write((token + "\n").encode("utf-8"))
  else:
    for token, count in zip(words, counts):
      vocab_file.write("{}\t{}\n".format(token, count).encode("utf-8"))
  vocab_file.flush()
  return vocab_file


class VocabInfoTest(tf.test.TestCase):
  """Tests VocabInfo class"""

  def setUp(self):
    super(VocabInfoTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.vocab_list = ["Hello", ".", "Bye"]
    self.vocab_file = create_temporary_vocab_file(self.vocab_list)

  def tearDown(self):
    super(VocabInfoTest, self).tearDown()
    self.vocab_file.close()

  def test_vocab_info(self):
    vocab_info = vocabulary.get_vocab_info(self.vocab_file.name)
    self.assertEqual(vocab_info.vocab_size, 3)
    self.assertEqual(vocab_info.path, self.vocab_file.name)
    self.assertEqual(vocab_info.special_vocab.UNK, 3)
    self.assertEqual(vocab_info.special_vocab.SEQUENCE_START, 4)
    self.assertEqual(vocab_info.special_vocab.SEQUENCE_END, 5)
    self.assertEqual(vocab_info.total_size, 6)


class CreateVocabularyLookupTableTest(tf.test.TestCase):
  """
    Tests Vocabulary lookup table operations.
    """

  def test_without_counts(self):
    vocab_list = ["Hello", ".", "笑"]
    vocab_file = create_temporary_vocab_file(vocab_list)

    vocab_to_id_table, id_to_vocab_table, _, vocab_size = \
        vocabulary.create_vocabulary_lookup_table(vocab_file.name)

    self.assertEqual(vocab_size, 6)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())

      ids = vocab_to_id_table.lookup(tf.convert_to_tensor(["Hello", ".", "笑", "??", "xxx"]))
      ids = sess.run(ids)
      self.assertAllEqual(ids, [0, 1, 2, 3, 3])

      words = id_to_vocab_table.lookup(tf.convert_to_tensor([0, 1, 2, 3], dtype=tf.int64))
      words = sess.run(words)
      self.assertAllEqual(np.char.decode(words.astype("S"), "utf-8"), ["Hello", ".", "笑", "UNK"])

  def test_with_counts(self):
    vocab_list = ["Hello", ".", "笑"]
    vocab_counts = [100, 200, 300]
    vocab_file = create_temporary_vocab_file(vocab_list, vocab_counts)

    vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_size = \
        vocabulary.create_vocabulary_lookup_table(vocab_file.name)

    self.assertEqual(vocab_size, 6)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())

      ids = vocab_to_id_table.lookup(tf.convert_to_tensor(["Hello", ".", "笑", "??", "xxx"]))
      ids = sess.run(ids)
      self.assertAllEqual(ids, [0, 1, 2, 3, 3])

      words = id_to_vocab_table.lookup(tf.convert_to_tensor([0, 1, 2, 3], dtype=tf.int64))
      words = sess.run(words)
      self.assertAllEqual(np.char.decode(words.astype("S"), "utf-8"), ["Hello", ".", "笑", "UNK"])

      counts = word_to_count_table.lookup(tf.convert_to_tensor(["Hello", ".", "笑", "??", "xxx"]))
      counts = sess.run(counts)
      self.assertAllEqual(counts, [100, 200, 300, -1, -1])


if __name__ == "__main__":
  tf.test.main()
