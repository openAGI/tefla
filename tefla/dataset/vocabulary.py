from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Vocabulary(object):
    """Vocabulary class for an image-to-text model."""

    def __init__(self,
                 vocab_file,
                 start_word="<S>",
                 end_word="</S>",
                 unk_word="<UNK>"):
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
            tf.logging.fatal("Vocab file %s not found.", vocab_file)
        tf.logging.info("Initializing vocabulary from file: %s", vocab_file)

        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab]
        assert start_word in reverse_vocab
        assert end_word in reverse_vocab
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

        tf.logging.info("Created vocabulary with %d words" % len(vocab))

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
