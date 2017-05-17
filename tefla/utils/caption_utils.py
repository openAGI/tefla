from __future__ import absolute_import, division, print_function

import heapq
import math
import numpy as np
import tensorflow as tf


class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Caption.

        Args:
            sentence: List of word ids in the caption.
            state: Model state after generating the previous word.
            logprob: Log-probability of the caption.
            score: Score of the caption.
            metadata: Optional metadata associated with the partial sentence. If not
               None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().

        Args:
            sort: Whether to return the elements in descending sorted order.

        Returns:
            A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class CaptionGenerator(object):
    """Class to generate captions from an image-to-text model."""

    def __init__(self,
                 model,
                 vocab,
                 beam_size=3,
                 max_caption_length=20,
                 length_normalization_factor=0.0):
        """Initializes the generator.

        Args:
            model: Object encapsulating a trained image-to-text model. Must have
                methods feed_image() and inference_step(). For example, an instance of
                InferenceWrapperBase.
            vocab: A Vocabulary object.
            beam_size: Beam size to use when generating captions.
            max_caption_length: The maximum caption length before stopping the search.
            length_normalization_factor: If != 0, a number x such that captions are
                scored by logprob/length^x, rather than logprob. This changes the
                relative scores of captions depending on their lengths. For example, if
                x > 0 then longer captions will be favored.
        """
        self.vocab = vocab
        self.model = model

        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, sess, encoded_image):
        """Runs beam search caption generation on a single image.

        Args:
            sess: TensorFlow Session object.
            encoded_image: An encoded image string.

        Returns:
            A list of Caption sorted by descending score.
        """
        # Feed in the image to get the initial state.
        initial_state = self.model.feed_image(sess, encoded_image)

        initial_beam = Caption(
            sentence=[self.vocab.start_id],
            state=initial_state[0],
            logprob=0.0,
            score=0.0,
            metadata=[""])
        partial_captions = TopN(self.beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(self.beam_size)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            input_feed = np.array([c.sentence[-1]
                                   for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])

            softmax, new_states, metadata = self.model.inference_step(sess,
                                                                      input_feed,
                                                                      state_feed)

            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[i]
                state = new_states[i]
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size]
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob
                    if metadata:
                        metadata_list = partial_caption.metadata + \
                            [metadata[i]]
                    else:
                        metadata_list = None
                    if w == self.vocab.end_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence)**self.length_normalization_factor
                        beam = Caption(sentence, state, logprob,
                                       score, metadata_list)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob,
                                       score, metadata_list)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                break

        if not complete_captions.size():
            complete_captions = partial_captions

        return complete_captions.extract(sort=True)


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
        tf.logging.info(
            "Initializing vocabulary from file: %s", vocab_file)

        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab]
        assert start_word in reverse_vocab
        assert end_word in reverse_vocab
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

        tf.logging.info("Created vocabulary with %d words" % len(vocab))

        self.vocab = vocab
        self.reverse_vocab = reverse_vocab

        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = vocab[unk_word]

    def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        else:
            return self.reverse_vocab[word_id]
