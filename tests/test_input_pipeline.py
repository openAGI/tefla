# -*- coding: utf-8 -*-
"""
Unit tests for input-related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import tensorflow as tf
import numpy as np

from tefla.dataset.input_pipeline import SplitTokensDecoder, ParallelDataProvider, make_parallel_data_provider


class SplitTokensDecoderTest(tf.test.TestCase):
    """Tests the SplitTokensDecoder class
    """

    def test_decode(self):
        decoder = SplitTokensDecoder(
            delimiter=" ",
            tokens_feature_name="source_tokens",
            length_feature_name="source_len")

        self.assertEqual(decoder.list_items(), ["source_tokens", "source_len"])

        data = tf.constant("Hello world ! 笑ｗ")

        decoded_tokens = decoder.decode(data, ["source_tokens"])
        decoded_length = decoder.decode(data, ["source_len"])
        decoded_both = decoder.decode(data, decoder.list_items())

        with self.test_session() as sess:
            decoded_tokens_ = sess.run(decoded_tokens)[0]
            decoded_length_ = sess.run(decoded_length)[0]
            decoded_both_ = sess.run(decoded_both)

        self.assertEqual(decoded_length_, 4)
        np.testing.assert_array_equal(
            np.char.decode(decoded_tokens_.astype("S"), "utf-8"),
            ["Hello", "world", "!", "笑ｗ"])

        self.assertEqual(decoded_both_[1], 4)
        np.testing.assert_array_equal(
            np.char.decode(decoded_both_[0].astype("S"), "utf-8"),
            ["Hello", "world", "!", "笑ｗ"])


if __name__ == "__main__":
    tf.test.main()
