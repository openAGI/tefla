from __future__ import absolute_import, division, print_function
import random
import six
from six.moves import xrange
from tefla.dataset.tokenizer import InvertibleTokenizer
import tensorflow as tf


class TokenizerTest(tf.test.TestCase):

    def setUp(self):
        super(TokenizerTest, self).setUp()
        self.tokenizer = InvertibleTokenizer()

    def test_encode(self):
        self.assertListEqual(
            [u"Dude", u"that", u"'", u"s", u"so", u"cool", u"."],
            self.tokenizer.encode(u"Dude that's so cool."))
        self.assertListEqual([u" ", u"Spaces", u"at", u"the", u"ends", u" "],
                             self.tokenizer.encode(u" Spaces at the ends "))
        self.assertListEqual([u"802", u".", u"11b"],
                             self.tokenizer.encode(u"802.11b"))
        self.assertListEqual([u"two", u". \n", u"lines"],
                             self.tokenizer.encode(u"two. \nlines"))

    def test_decode(self):
        self.assertEqual(
            u"Dude that's so cool.",
            self.tokenizer.decode(
                [u"Dude", u"that", u"'", u"s", u"so", u"cool", u"."]))

    def test_invertibility_on_random_strings(self):
        for _ in xrange(1000):
            s = u"".join(six.unichr(random.randint(0, 65535))
                         for _ in xrange(10))
            self.assertEqual(s, self.tokenizer.decode(self.tokenizer.encode(s)))


if __name__ == "__main__":
    tf.test.main()
