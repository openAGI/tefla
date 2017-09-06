from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import sys
import unicodedata

import six
from six.moves import xrange
import tensorflow as tf

# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in xrange(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


@six.add_metaclass(abc.ABCMeta)
class BaseTokenizer():

    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError


class InvertibleTokenizer(BaseTokenizer):

    def encode(self, text):
        """Encode a unicode string as a list of tokens.

        Args:
          text: a unicode string

        Returns:
          a list of tokens as Unicode strings
        """
        if not text:
            return []
        ret = []
        token_start = 0
        # Classify each character in the input string
        is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
        for pos in xrange(1, len(text)):
            if is_alnum[pos] != is_alnum[pos - 1]:
                token = text[token_start:pos]
                if token != u" " or token_start == 0:
                    ret.append(token)
                token_start = pos
        final_token = text[token_start:]
        ret.append(final_token)
        return ret

    def decode(self, tokens):
        """Decode a list of tokens to a unicode string.

        Args:
          tokens: a list of Unicode strings

        Returns:
          a unicode string
        """
        token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
        ret = []
        for i, token in enumerate(tokens):
            if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
                ret.append(u" ")
            ret.append(token)
        return "".join(ret)
