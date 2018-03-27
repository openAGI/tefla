# -*- coding: utf-8 -*-
"""A collection of commonly used post-processing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def strip_bpe(text):
  """Deodes text that was processed using BPE from
  https://github.com/rsennrich.

  /subword-nmt.
  """
  return text.replace("@@ ", "").strip()


def decode_sentencepiece(text):
  """Decodes text that uses https://github.com/google/sentencepiece encoding.

  Assumes that pieces are separated by a space
  """
  return "".join(text.split(" ")).replace("▁", " ").strip()


def slice_text(text, eos_token="SEQUENCE_END", sos_token="SEQUENCE_START"):
  """Slices text from SEQUENCE_START to SEQUENCE_END, not including these
  special tokens."""
  eos_index = text.find(eos_token)
  text = text[:eos_index] if eos_index > -1 else text
  sos_index = text.find(sos_token)
  text = text[sos_index + len(sos_token):] if sos_index > -1 else text
  return text.strip()
