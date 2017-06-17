
"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
from six.moves import urllib
import tensorflow as tf

from tefla.utils.seq2seq_utils import prepare_data
# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
    """Download filename from url unless it's already in directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print("Downloading %s to %s" % (url, filepath))
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes")
    return filepath


def gunzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path."""
    print("Unpacking %s to %s" % (gz_path, new_path))
    with gzip.open(gz_path, "rb") as gz_file:
        with open(new_path, "wb") as new_file:
            for line in gz_file:
                new_file.write(line)


def get_wmt_enfr_train_set(directory):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    train_path = os.path.join(directory, "giga-fren.release2.fixed")
    if not (tf.gfile.Exists(train_path + ".fr") and tf.gfile.Exists(train_path + ".en")):
        corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                     _WMT_ENFR_TRAIN_URL)
        print("Extracting tar file %s" % corpus_file)
        with tarfile.open(corpus_file, "r") as corpus_tar:
            corpus_tar.extractall(directory)
        gunzip_file(train_path + ".fr.gz", train_path + ".fr")
        gunzip_file(train_path + ".en.gz", train_path + ".en")
    return train_path


def get_wmt_enfr_dev_set(directory):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    dev_name = "newstest2013"
    dev_path = os.path.join(directory, dev_name)
    if not (tf.gfile.Exists(dev_path + ".fr") and tf.gfile.Exists(dev_path + ".en")):
        dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
        print("Extracting tgz file %s" % dev_file)
        with tarfile.open(dev_file, "r:gz") as dev_tar:
            fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
            en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
            # Extract without "dev/" prefix.
            fr_dev_file.name = dev_name + ".fr"
            en_dev_file.name = dev_name + ".en"
            dev_tar.extract(fr_dev_file, directory)
            dev_tar.extract(en_dev_file, directory)
    return dev_path


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
    """Get WMT data into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      en_vocabulary_size: size of the English vocabulary to create and use.
      fr_vocabulary_size: size of the French vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for English training data-set,
        (2) path to the token-ids for French training data-set,
        (3) path to the token-ids for English development data-set,
        (4) path to the token-ids for French development data-set,
        (5) path to the English vocabulary file,
        (6) path to the French vocabulary file.
    """
    # Get wmt data to the specified directory.
    train_path = get_wmt_enfr_train_set(data_dir)
    dev_path = get_wmt_enfr_dev_set(data_dir)

    from_train_path = train_path + ".en"
    to_train_path = train_path + ".fr"
    from_dev_path = dev_path + ".en"
    to_dev_path = dev_path + ".fr"
    return prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, en_vocabulary_size,
                        fr_vocabulary_size, tokenizer)


if __name__ == '__main__':
    prepare_wmt_data()
