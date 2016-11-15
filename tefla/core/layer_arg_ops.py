from __future__ import division, print_function, absolute_import

import tensorflow as tf
from collections import OrderedDict


def common_layer_args(is_training, reuse, **kwargs):
    args = {
        'is_training': is_training,
        'reuse': reuse,
        'outputs_collections': _collection_name(is_training),
    }
    args.update(kwargs)
    return args


def make_args(**kwargs):
    return kwargs


def end_points(is_training):
    return OrderedDict(tf.get_collection(_collection_name(is_training)))


def _collection_name(is_training):
    return 'training_activations' if is_training else 'prediction_activations'
