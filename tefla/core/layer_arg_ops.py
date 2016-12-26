from __future__ import division, print_function, absolute_import

import tensorflow as tf
from collections import OrderedDict


def common_layer_args(is_training, reuse, **kwargs):
    """
    Creates all common parameters

    Args:
        is_training: a bool, training or prediction
        resue: resue variables or initializes 
        **kwargs: other  common arguments
    """
    args = {
        'is_training': is_training,
        'reuse': reuse,
        'outputs_collections': _collection_name(is_training),
    }
    args.update(kwargs)
    return args


def make_args(**kwargs):
    """
    Creates all parameters from a dict
    """
    return kwargs


def end_points(is_training):
    """
    Returns end_points for training or validation

    Args:
        is_training: a bool, training or validation
    """
    return OrderedDict(tf.get_collection(_collection_name(is_training)))


def _collection_name(is_training):
    return 'training_activations' if is_training else 'prediction_activations'
