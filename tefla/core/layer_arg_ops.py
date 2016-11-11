from __future__ import division, print_function, absolute_import

from tefla.core.layers import batch_norm_lasagne
from tefla.core.layers import prelu


def common_conv_params(is_training, reuse, outputs_collections, activation=prelu, batch_norm=batch_norm_lasagne,
                       **kwargs):
    args = {
        'is_training': is_training,
        'reuse': reuse,
        'outputs_collections': outputs_collections,
        'activation': activation,
        'batch_norm': batch_norm,
    }
    args.update(kwargs)
    return args


def common_fc_params(is_training, reuse, outputs_collections, activation=prelu, **kwargs):
    args = {
        'is_training': is_training,
        'reuse': reuse,
        'outputs_collections': outputs_collections,
        'activation': activation,
    }
    args.update(kwargs)
    return args


def common_batch_norm_params(is_training, reuse, outputs_collections=None, **kwargs):
    args = {
        'is_training': is_training,
        'reuse': reuse,
        'outputs_collections': outputs_collections,
    }
    args.update(kwargs)
    return args


def common_pool_params(outputs_collections, **kwargs):
    args = {
        'outputs_collections': outputs_collections,
    }
    args.update(kwargs)
    return args


def common_dropout_params(is_training, outputs_collections):
    args = {
        'is_training': is_training,
        'outputs_collections': outputs_collections,
    }
    return args
