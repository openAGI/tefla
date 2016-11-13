from __future__ import division, print_function, absolute_import

from tefla.core.layers import batch_norm_lasagne
from tefla.core.layers import prelu


def common_layer_args(is_training, reuse, outputs_collections, **kwargs):
    args = {
        'is_training': is_training,
        'reuse': reuse,
        'outputs_collections': outputs_collections,
    }
    args.update(kwargs)
    return args


def make_args(**kwargs):
    return kwargs
