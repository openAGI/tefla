from __future__ import division, print_function, absolute_import

import numpy as np
from tefla.core.lr_policy import PolyDecayPolicy
from tefla.da.standardizer import SamplewiseStandardizer
from tefla.utils import util

cnf = {
    'name': __name__.split('.')[-1],
    'batch_size_train': 16,
    'batch_size_test': 16,
    'balance_ratio': 0.975,
    'standardizer': SamplewiseStandardizer(clip=6),
    'final_balance_weights': np.array([1, 2, 2, 2, 2], dtype=float),
    'l2_reg': 0.0005,
    'optname': 'momentum',
    'opt_kwargs': {'decay': 0.9},
    'summary_dir': '/media/Data/eyepacs/summary/512_bn',
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'num_epochs': 451,
    'lr_policy': PolyDecayPolicy(0.00005),
    'classification': True,
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
}
