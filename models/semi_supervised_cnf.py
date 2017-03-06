from __future__ import division, print_function, absolute_import

import numpy as np
from tefla.core.lr_policy import PolyDecayPolicy, StepDecayPolicy
from tefla.da.standardizer import SamplewiseStandardizer
from tefla.da.standardizer import AggregateStandardizer
from tefla.utils import util

cnf = {
    'name': __name__.split('.')[-1],
    'batch_size_train': 64,
    'batch_size_test': 64,
    'balance_ratio': 0.975,
    # 'standardizer': SamplewiseStandardizer(clip=6),
    # 'balance_weights' via data_set
    'final_balance_weights': np.array([1, 2, 2, 2, 2], dtype=float),
    'l2_reg': 0.0005,
    'optname': 'adam',
    'opt_kwargs': {'beta1': 0.5},
    'gpu_memory_fraction': 0.8,
    'num_gpus': 1,
    'd_label_smooth': 0.25,
    'generator_target_prob': 0.375,
    'summary_dir': '/media/Data/eyepacs/summary/512_bn',
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'standardizer': AggregateStandardizer(
        mean=np.array([108.64628601, 75.86886597, 54.34005737],
                      dtype=np.float32),
        std=np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32),
        u=np.array([[-0.56543481, 0.71983482, 0.40240142],
                    [-0.5989477, -0.02304967, -0.80036049],
                    [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32),
        ev=np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32),
        sigma=0.5
    ),
    'num_epochs': 555,
    # 'lr_policy': PolyDecayPolicy(0.00005),
    'lr_policy': StepDecayPolicy({0: 0.0002, 100: 0.0002, 200: 0.0002, 400: 0.0002, 500: 0.0001}),
    'classification': True,
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
    # 'validation_scores': [('validation kappa', util.kappa_wrapper)],
}
