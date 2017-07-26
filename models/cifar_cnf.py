from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
from tefla.core.lr_policy import PolyDecayPolicy, StepDecayPolicy
from tefla.da.standardizer import SamplewiseStandardizer
from tefla.da.standardizer import AggregateStandardizer
from tefla.utils import util
from tefla.core import metrics

kappav2 = metrics.KappaV2(num_classes=10, batch_size=64)
cnf = {
    'name': __name__.split('.')[-1],
    'batch_size_train': 64,
    'batch_size_test': 64,
    'balance_ratio': 0.975,
    'image_size': (32, 32),
    'crop_size': (28, 28),
    'tfrecords_im_size': (32, 32, 3),
    'num_gpus': 1,
    'init_probs': [0.1, 0.1, 0.1, 0.1, 0.2, 0.05, 0.05, 0.1, 0.1, 0.1],
    'TOWER_NAME': 'tower',
    'standardizer': SamplewiseStandardizer(clip=6),
    # 'balance_weights' via data_set
    'final_balance_weights': np.array([1, 2, 2, 2, 2], dtype=float),
    'l2_reg': 0.0005,
    'optname': 'momentum',
    'opt_kwargs': {'decay': 0.9},
    'summary_dir': '/tmp/summary/cifar',
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
    # 'lr_policy': StepDecayPolicy({0: 0.0002, 100: 0.0002, 200: 0.0002, 400: 0.0002, 500: 0.0001}),
    'classification': True,
    'validation_scores': [('validation accuracy', tf.contrib.metrics.accuracy), ('validation kappa', kappav2.metric)],
}
