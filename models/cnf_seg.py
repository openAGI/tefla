from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
from tefla.core.lr_policy import PolyDecayPolicy, StepDecayPolicy
from tefla.da.standardizer import SamplewiseStandardizerTF
from tefla.utils import util

num_classes = 10
cnf = {
    'name': __name__.split('.')[-1],
    'batch_size_train': 1,
    'batch_size_test': 1,
    'balance_ratio': 0.975,
    'im_height': 256,
    'im_width': 256,
    'num_gpus': 1,
    'TOWER_NAME': 'tower',
    'standardizer': SamplewiseStandardizerTF(clip=6),
    # 'balance_weights' via data_set
    'final_balance_weights': np.array([1 for i in range(0, num_classes)], dtype=float),
    'l2_reg': 0.0005,
    'optname': 'adam',
    'opt_kwargs': {'decay': 0.9},
    'summary_dir': '/media/Data/sumary/512_bn',
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    ),
    'num_epochs': 451,
    'lr_policy': PolyDecayPolicy(0.00005),
    # 'lr_policy': StepDecayPolicy({0: 0.0002, 100: 0.0002, 200: 0.0002, 400: 0.0002, 500: 0.0001}),
    'classification': True,
    'validation_scores': [('validation mean iou', tf.contrib.metrics.streaming_mean_iou)],
}
