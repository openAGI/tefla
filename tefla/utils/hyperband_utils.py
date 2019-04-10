"""Helper functions for hyperband
"""
import numpy as np


# pylint: disable=invalid-name
def layer_config(config, layer, layer_type='conv'):
  """layerwise configuration for a given layer-number, type and config

  Args:
      config: training configuration sampled from hyperband search space
      layer: layer number
      layer_type: whether conv layer or fc layer

  Returns:
      1. activation name
      2. hidden-units size in case of fc layer_type
         output channel size in case of conv layer_type
      3. whether or not maxpool if conv layer_type
         dropout value in case fc layer_type
  """
  layer = layer_type + '_' + 'layer_' + str(layer)
  if layer_type == 'conv':
    return config[layer + '_activation'], config[layer + '_size'], config[layer + '_maxpool']
  return config[layer + '_activation'], config[layer + '_size'], config[layer + '_dropout']


def get_config(config):
  """get configuration in a dict format

  Args:
      config: training configuration sampled from hyperband search space

  Returns:
      dict, layerwise configuration with better readable format
  """
  current_config = dict()
  current_config.update({
      'batch_norm': config['batch_norm'],
      'optname': config['optname']})

  if config['max_conv_layers'] > 0:
    for i in range(1, config['n_conv_layers'] + 1):
      activation, size, maxpool = layer_config(config, i, layer_type='conv')
      current_config.update({'conv_layer_{}'.format(i): {'size': size,
                                                         'activation': activation,
                                                         'maxpool': maxpool
                                                         }})

  if config['max_fc_layers'] > 0:
    for i in range(1, config['n_fc_layers'] + 1):
      activation, size, dropout = layer_config(config, i, layer_type='fc')
      current_config.update({'fc_layer_{}'.format(i): {'size': size,
                                                       'activation': activation,
                                                       'dropout': np.round(dropout, 2)
                                                       }})
  return current_config


def handle_integers(params):
  """Handle floats which should be integers

  Args:
      params: dict, parameters from hyperband search space

  Returns:
      new_params: dict, parameters with corrected data types
  """
  new_params = {}
  for k, v in params.items():
    if v.__class__ == float and int(v) == v:
      new_params[k] = int(v)
    else:
      new_params[k] = v
  return new_params
