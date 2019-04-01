import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tefla.core.mem_dataset import DataSet

def mnist_dataset():
  """creating DataSet object of MNIST data
  """
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

  width = 28
  height = 28

  train_images = mnist[0].images
  train_labels = mnist[0].labels

  validation_images = mnist[1].images
  validation_labels = mnist[1].labels

  return DataSet(train_images, train_labels, validation_images, validation_labels)

def layer_config(config, layer, layer_type='conv'):
  layer = layer_type + '_' + 'layer_'+ str(layer)
  if layer_type == 'conv':
    return config[layer+'_activation'], config[layer+'_size'], config[layer+'_maxpool']
  else:
    return config[layer+'_activation'], config[layer+'_size'], config[layer+'_dropout']

def get_config(config):
  current_config = dict()
  current_config.update({
    'max_conv_layers': config['max_conv_layers'],
    'n_conv_layers': config['n_conv_layers'],
    'n_fc_layers': config['n_fc_layers'],
    'batch_norm': config['batch_norm']})

  if config['max_conv_layers']>0:
    for i in range(1, config['n_conv_layers']+1):
      activation, size, maxpool = layer_config(config, i, layer_type='conv')
      current_config.update({'conv_layer_{}'.format(i): {'size': size,
       'activation': activation,
       'maxpool': maxpool
       }})

  if config['max_fc_layers']>0:
    for i in range(1, config['n_fc_layers']+1):
      activation, size, dropout = layer_config(config, i, layer_type='fc')
      current_config.update({'fc_layer_{}'.format(i): {'size': size,
             'activation': activation,
             'dropout': np.round(dropout,2)
             }})
  return current_config

def handle_integers(params):
  new_params = {}
  for k, v in params.items():
    if type( v ) == float and int( v ) == v:
      new_params[k] = int( v )
    else:
      new_params[k] = v
  return new_params