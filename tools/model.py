from tefla.core.layers import conv2d, fully_connected as fc, prelu, softmax
from tefla.core.layer_arg_ops import common_layer_args, end_points


def model(inputs, is_training, reuse, num_classes=2):
  common_args = common_layer_args(is_training, reuse)
  conv1 = conv2d(inputs, 32, name='conv1', activation=prelu, **common_args)
  conv1 = conv2d(conv1, 32, name='conv2', activation=prelu, **common_args)
  fc1 = fc(conv1, num_classes, name='logits', **common_args)
  prediction = softmax(fc1, name='prediction', **common_args)
  return end_points(is_training)
