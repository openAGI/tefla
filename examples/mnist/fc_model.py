from tefla.core.layers import input, fully_connected, softmax, relu
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points


width = 28
height = 28


def model(is_training, reuse):
    common_args = common_layer_args(is_training, reuse)
    fc_args = make_args(activation=relu, **common_args)
    logit_args = make_args(activation=None, **common_args)

    x = input((None, height * width), **common_args)
    x = fully_connected(x, n_output=100, name='fc1', **fc_args)
    logits = fully_connected(x, n_output=10, name="logits", **logit_args)
    predictions = softmax(logits, name='predictions', **common_args)

    return end_points(is_training)
