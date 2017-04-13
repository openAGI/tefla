import numpy as np
import tensorflow as tf
import cv2
from tefla.core import initializers as initz
from tefla.core.layers import conv2d, fully_connected, prelu, lrelu, max_pool
from tefla.core.layer_arg_ops import common_layer_args, make_args
from tefla.core.layers import dropout, rms_pool_2d, feature_max_pool_1d, global_avg_pool, softmax
from tefla.core.layers import upsample2d, subpixel2d, batch_norm_tf as batch_norm
from tefla.da import data
from tefla.da import iterator

# image_size = (128, 128)
# crop_size = (128, 128)
image_size = (32, 32)
crop_size = (32, 32)


def get_z(shape, reuse, mean=1.0, stddev=0.02, name='z', dtype=tf.float32, is_ref=False):
    assert dtype is tf.float32
    if is_ref:
        with tf.variable_scope(name, resue=reuse):
            z = tf.get_variable(name, shape, initializer=initz.random_normal(
                seed=None, mean=mean, stddev=stddev), trainable=False)
    else:
        z = tf.random_normal(shape, mean=mean, stddev=stddev,
                             dtype=tf.float32, seed=None, name=name)
    return z


def generator(z, is_training, reuse, batch_size=32):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=lrelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    conv_args_1st = make_args(batch_norm=None, activation=lrelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    fc_args = make_args(
        activation=lrelu, w_init=initz.he_normal(scale=1), **common_args)
    pool_args = make_args(padding='SAME', **common_args)
    # project `z` and reshape
    # TODO think about phase again
    end_points = {}
    # z = get_z(z_shape, reuse)
    end_points['z'] = z
    z_fc = fully_connected(z, 4 * 4 * 512, name="g_fc", **fc_args)
    end_points['g_fc'] = z_fc
    x = tf.reshape(z_fc, [batch_size, 4, 4, 512])
    end_points['g_reshaped'] = x
    x = upsample2d(x, [batch_size, 8, 8, 256],
                   name="g_deconv2d_1", **conv_args)
    end_points['g_deconv2d_1'] = x
    x = upsample2d(x, [batch_size, 16, 16, 128],
                   name="g_deconv2d_2", **conv_args)
    end_points['g_deconv2d_2'] = x
    # x = upsample2d(x, [batch_size, 32, 32, 16 * 3],
    #               name="g_deconv2d_3", **conv_args)
    # end_points['g_deconv2d_3'] = x
    # for now lets examine cifar
    # x = subpixel2d(x, 4, name='z_subpixel1')
    # x shape[batch_size, 128, 128, 3]
    # end_points['subpixel1'] = x
    x = upsample2d(x, [batch_size, 32, 32, 3],
                   name="g_deconv2d_3", **conv_args)
    end_points['g_deconv2d_3'] = x
    # x = upsample2d(x, [batch_size, 64, 64, 32],
    #               name="g_deconv2d_4", **conv_args)
    # end_points['g_deconv2d_4'] = x
    # x = upsample2d(x, [batch_size, 128, 128, 3],
    #               name="g_deconv2d_5", **conv_args_1st)
    # end_points['g_deconv2d_5'] = x

    end_points['softmax'] = tf.nn.tanh(x)
    return end_points


def encoder(inputs, is_training, reuse, z_dim=512):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=lrelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    conv_args_1st = make_args(batch_norm=None, activation=lrelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    logits_args = make_args(
        activation=None, w_init=initz.he_normal(scale=1), **common_args)
    pool_args = make_args(padding='SAME', **common_args)
    end_points = {}
    x = inputs
    end_points['inputs'] = x
    x = dropout(x, drop_p=0.2, name="input_dropout1", **common_args)
    x = conv2d(x, 96, filter_size=(5, 5), stride=(
        2, 2), name="e_conv1_1", **conv_args_1st)
    end_points['e_conv1_1'] = x
    x = conv2d(x, 96, name="e_conv1_2", **conv_args)
    end_points['e_conv1_2'] = x
    x = conv2d(x, 96, stride=(2, 2), name="e_conv1_3", **conv_args)
    end_points['e_conv1_3'] = x
    x = dropout(x, drop_p=0.2, name="dropout1", **common_args)
    x = conv2d(x, 192, name="e_conv2_1", **conv_args)
    end_points['e_conv2_1'] = x
    x = conv2d(x, 192, name="e_conv2_2", **conv_args)
    end_points['e_conv2_2'] = x
    # x = conv2d(x, 192, stride=(2, 2), name="e_conv2_3", **conv_args)
    # end_points['e_conv2_3'] = x
    x = dropout(x, drop_p=0.2, name="dropout2", **common_args)
    # x = conv2d(x, 192, stride=(2, 2), name="e_conv3_1", **conv_args)
    # end_points['e_conv3_1'] = x
    x = conv2d(x, 192, filter_size=(1, 1), name="e_conv4_1", **conv_args)
    end_points['e_conv4_1'] = x
    x = conv2d(x, 192, filter_size=(1, 1), name="e_conv4_2", **conv_args)
    end_points['e_conv4_2'] = x
    x = global_avg_pool(x, name="global_pool")
    end_points['global_pool'] = x
    logits1 = fully_connected(x, z_dim, name="e_logits1", **logits_args)
    logits2 = fully_connected(x, z_dim, name="e_logits2",
                              **logits_args)
    logits2 = tf.tanh(logits2, name='e_logits2_tanh')
    end_points['e_logits1'] = logits1
    end_points['e_logits2'] = logits2
    return end_points


def discriminator(inputs, is_training, reuse, num_classes=1):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=lrelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    conv_args_1st = make_args(batch_norm=None, activation=lrelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    logits_args = make_args(
        activation=None, w_init=initz.he_normal(scale=1), **common_args)
    pool_args = make_args(padding='SAME', **common_args)
    end_points = {}
    x = inputs
    end_points['inputs'] = x
    x = dropout(x, drop_p=0.2, name="input_dropout1", **common_args)
    x = conv2d(x, 96, filter_size=(5, 5), stride=(
        2, 2), name="d_conv1_1", **conv_args_1st)
    end_points['d_conv1_1'] = x
    x = conv2d(x, 96, name="d_conv1_2", **conv_args)
    end_points['d_conv1_2'] = x
    x = conv2d(x, 96, stride=(2, 2), name="d_conv1_3", **conv_args)
    end_points['d_conv1_3'] = x
    x = dropout(x, drop_p=0.2, name="dropout1", **common_args)
    x = conv2d(x, 192, name="d_conv2_1", **conv_args)
    end_points['d_conv2_1'] = x
    x = conv2d(x, 192, name="d_conv2_2", **conv_args)
    end_points['d_conv2_2'] = x
    # x = conv2d(x, 192, stride=(2, 2), name="d_conv2_3", **conv_args)
    # end_points['d_conv2_3'] = x
    x = dropout(x, drop_p=0.2, name="dropout2", **common_args)
    # x = conv2d(x, 192, stride=(2, 2), name="d_conv3_1", **conv_args)
    # end_points['d_conv3_1'] = x
    x = conv2d(x, 192, filter_size=(1, 1), name="d_conv4_1", **conv_args)
    end_points['d_conv4_1'] = x
    x = conv2d(x, 192, filter_size=(1, 1), name="d_conv4_2", **conv_args)
    end_points['d_conv4_2'] = x
    x = global_avg_pool(x, name="global_pool")
    end_points['global_pool'] = x
    logits = fully_connected(x, num_classes, name="d_logits", **logits_args)
    end_points['logits'] = logits
    end_points['predictions'] = softmax(
        logits, name='predictions', **common_args)
    return end_points
