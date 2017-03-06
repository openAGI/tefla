import numpy as np
import tensorflow as tf
from tefla.core import initializers as initz
from tefla.core.layers import conv2d, fully_connected, prelu, lrelu, max_pool
from tefla.core.layer_arg_ops import common_layer_args, make_args
from tefla.core.layers import dropout, rms_pool_2d, feature_max_pool_1d, global_avg_pool, softmax
from tefla.core.layers import upsample2d, subpixel2d, batch_norm_tf as batch_norm
from tefla.da import data
from tefla.da import iterator

image_size = (128, 128)
crop_size = (128, 128)


def get_z(shape, reuse, minval=-1.0, maxval=1.0, name='z', dtype=tf.float32, is_ref=False):
    assert dtype is tf.float32
    if is_ref:
        with tf.variable_scope(name, resue=reuse):
            z = tf.get_variable("z", shape, initializer=tf.random_uniform_initializer(
                minval, maxval), trainable=False)
    else:
        z = tf.random_uniform(shape, minval=minval,
                              maxval=maxval, name=name, dtype=tf.float32)
    return z


def discriminator(inputs, is_training, reuse, num_classes=11, batch_size=32):
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
    x = conv2d(x, 192, stride=(2, 2), name="d_conv2_3", **conv_args)
    end_points['d_conv2_3'] = x
    x = dropout(x, drop_p=0.2, name="dropout2", **common_args)
    x = conv2d(x, 192, stride=(2, 2), name="d_conv3_1", **conv_args)
    end_points['d_conv3_1'] = x
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

    if is_training:
        batch_size = 2 * batch_size
        generated_class_logits = tf.squeeze(
            tf.slice(logits, [0, num_classes - 1], [batch_size, 1]))
        end_points['generated_class_logits'] = generated_class_logits
        positive_class_logits = tf.slice(
            logits, [0, 0], [batch_size, num_classes - 1])
        end_points['positive_class_logits'] = positive_class_logits

        max_ = tf.reduce_max(positive_class_logits, 1, keep_dims=True)
        safe_pos_class_logits = positive_class_logits - max_
        end_points['safe_pos_class_logits'] = safe_pos_class_logits

        gan_logits = tf.log(tf.reduce_sum(
            tf.exp(safe_pos_class_logits), 1)) + tf.squeeze(max_) - generated_class_logits
        end_points['gan_logits'] = gan_logits
        assert len(gan_logits.get_shape()) == 1

        probs = tf.nn.sigmoid(gan_logits)
        end_points['probs'] = probs
        class_logits = tf.slice(logits, [0, 0], [batch_size / 2, num_classes])
        end_points['class_logits'] = class_logits
        D_on_data = tf.slice(probs, [0], [batch_size / 2])
        end_points['D_on_data'] = D_on_data
        D_on_data_logits = tf.slice(gan_logits, [0], [batch_size / 2])
        end_points['D_on_data_logits'] = D_on_data_logits
        D_on_G = tf.slice(probs, [batch_size / 2], [batch_size / 2])
        end_points['D_on_G'] = D_on_G
        D_on_G_logits = tf.slice(
            gan_logits, [batch_size / 2], [batch_size / 2])
        end_points['D_on_G_logits'] = D_on_G_logits

        return end_points
    else:
        return end_points


def generator(z_shape, is_training, reuse, batch_size=32):
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
    z = get_z(z_shape, reuse)
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
    x = upsample2d(x, [batch_size, 32, 32, 64],
                   name="g_deconv2d_3", **conv_args)
    end_points['g_deconv2d_3'] = x
    x = upsample2d(x, [batch_size, 64, 64, 32],
                   name="g_deconv2d_4", **conv_args)
    end_points['g_deconv2d_4'] = x
    x = upsample2d(x, [batch_size, 128, 128, 3],
                   name="g_deconv2d_5", **conv_args_1st)
    end_points['g_deconv2d_5'] = x

    end_points['softmax'] = tf.nn.tanh(x)
    return end_points
