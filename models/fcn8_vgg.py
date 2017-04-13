from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import conv2d, upsample2d, fully_connected, max_pool, lrelu, prelu, register_to_collections
from tefla.core.layers import input, softmax, dropout, avg_pool_2d, batch_norm_tf as batch_norm
from tefla.core import initializers as initz
from tefla.utils import util


def model(inputs, is_training, reuse, num_classes=21, batch_size=1):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=lrelu, w_init=initz.he_normal(
        scale=1), untie_biases=False, **common_args)
    upsample_args = make_args(
        batch_norm=False, activation=lrelu, use_bias=False, **common_args)
    logits_args = make_args(
        activation=None, **common_args)
    pool_args = make_args(padding='SAME', **common_args)

    conv1_1 = conv2d(inputs, 64, name="vgg_19/conv1/conv1_1", **conv_args)
    conv1_2 = conv2d(conv1_1, 64, name="vgg_19/conv1/conv1_2", **conv_args)
    pool1 = max_pool(conv1_2, stride=2, name='pool1', **pool_args)
    conv2_1 = conv2d(pool1, 128, name="vgg_19/conv2/conv2_1", **conv_args)
    conv2_2 = conv2d(conv2_1, 128, name="vgg_19/conv2/conv2_2", **conv_args)
    pool2 = max_pool(conv2_2, stride=2, name='pool2', **pool_args)
    conv3_1 = conv2d(pool2, 256, name="vgg_19/conv3/conv3_1", **conv_args)
    conv3_2 = conv2d(conv3_1, 256, name="vgg_19/conv3/conv3_2", **conv_args)
    conv3_3 = conv2d(conv3_2, 256, name="vgg_19/conv3/conv3_3", **conv_args)
    conv3_4 = conv2d(conv3_3, 256, name="vgg_19/conv3/conv3_4", **conv_args)
    pool3 = max_pool(conv3_4, stride=2, name='pool3', **pool_args)
    conv4_1 = conv2d(pool3, 512, name="vgg_19/conv4/conv4_1", **conv_args)
    conv4_2 = conv2d(conv4_1, 512, name="vgg_19/conv4/conv4_2", **conv_args)
    conv4_3 = conv2d(conv4_2, 512, name="vgg_19/conv4/conv4_3", **conv_args)
    conv4_4 = conv2d(conv4_3, 512, name="vgg_19/conv4/conv4_4", **conv_args)
    pool4 = max_pool(conv4_4, stride=2, name='pool4', **pool_args)
    conv5_1 = conv2d(pool4, 512, name="vgg_19/conv5/conv5_1", **conv_args)
    conv5_2 = conv2d(conv5_1, 512, name="vgg_19/conv5/conv5_2", **conv_args)
    conv5_3 = conv2d(conv5_2, 512, name="vgg_19/conv5/conv5_3", **conv_args)
    conv5_4 = conv2d(conv5_3, 512, name="vgg_19/conv5/conv5_4", **conv_args)
    pool5 = max_pool(conv5_4, stride=2, name='pool5', **pool_args)

    fc6 = conv2d(pool5, 4096, filter_size=(7, 7),
                 name="vgg_19/fc6", **conv_args)
    fc6 = dropout(fc6, **common_args)
    fc7 = conv2d(fc6, 4096, filter_size=(1, 1), name="vgg_19/fc7", **conv_args)
    fc7 = dropout(fc7, **common_args)
    score_fr = conv2d(fc7, num_classes, filter_size=(1, 1),
                      name="score_fr", **conv_args)

    pred = tf.argmax(score_fr, axis=3)
    pool4_shape = pool4.get_shape().as_list()
    upscore2 = upsample2d(score_fr, [batch_size, pool4_shape[1], pool4_shape[2], num_classes], filter_size=(4, 4), stride=(2, 2),
                          name="deconv2d_1", w_init=initz.bilinear((4, 4, num_classes, num_classes)), **upsample_args)
    score_pool4 = conv2d(pool4, num_classes, filter_size=(1, 1),
                         name="score_pool4", **conv_args)
    fuse_pool4 = tf.add(upscore2, score_pool4)

    pool3_shape = pool3.get_shape().as_list()
    upscore4 = upsample2d(fuse_pool4, [batch_size, pool3_shape[1], pool3_shape[2], num_classes], filter_size=(4, 4), stride=(2, 2),
                          name="deconv2d_2", w_init=initz.bilinear((4, 4, num_classes, num_classes)), **upsample_args)
    score_pool3 = conv2d(pool3, num_classes, filter_size=(1, 1),
                         name="score_pool3", **conv_args)
    fuse_pool3 = tf.add(upscore4, score_pool3)
    input_shape = inputs.get_shape().as_list()
    upscore32 = upsample2d(fuse_pool3, [batch_size, input_shape[1], input_shape[2], num_classes], filter_size=(16, 16), stride=(8, 8),
                           name="deconv2d_3", w_init=initz.bilinear((16, 16, num_classes, num_classes)), **logits_args)
    logits = register_to_collections(tf.reshape(
        upscore32, shape=(-1, num_classes)), name='logits', **common_args)
    pred_up = tf.argmax(upscore32, axis=3)
    pred_up = register_to_collections(
        pred_up, name='final_prediction_map', **common_args)
    predictions = softmax(logits, name='predictions', **common_args)
    return end_points(is_training)
