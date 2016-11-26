# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf
import os

from utils.utils import rms


def _formatted_name(tensor):
    return ':'.join(tensor.name.split(':')[:-1])


def summary_metric(tensor, name=None):
    if name is None:
        name = _formatted_name(tensor)
    ndims = tensor.get_shape().ndims
    with tf.name_scope('summary/metric'):
        tf.scalar_summary(name + '/mean', tf.reduce_mean(tensor))
        if ndims >= 2:
            tf.histogram_summary(name, tensor)


def summary_activation(tensor, name=None):
    if name is None:
        name = _formatted_name(tensor)
    ndims = tensor.get_shape().ndims
    with tf.name_scope('summary/activation'):
        if ndims >= 2:
            tf.histogram_summary(name, tensor)
        tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(tensor))
        tf.scalar_summary(name + '/rms', rms(tensor))


def create_summary_writer(summary_dir, sess):
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    if not os.path.exists(summary_dir + '/train'):
        os.mkdir(summary_dir + '/train')
    if not os.path.exists(summary_dir + '/test'):
        os.mkdir(summary_dir + '/test')
    train_writer = tf.train.SummaryWriter(summary_dir + '/train', graph=sess.graph)
    val_writer = tf.train.SummaryWriter(summary_dir + '/test', graph=sess.graph)
    return train_writer, val_writer


def summary_param(op, tensor, ndims, name):
    return {
        'scalar': tf.scalar_summary(name, tensor) if ndims == 0 else tf.scalar_summary(name + '/mean', tf.reduce_mean(tensor)),
        'histogram': tf.histogram_summary(name, tensor) if ndims >= 2 else None,
        'sparsity': tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(tensor)),
        'mean': tf.scalar_summary(name + '/mean', tf.reduce_mean(tensor)),
        'rms': tf.scalar_sumamry(name + '/rms', rms(tensor)),
        'stddev': tf.scalar_sumamry(name + '/stddev', tf.sqrt(tf.reduce_sum(tf.square(tensor - tf.reduce_mean(tensor, name='mean_op'))), name='stddev_op')),
        'max': tf.scalar_summary(name + '/max', tf.reduce_max(tensor)),
        'min': tf.scalar_summary(name + '/min', tf.reduce_min(tensor)),
        'norm': tf.scalar_summary(name + '/norm', tf.srqt(tf.reduce_sum(tensor * tensor))),
    }[op]


def summary_trainable_params(summary_types):
    params = tf.trainable_variables()
    with tf.name_scope('summary/trainable'):
        for tensor in params:
            name = _formatted_name(tensor)
            ndims = tensor.get_shape().ndims
            for s_type in summary_types:
                summary_param(s_type, tensor, ndims, name)


def summary_gradients(grad_vars, summary_types):
    with tf.name_scope('summary/gradient'):
        for grad, var in grad_vars:
            ndims = grad.get_shape().ndims
            for s_type in summary_types:
                summary_param(s_type, grad, ndims, var.op.name + '/grad')
        try:
            tf.scalar_summary('/global_norm', tf.global_norm(map(lambda grad_v: grad_v[0], grad_vars)))
        except:
            return


def summary_image(tensor, name=None):
    if name is None:
        name = name + _formatted_name(tensor)
    with tf.name_scope('summary/image'):
        tf.image_summary(name, tensor)
