# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import os
import tensorflow as tf
from ..utils.util import rms


__all__ = ['summary_metric', 'summary_activation', 'create_summary_writer',
           'summary_param', 'summary_trainable_params', 'summary_gradients', 'summary_image']


def _formatted_name(tensor):
    return ':'.join(tensor.name.split(':')[:-1])


def summary_metric(tensor, name=None, collections=None):
    """
    Add summary to a tensor, scalar summary if the tensor is 1D, else scalar and histogram summary

    Args:
        tensor: a tensor to add summary
        name: name of the tensor
        collections: training or validation collections
    """
    if name is None:
        name = _formatted_name(tensor)
    ndims = tensor.get_shape().ndims
    with tf.name_scope('summary/metric'):
        tf.summary.scalar(
            name + '/mean', tf.reduce_mean(tensor), collections=collections)
        if ndims >= 2:
            tf.summary.histogram(name, tensor, collections=collections)


def summary_activation(tensor, name=None, collections=None):
    """
    Add summary to a tensor, scalar summary if the tensor is 1D, else  scalar and histogram summary

    Args:
        tensor: a tensor to add summary
        name: name of the tensor
        collections: training or validation collections
    """
    if name is None:
        name = _formatted_name(tensor)
    ndims = tensor.get_shape().ndims
    with tf.name_scope('summary/activation'):
        if ndims >= 2:
            tf.summary.histogram(name, tensor)
        tf.summary.scalar(name + '/sparsity',
                          tf.nn.zero_fraction(tensor), collections=collections)
        tf.summary.scalar(name + '/rms', rms(tensor), collections=collections)


def create_summary_writer(summary_dir, sess):
    """
    creates the summar writter for training and validation

    Args:
        summary_dir: the directory to write summary
        sess: the session to sun the ops

    Returns:
        training and vaidation summary writter
    """
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    if not os.path.exists(summary_dir + '/train'):
        os.mkdir(summary_dir + '/train')
    if not os.path.exists(summary_dir + '/test'):
        os.mkdir(summary_dir + '/test')
    train_writer = tf.summary.FileWriter(
        summary_dir + '/train', graph=sess.graph)
    val_writer = tf.summary.FileWriter(summary_dir + '/test', graph=sess.graph)
    return train_writer, val_writer


def summary_param(op, tensor, ndims, name, collections=None):
    """
    Add summary as per the ops mentioned

    Args:
        op: name of the summary op; e.g. 'stddev'
            available ops: ['scalar', 'histogram', 'sparsity', 'mean', 'rms', 'stddev', 'norm', 'max', 'min']
        tensor: the tensor to add summary
        ndims: dimension of the tensor
        name: name of the op
        collections: training or validation collections
    """
    return {
        'scalar': tf.summary.scalar(name, tensor, collections=collections) if ndims == 0 else tf.summary.scalar(name + '/mean', tf.reduce_mean(tensor), collections=collections),
        'histogram': tf.summary.histogram(name, tensor, collections=collections) if ndims >= 2 else None,
        'sparsity': tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor), collections=collections),
        'mean': tf.summary.scalar(name + '/mean', tf.reduce_mean(tensor), collections=collections),
        'rms': tf.summary.scalar(name + '/rms', rms(tensor), collections=collections),
        'stddev': tf.summary.scalar(name + '/stddev', tf.sqrt(tf.reduce_sum(tf.square(tensor - tf.reduce_mean(tensor, name='mean_op'))), name='stddev_op'), collections=collections),
        'max': tf.summary.scalar(name + '/max', tf.reduce_max(tensor), collections=collections),
        'min': tf.summary.scalar(name + '/min', tf.reduce_min(tensor), collections=collections),
        'norm': tf.summary.scalar(name + '/norm', tf.sqrt(tf.reduce_sum(tensor * tensor)), collections=collections),
    }[op]


def summary_trainable_params(summary_types, collections=None):
    """
    Add summary to all trainable tensors

    Args:
        summary_type: a list of all sumary types to add
            e.g.: ['scalar', 'histogram', 'sparsity', 'mean', 'rms', 'stddev', 'norm', 'max', 'min']
        collections: training or validation collections
    """
    params = tf.trainable_variables()
    with tf.name_scope('summary/trainable'):
        for tensor in params:
            name = _formatted_name(tensor)
            ndims = tensor.get_shape().ndims
            for s_type in summary_types:
                summary_param(s_type, tensor, ndims, name,
                              collections=collections)


def summary_gradients(grad_vars, summary_types, collections=None):
    """
    Add summary to all gradient tensors

    Args:
        grads_vars: grads and vars list
        summary_type: a list of all sumary types to add
            e.g.: ['scalar', 'histogram', 'sparsity', 'mean', 'rms', 'stddev', 'norm', 'max', 'min']
        collections: training or validation collections
    """
    with tf.name_scope('summary/gradient'):
        for grad, var in grad_vars:
            ndims = grad.get_shape().ndims
            for s_type in summary_types:
                summary_param(s_type, grad, ndims, var.op.name +
                              '/grad', collections=None)
        try:
            tf.summary.scalar('/global_norm', tf.global_norm(
                map(lambda grad_v: grad_v[0], grad_vars)), collections=collections)
        except Exception:
            return


def summary_image(tensor, name=None, max_images=10, collections=None):
    """
    Add image summary to a image tensor

    Args:
        tensor: a tensor to add summary
        name: name of the tensor
        max_images: num of images to add summary
        collections: training or validation collections
    """
    if name is None:
        name = name + _formatted_name(tensor)
    with tf.name_scope('summary/image'):
        tf.summary.image(name, tensor, max_outputs=max_images,
                         collections=collections)
