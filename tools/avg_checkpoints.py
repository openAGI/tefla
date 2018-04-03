# -------------------------------------------------------------------#
# Tool to save tenorflow model def file as GraphDef prototxt file
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import click
import numpy as np
import six
from six.moves import zip
import tensorflow as tf


def checkpoint_exists(path):
  return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta")
          or tf.gfile.Exists(path + ".index"))


@click.command()
@click.option(
    '--checkpoints',
    default='',
    show_default=True,
    help="Comma-separated list of checkpoints to average.")
@click.option(
    '--weights',
    default=None,
    show_default=True,
    help="Comma-separated list of weights for corresponding checkpoints.")
@click.option(
    '--prefix',
    default=None,
    show_default=True,
    help="Prefix (e.g., directory) to append to each checkpoint.")
@click.option(
    '--output_path',
    default='data/train',
    show_default=True,
    help="Path to output the averaged checkpoint to.")
def checkpoint_mean(checkpoints, prefix, output_path, weights):
  checkpoints = [c.strip() for c in checkpoints.split(",")]
  checkpoints = [c for c in checkpoints if c]
  if not checkpoints:
    raise ValueError("No checkpoints provided for averaging.")
  if prefix:
    checkpoints = [os.path.join(prefix, c) for c in checkpoints]
  checkpoints = [c for c in checkpoints if checkpoint_exists(c)]

  var_list = tf.contrib.framework.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if not name.startswith("global_step"):
      var_values[name] = np.zeros(shape)
  if weights is None:
    weights = np.ones(len(checkpoints), dtype=np.float32)
  else:
    weights = np.array([float(w.strip()) for w in weights.split(",")])
    print(weights)
  for idx, checkpoint in enumerate(checkpoints):
    reader = tf.contrib.framework.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      var_dtypes[name] = tensor.dtype
      var_values[name] += tensor * weights[idx]
    print("Read from checkpoint %s", checkpoint)
  for name in var_values:  # Average.
    var_values[name] /= len(checkpoints)

  tf_vars = [tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v]) for v in var_values]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops, six.iteritems(var_values)):
      sess.run(assign_op, {p: value})
    saver.save(sess, output_path, global_step=global_step)

  print("Averaged checkpoints saved in %s", output_path)


if __name__ == '__main__':
  checkpoint_mean()
