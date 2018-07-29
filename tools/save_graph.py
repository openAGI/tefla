# -------------------------------------------------------------------#
# Tool to save tenorflow model def file as GraphDef prototxt file
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tefla.utils import util
import sys


def save_graph(model, name, output_dir, output_model):
  model_def = util.load_module(model)
  model = model_def.model
  try:
    with tf.Graph().as_default():
      sess = tf.Session()

      end_points_predict = model(is_training=False, reuse=None)
      inputs = end_points_predict['inputs']
      predictions = tf.identity(end_points_predict['predictions'], name=name)
      init = tf.global_variables_initializer()
      sess.run(init)
      tf.train.write_graph(sess.graph_def, output_dir, output_model)
      print('finsihes writing')
      sess.close()
  except Exception as e:
    print(e.message)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input_model", type=str, default="", help="TensorFlow \'Model Def\' file to load.")
  parser.add_argument(
      "--model_name", type=str, default="", help="model name o use for output node.")
  parser.add_argument(
      "--output_dir", type=str, default=".", help="Directory to save the graph def pb.")
  parser.add_argument(
      "--output_model",
      type=str,
      default="output.pb",
      help="TensorFlow \'Model Def\' pb output file name.")
  args, unparsed = parser.parse_known_args()
  save_graph(args.input_model, args.model_name, args.output_dir, args.output_model)
