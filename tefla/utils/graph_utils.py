import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat


def graph_from_pb(graph_filename, logdir):
  with tf.Session() as sess:
    with gfile.GFile(graph_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      g_in = tf.import_graph_def(graph_def)

  train_writer = tf.summary.FileWriter(logdir)
  train_writer.add_graph(sess.graph)
  return graph_def


def graph_from_pb_v2(graph_filename, logdir):
  with tf.Session() as sess:
    with gfile.GFile(graph_filename, 'rb') as f:
      data = compat.as_bytes(f.read())
      sm = saved_model_pb2.SavedModel()
      sm.ParseFromString(data)
      g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
  train_writer = tf.summary.FileWriter(logdir)
  train_writer.add_graph(sess.graph)
