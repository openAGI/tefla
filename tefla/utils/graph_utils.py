import tensorflow as tf
from tensorflow.python.platform import gfile


def graph_from_pb(graph_filename, logdir):
  with tf.Session() as sess:
    with gfile.FastGFile(graph_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      g_in = tf.import_graph_def(graph_def)

  train_writer = tf.summary.FileWriter(logdir)
  train_writer.add_graph(sess.graph)
  return graph_def
