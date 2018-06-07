import click
import mltest
import tensorflow as tf
import numpy as np

from tefla.utils import util


def setup():
  mltest.setup()


@click.command()
@click.option('--model', default=None, show_default=True, help='Relative path to model.')
@click.option('--input_shape', default='', show_default=True, help='Input shape')
@click.option('--loss_type', default='softmax', show_default=True, help='Loss fuction type.')
def test_model(model, input_shape, loss_type='softmax'):
  print(input_shape)
  input_shape = [c.strip() for c in input_shape.split(",")]
  model_def = util.load_module(model)
  model = model_def.model
  inputs = tf.placeholder(tf.float32, input_shape)
  end_points = model(inputs, True, None)
  logits = end_points['logits']
  labels = tf.placeholder(tf.int32, (None, logits.get_shape().as_list()[1]))
  prediction = end_points['prediction']
  if loss_type == 'sigmoid':
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
  else:
    loss = tf.losses.softmax_cross_entropy(labels, logits)
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(loss)
  feed_dict = {
      inputs: np.random.normal(size=inputs.get_shape().as_list()),
      labels: np.random.randint(0, high=1, size=logits.get_shape().as_list())
  }
  mltest.test_suite(prediction, train_op, feed_dict=feed_dict, output_range=(0, 1))


if __name__ == '__main__':
  test_model()
