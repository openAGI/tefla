import click
import tensorflow as tf

from tefla.utils.graph_utils import graph_from_pb, graph_from_pb_v2


@click.command()
@click.option('--graph_filename', default='', show_default=True, help='Input graph (.pb) filename')
@click.option(
    '--logdir', default='log', show_default=True, help='Directory to store the summary log')
def main(graph_filename, logdir):
  tf.logging.info('Extracting graph from pb file.')
  try:
    graph_from_pb(graph_filename, logdir)
  except Exception:
    graph_from_pb_v2(graph_filename, logdir)
  tf.logging.info('Extracting graph from pb file is successful.')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
