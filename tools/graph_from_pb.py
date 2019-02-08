import click
from tefla.utils.graph_utils import graph_from_pb


@click.command()
@click.option('--graph_filename', default='', show_default=True, help='Input graph (.pb) filename')
@click.option(
    '--logdir', default='log', show_default=True, help='Directoey to store the summary log')
def main(graph_filename, logdir):
  graph_from_pb(graph_filename, logdir)


if __name__ == '__main__':
  main()
