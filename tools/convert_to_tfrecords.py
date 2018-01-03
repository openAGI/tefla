# -------------------------------------------------------------------#
# Tool to save tenorflow model def file as GraphDef prototxt file
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
import os
import click

from tefla.dataset.image_to_tfrecords import TFRecords


@click.command()
@click.option('--records_name', default='dataset_serealized', show_default=True,
              help="TFRecords output file name.")
@click.option('--num_shards', default=8, show_default=True,
              help="Number of shards for database.")
@click.option('--data_dir', default='data/train', show_default=True,
              help="Datset dir with jpeg/png images.")
@click.option('--output_data_dir', default='data/train', show_default=True,
              help="Datset dir with jpeg/png images.")
@click.option('--label_file', show_default=True,
              help="Path to the label file.")
def process_dataset(records_name, num_shards, data_dir, output_data_dir, label_file):
    im2r = TFRecords()
    im2r.process_dataset(records_name, data_dir,
                         output_data_dir, num_shards, label_file)


if __name__ == '__main__':
    process_dataset()
