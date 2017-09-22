# -------------------------------------------------------------------#
# Tool to save tenorflow model def file as GraphDef prototxt file
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
import os
import sys
import click
import random
from shutil import copyfile


@click.command()
@click.option('--class_names', default='', show_default=True,
              help="Comma-separated list of class names; class names corresponding to folder names with files")
@click.option('--class_labels', default=None, show_default=True,
              help="Comma-separated list of class labels.")
@click.option('--data_dir', default=None, show_default=True,
              help="Path to input data")
def format_data(data_dir, class_names, class_labels):

    classnames = [c.strip() for c in class_names.split(",")]
    classnames = [c for c in classnames if c]

    labels = [c.strip() for c in class_labels.split(",")]
    labels = [c for c in labels if c]

    data_locs = []
    for clsname in classnames:
        data_locs.append(os.path.join(data_dir, clsname))

    if not os.path.exists(os.path.join(data_dir, 'training')):
        os.mkdir(os.path.join(data_dir, 'training'))
    if not os.path.exists(os.path.join(data_dir, 'validation')):
        os.mkdir(os.path.join(data_dir, 'validation'))

    all_files = []
    for data_loc in data_locs:
        all_files.append(os.listdir(data_loc))

    cls_val_samples = []
    for cls_files in all_files:
        random.shuffle(cls_files)
        cls_val_samples.append(len(cls_files) / 10)

    with open(os.path.join(data_dir, 'training_labels.csv'), 'w') as f:
        f.write('image' + ',level\n')
        for idx, cls_files in enumerate(all_files):
            for fil in cls_files[cls_val_samples[idx]:]:
                if not os.path.isfile(os.path.join(data_dir, 'training', fil)):
                    copyfile(os.path.join(data_locs[idx], fil), os.path.join(
                        data_dir, 'training', fil))
                    f.write(fil[:-4] + ',' + str(labels[idx]) + '\n')
                else:
                    copyfile(os.path.join(data_locs[idx], fil), os.path.join(
                        data_dir, 'training', 'image_' + str(idx) + fil))
                    f.write('image_' + str(idx) +
                            fil[:-4] + ',' + str(labels[idx]) + '\n')

    with open(os.path.join(data_dir, 'validation_labels.csv'), 'w') as f:
        f.write('image' + ',level\n')
        for idx, cls_files in enumerate(all_files):
            for fil in cls_files[:cls_val_samples[idx]]:
                if not os.path.isfile(os.path.join(data_dir, 'validation', fil)):
                    copyfile(os.path.join(data_locs[idx], fil), os.path.join(
                        data_dir, 'validation', fil))
                    f.write(fil[:-4] + ',' + str(labels[idx]) + '\n')
                else:
                    copyfile(os.path.join(data_locs[idx], fil), os.path.join(
                        data_dir, 'validation', 'image_' + str(idx) + fil))
                    f.write('image_' + str(idx) +
                            fil[:-4] + ',' + str(labels[idx]) + '\n')

    with open(os.path.join(data_dir, 'training_labels.csv'), 'r+') as f:
        lines = f.readlines()
        random.shuffle(lines)
        f.seek(0)
        f.writelines(lines)


if __name__ == '__main__':
    format_data()
