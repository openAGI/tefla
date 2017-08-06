"""Resize and crop images to square, save as tiff."""
# -------------------------------------------------------------------#
# Tool to convert labels
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division, print_function

import os
from PIL import Image, ImageFilter
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import scipy.misc
from skimage.io import imread, imsave


import click
import numpy as np

from tefla.da import data

N_PROC = cpu_count()


def pascal_classes():
    classes = {'aeroplane': 1,  'bicycle': 2,  'bird': 3,  'boat': 4,
               'bottle': 5,  'bus': 6,  'car': 7,  'cat': 8,
               'chair': 9,  'cow': 10, 'diningtable': 11, 'dog': 12,
               'horse': 13, 'motorbike': 14, 'person': 15, 'potted-plant': 16,
               'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}

    return classes


def pascal_palette():
    palette = {(0,   0,   0): 0,
               (128,   0,   0): 1,
               (0, 128,   0): 2,
               (128, 128,   0): 3,
               (0,   0, 128): 4,
               (128,   0, 128): 5,
               (0, 128, 128): 6,
               (128, 128, 128): 7,
               (64,   0,   0): 8,
               (192,   0,   0): 9,
               (64, 128,   0): 10,
               (192, 128,   0): 11,
               (64,   0, 128): 12,
               (192,   0, 128): 13,
               (64, 128, 128): 14,
               (192, 128, 128): 15,
               (0,  64,   0): 16,
               (128,  64,   0): 17,
               (0, 192,   0): 18,
               (128, 192,   0): 19,
               (0,  64, 128): 20
               }

    return palette


def palette_demo():
    palette_list = pascal_palette().keys()
    palette = ()

    for color in palette_list:
        palette += color

    return palette


def convert_labels(label_image, image_height, image_width):
    arr_3d = label_image
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = pascal_palette()

    for i in range(0, arr_3d.shape[0]):
        for j in range(0, arr_3d.shape[1]):
            key = (arr_3d[i, j, 0], arr_3d[i, j, 1], arr_3d[i, j, 2])
            arr_2d[i, j] = palette.get(key, 0)
    print(sum(sum(arr_2d)))
    # import matplotlib.pyplot as plt
    # plt.imshow(arr_2d)
    # plt.show()
    return arr_2d


def convert_seg_labels(label_file, image_height, image_width):
    image = scipy.misc.imread(label_file, mode='RGB')
    print(label_file)
    # arr_3d = scipy.misc.imresize(image, size=(
    #    image_height, image_width), interp='cubic')
    arr_3d = image
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = pascal_palette()

    for i in range(0, arr_3d.shape[0]):
        for j in range(0, arr_3d.shape[1]):
            key = (arr_3d[i, j, 0], arr_3d[i, j, 1], arr_3d[i, j, 2])
            arr_2d[i, j] = palette.get(key, 0)
    print(sum(sum(arr_2d)))
    # import matplotlib.pyplot as plt
    # plt.imshow(arr_2d)
    # plt.show()
    return arr_2d


def convert_to_one_hot_labels(label_file):
    image = scipy.misc.imread(label_file, mode='RGB')
    image = scipy.misc.imresize(image, size=(
        image_height, image_width), interp='cubic')
    gt_classes = []
    palette_list = pascal_palette().keys()
    for cls in palette_list:
        gt_classes.append(np.all(image == cls, axis=2))
    gt_classes = np.asarray(gt_classes).transpose(1, 2, 0)
    return gt_classes


def get_convert_fname(fname, extension, directory, convert_directory):
    def replace_last(s, o, n):
        return "%s%s" % (s[0:s.rfind(o)], n)

    def replace_first(s, o, n):
        return s.replace(o, n, 1)

    if not directory.endswith("/"):
        directory += "/"

    if not convert_directory.endswith("/"):
        convert_directory += "/"

    extension0 = fname.split("/")[-1].split(".")[-1]
    # print("file: %s, old ext: %s, new ext: %s, old dir: %s, new dir: %s" % (
    #     fname, extension0, extension, directory, convert_directory))
    fname2 = replace_last(fname, extension0, extension)
    return replace_first(fname2, directory, convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_height, crop_width, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory,
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_height, crop_width)
        save(img, convert_fname)


def save(img, fname):
    imsave(fname, img)


@click.command()
@click.option('--directory', default='data/train', show_default=True,
              help="Directory with original images.")
@click.option('--convert_directory', default='data/train_res', show_default=True,
              help="Where to save converted images.")
@click.option('--test', is_flag=True, default=False, show_default=True,
              help="Convert images one by one and examine them on screen.")
@click.option('--crop_height', default=512, show_default=True,
              help="Size of converted images.")
@click.option('--crop_width', default=512, show_default=True,
              help="Size of converted images.")
@click.option('--extension', default='png', show_default=True,
              help="Filetype of converted images.")
def main(directory, convert_directory, test, crop_height, crop_width, extension):
    try:
        os.mkdir(convert_directory)
    except OSError:
        pass

    supported_extensions = set(['jpg', 'png', 'tiff', 'jpeg', 'tif'])

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
                 for f in fn if f.split('.')[-1].lower() in supported_extensions]
    filenames = sorted(filenames)

    print("Resizing images in {} to {}, this takes a while."
          "".format(directory, convert_directory))

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool on my machine
    batchsize = 500
    batches = n // batchsize + 1
    pool = Pool(N_PROC)

    args = []

    for f in filenames:
        args.append((convert_seg_labels, (directory, convert_directory, f, crop_height, crop_width,
                                          extension)))

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')


if __name__ == '__main__':
    main()
