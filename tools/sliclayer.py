# -------------------------------------------------------------------#
# Tool to save tenorflow model def file as GraphDef prototxt file
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
import click
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn.cluster import KMeans
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
from tefla.convert import convert


def process_image(image_filename, image_size):
    image = convert(image_filename, image_size)
    image = img_as_float(image)
    return image


def compute_hog(image, locations):
    hog = cv2.HOGDescriptor()
    winStride = (8, 8)
    padding = (8, 8)
    hist = hog.compute(image, winStride, padding, locations)
    return hist


def superpixel_smoothing(image, feature, numSegments, numClusters):
    segments = slic(image, n_segments=numSegments,
                    compactness=1.5, max_iter=50, sigma=8, convert2lab=True)
    # feature = np.random.randn(image.shape[0], image.shape[1], 32)
    for i in range(feature.shape[-1]):
        for j in range(numSegments):
            temp_idx = np.where(segments == j)
            slic_mean = feature[temp_idx[0], temp_idx[1], i].mean()
            feature[temp_idx[0], temp_idx[1], i] = slic_mean


def cross_sp_voting(image, segments, numSegments):
    hogfeats = []
    rgb = []
    lab = []
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    for i in range(numSegments):
        temp_idx = np.where(segments == i)
        locations = zip(tuple(temp_idx[0]), tuple(idx[1]))
        hogfeats.append(compute_hog(image, locations))
        rgb.append(image[temp_idx[0], temp_idx[1], :])
        lab.append(image_lab[temp_idx[0], temp_idx[1], :])


@click.command()
@click.option('--image_filename', show_default=True,
              help="path to image.")
@click.option('--num_segments', default=50, show_default=True,
              help="Number of segmented region for slic")
@click.option('--num_clusters', default=30, show_default=True,
              help="Num clusters")
@click.option('--image_size', default=448, show_default=True,
              help="Size of converted images.")
def main(image_filename, num_segments, num_clusters, image_size):
    image = process_image(image_filename, image_size)
    # superpixel_smoothing(image, num_segments, num_clusters)
    im = cv2.imread(image_filename)
    hist = compute_hog(im, [(100, 102), (200, 300)])
    print(hist.shape)


if __name__ == '__main__':
    main()
