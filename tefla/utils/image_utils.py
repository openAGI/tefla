import numpy as np
from PIL import Image
import cv2
import h5py
import os
import math
import random
import pprint
import scipy.misc
import tensorflow as tf
from time import gmtime, strftime

pp = pprint.PrettyPrinter()


def preproc(data):
    assert (data.shape[0] == 3)
    img = rgb2gray(data)
    img = im_normalized(img)
    img = clahe_equalized(img)
    img = adjust_gamma(img, 1.2)
    img = img / 255.
    return img


def histo_equalized(img):
    assert (imgs.shape[0] == 1)
    img_equalized = cv2.equalizeHist(
        np.array(img, dtype=np.uint8))
    return img_equalized


def clahe_equalized(img):
    assert (img.shape[0] == 1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_equalized = clahe.apply(
        np.array(img, dtype=np.uint8))
    return imgs_equalized


def im_normalized(img):
    assert (img.shape[0] == 1)
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img - img_mean) / img_std
    img_normalized = ((img_normalized - np.min(img_normalized)) / (
        np.max(img_normalized) - np.min(img_normalized))) * 255
    return img_normalized


def adjust_gamma(img, gamma=1.0):
    assert (img.shape[0] == 1)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype("uint8")
    new_img = cv2.LUT(np.array(img, dtype=np.uint8), table)
    return new_img


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["image"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def rgb2gray(rgb):
    assert (rgb.shape[0] == 3)
    bn_img = rgb[0, :, :] * 0.299 + \
        rgb[1, :, :] * 0.587 + rgb[2, :, :] * 0.114
    bn_img = np.reshape(bn_img, (1, rgb.shape[1], rgb.shape[2]))
    return bn_img


def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)


def save_images(images, size, image_path, gray=False):
    return imsave(inverse_transform(images), size, image_path, gray=gray)


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size, gray=False):
    h, w = images.shape[1], images.shape[2]
    if gray:
        img = np.zeros((h * size[0], w * size[1]))
    else:
        img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image

    return img


def imsave(images, size, path, gray=False):
    return scipy.misc.imsave(path, merge(images, size, gray=gray))


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])


def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.
