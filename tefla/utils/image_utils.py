import numpy as np
from PIL import Image
import cv2
import h5py


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
