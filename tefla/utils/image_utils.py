import numpy as np
from PIL import Image
import cv2
import h5py
import os
import math
import random
import pickle
from tqdm import tqdm
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


class PerspectiveTransformer:

    def __init__(self, src, dst):
        """Perspective and Inverse perspective transformer

        Args:
            src: Source coordinates for perspective transformation
            dst: Destination coordinates for perspective transformation
        """
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img, matrix):
        return cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def transform(self, img, offset=0):
        if offset == 0:
            return self.warp(img, self.M)
        else:
            src = self.src.copy()
            src[:, 0] = src[:, 0] + offset

            dst = self.dst.copy()
            dst[:, 0] = dst[:, 0] + offset

            M_inv = cv2.getPerspectiveTransform(src, dst)
            return self.warp(img, M_inv)

    def inverse_transform(self, img, offset=0):
        if offset == 0:
            return self.warp(img, self.M_inv)
        else:
            src = self.src.copy()
            src[:, 0] = src[:, 0] + offset

            dst = self.dst.copy()
            dst[:, 0] = dst[:, 0] + offset

            M_inv = cv2.getPerspectiveTransform(dst, src)
            return self.warp(img, M_inv)


def calculate_camera_calibration(calib_path, rows, cols, cal_image_size):
    """Calculates the camera calibration based on chessboard images.

    Args:
        calib_path: calibration data (imgs) dir path
        rows: number of rows on chessboard
        cols: number of columns on chessboard

    Returns:
        a `dict` with calibration points
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob(calib_path)
    cal_images = np.zeros((len(images), *cal_image_size), dtype=np.uint8)

    successfull_cnt = 0
    for idx, fname in enumerate(tqdm(images, desc='Processing image')):
        img = scipy.misc.imread(fname)
        if img.shape[0] != cal_image_size[0] or img.shape[1] != cal_image_size[1]:
            img = scipy.misc.imresize(img, cal_image_size)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            successfull_cnt += 1

            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
            cal_images[idx] = img

    print("%s/%s camera calibration images processed." %
          (successfull_cnt, len(images)))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, cal_image_size[:-1], None, None)

    calibration = {'objpoints': objpoints,
                   'imgpoints': imgpoints,
                   'cal_images': cal_images,
                   'mtx': mtx,
                   'dist': dist,
                   'rvecs': rvecs,
                   'tvecs': tvecs}

    return calibration


def get_camera_calibration(cal_calib_points, calib_data_path=None, rows=6, cols=9, cal_image_shape=None, calib_out_path=None):
    """The camera calibration will be
    calculated and stored on disk or loaded.
    """
    if cal_calib_points:
        calibration = calculate_camera_calibration(
            calib_data_path, rows, cols, cal_image_shape)
        with open(calib_out_path, 'wb') as f:
            pickle.dump(calibration, file=f)
    else:
        with open(calib_out_path, 'rb') as f:
            calibration = pickle.load(f)
    return calibration


class CameraCalibrator:

    def __init__(self, calibration=None, calib_data_path=None, rows=6, cols=9, cal_image_shape=None):
        """Helper class to remove lens distortion from images (camera calibration)

        Args:
            calibration: precalculated calibration matrices
            calib_data_path: path to data for camera calibration
            rows: number of rows on chessboard
            cols: number of columns on chessboard
            cal_image_shape: calibration image shape
        """
        if calibration is not None:
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                calibration['objpoints'], calibration['imgpoints'], cal_image_shape, None, None)
        else:
            calibration = calculate_camera_calibration(
                calib_data_path, rows, cols, cal_image_shape)
            self.mtx = calibration['mtx']
            self.dist = calibration['dist']

    def undistort(self, img):
        """ Restore image from camera distrotation using calibration matrics

        Args:
            img: input image

        Returns:
            restored image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
