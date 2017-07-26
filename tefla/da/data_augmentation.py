# -------------------------------------------------------------------#
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
"""Read and preprocess image data.
 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.
 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.
 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.
 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.
 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import scipy.ndimage
import tensorflow as tf
from ..core import logger as log


def inputs(dataflow, tfrecords_image_size, crop_size, im_size=None, batch_size=None, num_preprocess_threads=32, num_readers=1, image_preprocessing=None):
    """Generate batches of ImageNet images for evaluation.

    Args:
        dataflow: instance of Dataflow class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
        None defaults to FLAGS.num_preprocess_threads.

    Returns:
        images: Images. 4D tensor of size [batch_size, cfg.TRAIN.image_size,cfg.TRAIN.image_size, 3].
        labels: 1-D integer Tensor of [cfg.TRAIN.batch_size].
    """
    with tf.device('/cpu:0'):
        images, labels = dataflow.batch_inputs(batch_size, False, tfrecords_image_size, crop_size,
                                               im_size=im_size, num_preprocess_threads=num_preprocess_threads, image_preprocessing=image_preprocessing)

    return images, labels


def distorted_inputs(dataflow, tfrecords_image_size, crop_size, im_size=None, batch_size=None, num_preprocess_threads=32, num_readers=1, target_probs=None, init_probs=None, image_preprocessing=None, data_balancing=0):
    """Generate batches of distorted versions of Training images.

    Args:
        dataflow: instance of Dataflow class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
            None defaults to cfg.num_preprocess_threads.

    Returns:
        images: Images. 4D tensor of size [batch_size, cfg.TRAIN.crop_image_size, cfg.TRAIN.crop_image_size, 3].
        labels: 1-D integer Tensor of [cfg.TRAIN.batch_size].
    """
    with tf.device('/cpu:0'):
        images, labels = dataflow.get_batch(batch_size, target_probs, tfrecords_image_size, crop_size=crop_size,
                                            resize_size=im_size, num_preprocess_threads=num_preprocess_threads, image_preprocessing=image_preprocessing, init_probs=init_probs, data_balancing=data_balancing)
    return images, labels


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.

    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: Tensor containing single image.
        thread_id: preprocessing thread ID.
        scope: Optional scope for name_scope.

    Returns:
        color-distorted image
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        # image = tf.clip_by_value(image, 0.0, 1.0)
        return image


VGG_MEAN = [103.939, 116.779, 123.68]


def vggnet_input(im_tf):
    im_tf = tf.image.convert_image_dtype(im_tf, dtype=tf.float32)
    # im_tf = tf.image.central_crop(im_tf, central_fraction=0.875)
    # im_tf = tf.expand_dims(im_tf, 0)
    # im_tf = tf.image.resize_bilinear(im_tf, [224, 224], align_corners=False)
    # im_tf = tf.squeeze(im_tf, [0])
    im_tf = tf.subtract(im_tf, 0.5)
    im_tf = tf.multiply(im_tf, 2.0)
    im_tf = im_tf * 255.0
    r_, g_, b_ = tf.split(im_tf, 3, axis=2)
    r_ = r_ - VGG_MEAN[2]
    g_ = b_ - VGG_MEAN[1]
    b_ = b_ - VGG_MEAN[0]
    im_tf = tf.concat([r_, g_, b_], axis=2)
    # im_tf = tf.expand_dims(im_tf, 0)
    return im_tf


def random_crop(image, crop_size, padding=None):
    """Randmly crop a image.

    Args:
        image: 3-D float Tensor of image
        crop_size:int/tuple, output image height, width, for deep network we prefer same width and height
        padding: int, padding use to restore original image size, padded with 0's

    Returns:
        3-D float Tensor of randomly flipped updown image used for training.
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    oshape = np.shape(image)
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    npad = ((padding, padding), (padding, padding), (0, 0))
    modified_image = image
    if padding:
        modified_image = np.lib.pad(
            image, pad_width=npad, mode='constant', constant_values=0)
    nh = random.randint(0, oshape[0] - crop_size[0])
    nw = random.randint(0, oshape[1] - crop_size[1])
    modified_image = modified_image[nh:nh + crop_size[0], nw:nw + crop_size[1]]
    return modified_image


def random_flip_leftright(image):
    """Randmly flip one image updown.

    Args:
        image: 3-D float Tensor of image

    Returns:
        3-D float Tensor of randomly flipped left right image used for training.
    """
    if bool(random.getrandbits(1)):
        image = np.fliplr(image)
    return image


def random_flip_updown(image):
    """Randmly flip one image updown.

    Args:
        image: 3-D float Tensor of image

    Returns:
        3-D float Tensor of randomly flipped updown image used for training.
    """
    if bool(random.getrandbits(1)):
        image = np.flipud(image)
    return image


def random_rotation(image, max_angle):
    """Randmly rotate one image. Random rotation introduces rotation invarant in the image.

    Args:
        image: 3-D float Tensor of image
        max_angle: float, max value of rotation

    Returns:
        3-D float Tensor of randomly rotated image used for training.
    """
    if bool(random.getrandbits(1)):
        angle = random.uniform(-max_angle, max_angle)
        image = scipy.ndimage.interpolation.rotate(image, angle, reshape=False)
    return image


def random_blur(image, sigma_max):
    """Randmly blur one image with gaussian blur. Bluring reduces noise present in the image.

    Args:
        image: 3-D float Tensor of image
        sigma_max: maximum value of standard deviation to use

    Returns:
        3-D float Tensor of randomly blurred image used for training.
    """
    if bool(random.getrandbits(1)):
        sigma = random.uniform(0., sigma_max)
        image = scipy.ndimage.filters.gaussian_filter(image, sigma)
    return image


def distort_image(image, crop_size, im_size=None, thread_id=0, scope=None):
    """Distort one image for training a network.

    Args:
        image: 3-D float `Tensor` of image
        im_size: 1-D int `Tensor` of 2 elements, image height and width, for real time resizing
        scope: Optional scope for name_scope.

    Returns:
        3-D float `Tensor` of distorted image used for training.
    """
    with tf.name_scope(scope, 'distort_image', [image, crop_size, im_size]):
        # Crop the image to the specified bounding box.
        # Resize image as per memroy constarints
        # if not isinstance(crop_size, tf.Tensor):
        #    crop_size = tf.convert_to_tensor(crop_size)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if im_size is not None:
            if not isinstance(im_size, tf.Tensor):
                im_size = tf.convert_to_tensor(im_size)
            resize_method = thread_id % 4
            image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_images(
                image, im_size, resize_method)
            image = tf.squeeze(image)

        distorted_image = tf.random_crop(
            image, [crop_size[0], crop_size[1], 3], 12345)

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # Randomly flip the image up and down.
        distorted_image = tf.image.random_flip_up_down(distorted_image)
        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)
        return distorted_image


def eval_image(image, crop_size, im_size=None, thread_id=0, scope=None):
    """Prepare one image for evaluation.

    Args:
        image: 3-D float Tensor
        im_size: 1-D int `Tensor` of 2 elements, image height and width, for real time resizing
        crop_size: 1-D int `Tensor` or `Tuple` or single int of 2 elemnts,  image crop height and width, for training crops
        scope: Optional scope for name_scope.

    Returns:
        3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, crop_size]):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if im_size is not None:
            if not isinstance(im_size, tf.Tensor):
                im_size = tf.convert_to_tensor(im_size)
            resize_method = thread_id % 4
            image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_images(
                image, im_size, resize_method)
            image = tf.squeeze(image)
        image = tf.image.central_crop(image, central_fraction=0.875)

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if not isinstance(crop_size, tf.Tensor):
            crop_size = tf.convert_to_tensor(crop_size)
        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
            image, crop_size, align_corners=False)
        image = tf.squeeze(image, [0])
        return image


def image_preprocessing(image, train, crop_size, im_size=None, thread_id=0, bbox=None):
    """Decode and preprocess one image for evaluation or training.

    Args:
        image_buffer: JPEG encoded string Tensor
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax].
        train: boolean
        im_size: 1-D int `Tensor` of 2 elements, image height and width, for real time resizing
        crop_size: 1-D int `Tensor` or `Tuple` or single int of 2 elemnts,  image crop height and width, for training crops
        thread_id: integer indicating preprocessing thread

    Returns:
        3-D float Tensor containing an appropriately scaled image
    """

    # image = decode_jpeg(image_buffer)
    if im_size is not None:
        if not isinstance(im_size, tf.Tensor):
            im_size = tf.convert_to_tensor(im_size)

    if train:
        image = distort_image(
            image, crop_size, im_size=im_size, thread_id=thread_id)
    else:
        image = eval_image(image, crop_size, im_size=im_size,
                           thread_id=thread_id)
    image = tf.image.per_image_standardization(image)

    return image


def random_image_scaling(image, label):
    """Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    scale = tf.random_uniform(
        [1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=1)
    image = tf.image.resize_images(image, new_shape)
    label = tf.image.resize_nearest_neighbor(
        tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, axis=0)

    return image, label


def random_flip_left_right(image, label, im_shape=(512, 512, 3), label_shape=(512, 512), seed=1234):
    """Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = tf.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
    label = tf.cond(mirror_cond, lambda: tf.reverse(label, [1]), lambda: label)
    return image, label


def random_flip_up_down(image, label, im_shape=(512, 512, 3), label_shape=(512, 512), seed=1234):
    """Randomly flip up/down the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = tf.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [0]), lambda: image)
    label = tf.cond(mirror_cond, lambda: tf.reverse(label, [0]), lambda: label)
    return image, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = tf.expand_dims(label, 2)
    label = label - ignore_label
    combined = tf.concat([image, label], axis=2)
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(
        crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    # last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    label_crop = tf.squeeze(label_crop, axis=[2])
    # img_crop.set_shape((crop_h, crop_w, 3))
    # label_crop.set_shape((crop_h, crop_w))
    return img_crop, label_crop


def seg_input_aug(image, label, crop_h=448, crop_w=448):
    """
    image, label = random_flip_left_right(image, label)
    image, label = random_flip_up_down(image, label)
    image, label = random_crop_and_pad_image_and_labels(
        image, label, crop_h, crop_w)
    """
    image = tf.image.resize_images(
        image, [448, 448], method=0)
    label = tf.expand_dims(label, 2)
    label = tf.image.resize_images(
        label, [448, 448], method=0)
    label = tf.squeeze(label)
    # image = tf.reshape(image, (448, 448, 3))
    return image, label
