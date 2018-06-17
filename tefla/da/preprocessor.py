import abc
import six
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


@six.add_metaclass(abc.ABCMeta)
class Preprocessor(object):

  def __init__(self, *args, **kwargs):
    super(Preprocessor, self).__init__()

  @abc.abstractmethod
  def preprocess_for_train(self, *args, **kwargs):
    raise NotImplementedError

  @abc.abstractmethod
  def preprocess_for_eval(self, *args, **kwargs):
    raise NotImplementedError

  def apply_with_random_selector(self, x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.

    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)
    ])[0]

  def preprocess_image(self,
                       image,
                       output_height,
                       output_width,
                       is_training,
                       resize_side_min=256,
                       resize_side_max=512,
                       **kwargs):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
        ignored. Otherwise, the resize side is sampled from
          [resize_size_min, resize_size_max].

    Returns:
      A preprocessed image.
    """
    if is_training:
      return self.preprocess_for_train(
          image,
          output_height,
          output_width,
          resize_side_min=resize_side_min,
          resize_side_max=resize_side_max,
          **kwargs)
    else:
      return self.preprocess_for_eval(
          image, output_height, output_width, resize_side_min=resize_side_min, **kwargs)


class SegPreprocessor(Preprocessor):

  def random_image_scaling(self, image, label):
    """Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=1)
    image = tf.image.resize_images(image, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, axis=0)

    return image, label

  def random_flip_left_right(self,
                             image,
                             label,
                             im_shape=(512, 512, 3),
                             label_shape=(512, 512),
                             seed=1234):
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

  def random_flip_up_down(self,
                          image,
                          label,
                          im_shape=(512, 512, 3),
                          label_shape=(512, 512),
                          seed=1234):
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

  def random_crop_and_pad_image_and_label(self, image, label, crop_h, crop_w, ignore_label=255):
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
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    label_crop = tf.squeeze(label_crop, axis=[2])
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w))
    return img_crop, label_crop

  def preprocess_for_train(self, image, label, output_height, output_width, standardizer=None):
    image, label = self.random_flip_left_right(image, label)
    image, label = self.random_flip_up_down(image, label)
    image = tf.image.resize_images(image, [output_height, output_width], method=0)
    label = tf.expand_dims(label, 2)
    label = tf.image.resize_images(label, [output_height, output_width], method=0)
    label = tf.squeeze(label)
    if standardizer is None:
      image = tf.image.per_image_standardization(image)
    else:
      image = standardizer(image, True)
    return image, label

  def preprocess_for_eval(self, image, label, output_height, output_width, standardizer=None):
    image = tf.image.resize_images(image, [output_height, output_width], method=0)
    label = tf.expand_dims(label, 2)
    label = tf.image.resize_images(label, [output_height, output_width], method=0)
    label = tf.squeeze(label)
    if standardizer is None:
      image = tf.image.per_image_standardization(image)
    else:
      image = standardizer(image, False)
    return image, label

  def preprocess_image(self,
                       image,
                       label,
                       output_height,
                       output_width,
                       is_training,
                       standardizer=None):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
        ignored. Otherwise, the resize side is sampled from
          [resize_size_min, resize_size_max].

    Returns:
      A preprocessed image.
    """
    if is_training:
      return self.preprocess_for_train(
          image, label, output_height, output_width, standardizer=standardizer)
    else:
      return self.preprocess_for_eval(
          image, label, output_height, output_width, standardizer=standardizer)


class VggPreprocessor(Preprocessor):

  def __init__(self):
    self._R_MEAN = 123.68
    self._G_MEAN = 116.78
    self._B_MEAN = 103.94
    self._RESIZE_SIDE_MIN = 256
    self._RESIZE_SIDE_MAX = 512

  def _crop(self, image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes. Note that
    the method doesn't assume we know the input image size but it does assume
    we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      the cropped (and resized) image.

    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
      cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
    with tf.control_dependencies([size_assertion]):
      image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)

  def _random_crop(self, image_list, crop_height, crop_width):
    """Crops the given list of images.
        The function applies the same crop to each image in the list. This can be
        effectively applied when there are multiple image inputs of the same
        dimension such as:
          image, depths, normals = _random_crop(
              [image, depths, normals], 120, 150)

        Args:
          image_list: a list of image tensors of the same dimension but possibly
            varying channel.
          crop_height: the new height.
          crop_width: the new width.

        Returns:
          the image_list with cropped images.

        Raises:
          ValueError: if there are multiple image inputs provided with different size
            or the images are smaller than the crop dimensions.
        """
    if not image_list:
      raise ValueError('Empty image_list.')

    rank_assertions = []
    for i in range(len(image_list)):
      image_rank = tf.rank(image_list[i])
      rank_assert = tf.Assert(
          tf.equal(image_rank, 3),
          ['Wrong rank for tensor  %s [expected] [actual]', image_list[i].name, 3, image_rank])
      rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
      image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height), tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
      image = image_list[i]
      asserts.append(rank_assertions[i])
      with tf.control_dependencies([rank_assertions[i]]):
        shape = tf.shape(image)
      height = shape[0]
      width = shape[1]

      height_assert = tf.Assert(
          tf.equal(height, image_height),
          ['Wrong height for tensor %s [expected][actual]', image.name, height, image_height])
      width_assert = tf.Assert(
          tf.equal(width, image_width),
          ['Wrong width for tensor %s [expected][actual]', image.name, width, image_width])
      asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
      max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
      max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    return [
        self._crop(image, offset_height, offset_width, crop_height, crop_width)
        for image in image_list
    ]

  def _central_crop(self, image_list, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      the list of cropped images.
    """
    outputs = []
    for image in image_list:
      image_height = tf.shape(image)[0]
      image_width = tf.shape(image)[1]

      offset_height = (image_height - crop_height) / 2
      offset_width = (image_width - crop_width) / 2

      outputs.append(self._crop(image, offset_height, offset_width, crop_height, crop_width))
    return outputs

  def _mean_image_subtraction(self, image, means):
    """Subtracts the given means from each image channel.
        For example:
          means = [123.68, 116.779, 103.939]
          image = _mean_image_subtraction(image, means)
        Note that the rank of `image` must be known.

        Args:
          image: a tensor of size [height, width, C].
          means: a C-vector of values to subtract from each channel.

        Returns:
          the centered image.

        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """
    if image.get_shape().ndims != 3:
      raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
      channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

  def _smallest_size_at_least(self, height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(
        tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width

  def _aspect_preserving_resize(self, image, smallest_side):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

  def preprocess_for_train(self, image, output_height, output_width, resize_side_min,
                           resize_side_max):
    """Preprocesses the given image for training. Note that the actual resizing
    scale is sampled from.

      [`resize_size_min`, `resize_size_max`].

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing.

    Returns:
      A preprocessed image.
    """
    resize_side = tf.random_uniform(
        [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

    image = self._aspect_preserving_resize(image, resize_side)
    image = self._random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return self._mean_image_subtraction(image, [self._R_MEAN, self._G_MEAN, self._B_MEAN])

  def preprocess_for_eval(self, image, output_height, output_width, resize_side):
    """Preprocesses the given image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      resize_side: The smallest side of the image for aspect-preserving resizing.

    Returns:
      A preprocessed image.
    """
    image = self._aspect_preserving_resize(image, resize_side)
    image = self._central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return self._mean_image_subtraction(image, [self._R_MEAN, self._G_MEAN, self._B_MEAN])


class ImagenetPreprocessor(Preprocessor):

  def _crop(self, image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes. Note that
    the method doesn't assume we know the input image size but it does assume
    we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      the cropped (and resized) image.

    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
      cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
    with tf.control_dependencies([size_assertion]):
      image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)

  def _random_crop(self, image_list, crop_height, crop_width):
    """Crops the given list of images.
        The function applies the same crop to each image in the list. This can be
        effectively applied when there are multiple image inputs of the same
        dimension such as:
          image, depths, normals = _random_crop(
              [image, depths, normals], 120, 150)

        Args:
          image_list: a list of image tensors of the same dimension but possibly
            varying channel.
          crop_height: the new height.
          crop_width: the new width.

        Returns:
          the image_list with cropped images.

        Raises:
          ValueError: if there are multiple image inputs provided with different size
            or the images are smaller than the crop dimensions.
        """
    if not image_list:
      raise ValueError('Empty image_list.')

    rank_assertions = []
    for i in range(len(image_list)):
      image_rank = tf.rank(image_list[i])
      rank_assert = tf.Assert(
          tf.equal(image_rank, 3),
          ['Wrong rank for tensor  %s [expected] [actual]', image_list[i].name, 3, image_rank])
      rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
      image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height), tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
      image = image_list[i]
      asserts.append(rank_assertions[i])
      with tf.control_dependencies([rank_assertions[i]]):
        shape = tf.shape(image)
      height = shape[0]
      width = shape[1]

      height_assert = tf.Assert(
          tf.equal(height, image_height),
          ['Wrong height for tensor %s [expected][actual]', image.name, height, image_height])
      width_assert = tf.Assert(
          tf.equal(width, image_width),
          ['Wrong width for tensor %s [expected][actual]', image.name, width, image_width])
      asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
      max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
      max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    return [
        self._crop(image, offset_height, offset_width, crop_height, crop_width)
        for image in image_list
    ]

  def _smallest_size_at_least(self, height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(
        tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width

  def _aspect_preserving_resize(self, image, smallest_side):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

  def random_flip_left_right(self, image, im_shape=(512, 512, 3), seed=1234):
    """Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = tf.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
    return image

  def random_flip_up_down(self, image, im_shape=(512, 512, 3), seed=1234):
    """Randomly flip up/down the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = tf.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [0]), lambda: image)
    return image

  def preprocess_for_train(self,
                           image,
                           output_height,
                           output_width,
                           resize_side_min=256,
                           resize_side_max=512,
                           standardizer=None):
    resize_side = tf.random_uniform(
        [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

    image = self._aspect_preserving_resize(image, resize_side)
    image = self._random_crop([image], output_height, output_width)[0]
    image = self.random_flip_left_right(image)
    image = self.random_flip_up_down(image)
    image = tf.image.resize_images(image, [output_height, output_width], method=0)
    if standardizer is None:
      image = tf.image.per_image_standardization(image)
    else:
      image = standardizer(image, True)
    return image

  def preprocess_for_eval(self, image, output_height, output_width, standardizer=None):
    image = tf.image.resize_images(image, [output_height, output_width], method=0)
    if standardizer is None:
      image = tf.image.per_image_standardization(image)
    else:
      image = standardizer(image, False)
    return image


class InceptionPreprocessor(Preprocessor):

  def distort_color(self, image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
      if fast_mode:
        if color_ordering == 0:
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        else:
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        if color_ordering == 0:
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
          image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
          raise ValueError('color_ordering must be in [0, 3]')

      return tf.clip_by_value(image, 0.0, 1.0)

  def distorted_bounding_box_crop(self,
                                  image,
                                  bbox,
                                  min_object_covered=0.1,
                                  aspect_ratio_range=(0.75, 1.33),
                                  area_range=(0.05, 1.0),
                                  max_attempts=100,
                                  scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
        image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
      aspect_ratio_range: An optional list of `floats`. The cropped area of the
        image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
      scope: Optional scope for name_scope.
    Returns:
      A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
      sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bbox,
          min_object_covered=min_object_covered,
          aspect_ratio_range=aspect_ratio_range,
          area_range=area_range,
          max_attempts=max_attempts,
          use_image_if_no_bounding_boxes=True)
      bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

      cropped_image = tf.slice(image, bbox_begin, bbox_size)
      return cropped_image, distort_bbox

  def random_rotate(self, image, theta=80):
    angle = tf.random_uniform([], minval=-theta, maxval=theta, name='random_angle')
    image = tf.contrib.image.rotate(image, angle)
    return image

  def random_translate(self, image, shift=5):
    final_shift = tf.random_uniform([], minval=-shift, maxval=shift, name='random_translation')
    image = tf.contrib.image.transform(image, [1, 1, final_shift, 1, 1, final_shift, 0, 0])
    return image

  def preprocess_for_train(self,
                           image,
                           height,
                           width,
                           bbox=None,
                           fast_mode=True,
                           scope=None,
                           **kwargs):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Additionally it would create image_summaries to display the different
    transformations applied to the image.

    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations (i.e.
        bi-cubic resizing, random_hue or random_contrast).
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training with range [-1, 1].
    """
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
      if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
      if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
      distorted_image, distorted_bbox = self.distorted_bounding_box_crop(image, bbox)
      distorted_image.set_shape([None, None, 3])
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distorted_bbox)
      num_resize_cases = 1 if fast_mode else 4
      distorted_image = self.apply_with_random_selector(
          distorted_image,
          lambda x, method: tf.image.resize_images(x, [height, width], method=method),
          num_cases=num_resize_cases)

      distorted_image = tf.image.random_flip_left_right(distorted_image)
      distorted_image = tf.image.random_flip_up_down(distorted_image)
      distorted_image = self.random_rotate(distorted_image)
      # distorted_image = self.random_translate(distorted_image)
      distorted_image = self.apply_with_random_selector(
          distorted_image,
          lambda x, ordering: self.distort_color(x, ordering, fast_mode),
          num_cases=4)

      distorted_image = tf.subtract(distorted_image, 0.5)
      distorted_image = tf.multiply(distorted_image, 2.0)
      return distorted_image

  def preprocess_for_eval(self, image, height, width, central_fraction=0.875, scope=None, **kwargs):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would crop the central fraction of the
    input image.

    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
      height: integer
      width: integer
      central_fraction: Optional Float, fraction of the image to crop.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      if central_fraction:
        image = tf.image.central_crop(image, central_fraction=central_fraction)

      if height and width:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)
      return image


def normalize_image(image, original_minval, original_maxval, target_minval, target_maxval):
  """Normalizes pixel values in the image.

  Moves the pixel values from the current [original_minval, original_maxval]
  range to a the [target_minval, target_maxval] range.

  Args:
    image: rank 3 float32 tensor containing 1
           image -> [height, width, channels].
    original_minval: current image minimum value.
    original_maxval: current image maximum value.
    target_minval: target image minimum value.
    target_maxval: target image maximum value.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('NormalizeImage', values=[image]):
    original_minval = float(original_minval)
    original_maxval = float(original_maxval)
    target_minval = float(target_minval)
    target_maxval = float(target_maxval)
    image = tf.to_float(image)
    image = tf.subtract(image, original_minval)
    image = tf.multiply(image, (target_maxval - target_minval) / (original_maxval - original_minval))
    image = tf.add(image, target_minval)
    return image


def random_rgb_to_gray(image, probability=0.1, seed=None):
  """Changes the image from RGB to Grayscale with the given probability.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    probability: the probability of returning a grayscale image.
            The probability should be a number between [0, 1].
    seed: random seed.

  Returns:
    image: image which is the same shape as input image.
  """

  def _image_to_gray(image):
    image_gray1 = tf.image.rgb_to_grayscale(image)
    image_gray3 = tf.image.grayscale_to_rgb(image_gray1)
    return image_gray3

  with tf.name_scope('RandomRGBtoGray', values=[image]):
    # random variable defining whether to do flip or not
    do_gray_random = tf.random_uniform([], seed=seed)

    image = tf.cond(
        tf.greater(do_gray_random, probability), lambda: image, lambda: _image_to_gray(image))

  return image


def random_adjust_brightness(image, max_delta=0.2):
  """Randomly adjusts brightness.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: how much to change the brightness. A value between [0, 1).

  Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
  """
  with tf.name_scope('RandomAdjustBrightness', values=[image]):
    image = tf.image.random_brightness(image, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25):
  """Randomly adjusts contrast.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustContrast', values=[image]):
    image = tf.image.random_contrast(image, min_delta, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_hue(image, max_delta=0.02):
  """Randomly adjusts hue.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: change hue randomly with a value between 0 and max_delta.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustHue', values=[image]):
    image = tf.image.random_hue(image, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25):
  """Randomly adjusts saturation.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustSaturation', values=[image]):
    image = tf.image.random_saturation(image, min_delta, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_distort_color(image, color_ordering=0):
  """Randomly distorts color.

  Randomly distorts color using a combination of brightness, hue, contrast
  and saturation changes. Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0, 1).

  Returns:
    image: image which is the same shape as input image.

  Raises:
    ValueError: if color_ordering is not in {0, 1}.
  """
  with tf.name_scope('RandomDistortColor', values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
    else:
      raise ValueError('color_ordering must be in {0, 1}')

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def resize_to_range(image, masks=None, min_dimension=None, max_dimension=None, align_corners=False):
  """Resizes an image so its dimensions are within the provided value.
    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum dimension is equal to the
       provided value without the other dimension exceeding max_dimension,
       then do so.
    2. Otherwise, resize so the largest dimension is equal to max_dimension.
    Args:
      image: A 3D tensor of shape [height, width, channels]
      masks: (optional) rank 3 float32 tensor with shape
             [num_instances, height, width] containing instance masks.
      min_dimension: (optional) (scalar) desired size of the smaller image
                     dimension.
      max_dimension: (optional) (scalar) maximum allowed size
                     of the larger image dimension.
      align_corners: bool. If true, exactly align all 4 corners of the input
                     and output. Defaults to False.
    Returns:
      A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension or
      max(new_height, new_width) == max_dimension.
      If masks is not None, also outputs masks:
      A 3D tensor of shape [num_instances, new_height, new_width]
    Raises:
      ValueError: if the image is not a 3D tensor.
    """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
    image_shape = tf.shape(image)
    orig_height = tf.to_float(image_shape[0])
    orig_width = tf.to_float(image_shape[1])
    orig_min_dim = tf.minimum(orig_height, orig_width)

    min_dimension = tf.constant(min_dimension, dtype=tf.float32)
    large_scale_factor = min_dimension / orig_min_dim
    large_height = tf.to_int32(tf.round(orig_height * large_scale_factor))
    large_width = tf.to_int32(tf.round(orig_width * large_scale_factor))
    large_size = tf.stack([large_height, large_width])

    if max_dimension:
      orig_max_dim = tf.maximum(orig_height, orig_width)
      max_dimension = tf.constant(max_dimension, dtype=tf.float32)
      small_scale_factor = max_dimension / orig_max_dim
      small_height = tf.to_int32(tf.round(orig_height * small_scale_factor))
      small_width = tf.to_int32(tf.round(orig_width * small_scale_factor))
      small_size = tf.stack([small_height, small_width])

      new_size = tf.cond(
          tf.to_float(tf.reduce_max(large_size)) > max_dimension, lambda: small_size,
          lambda: large_size)
    else:
      new_size = large_size

    new_image = tf.image.resize_images(image, new_size, align_corners=align_corners)

    result = new_image
    if masks is not None:
      num_instances = tf.shape(masks)[0]

      def resize_masks_branch():
        new_masks = tf.expand_dims(masks, 3)
        new_masks = tf.image.resize_nearest_neighbor(
            new_masks, new_size, align_corners=align_corners)
        new_masks = tf.squeeze(new_masks, axis=3)
        return new_masks

      def reshape_masks_branch():
        new_masks = tf.reshape(masks, [0, new_size[0], new_size[1]])
        return new_masks

      masks = tf.cond(num_instances > 0, resize_masks_branch, reshape_masks_branch)
      result = [new_image, masks]

    return result


def random_black_patches(image,
                         max_black_patches=10,
                         probability=0.5,
                         size_to_image_ratio=0.1,
                         random_seed=None):
  """Randomly adds some black patches to the image.

  This op adds up to max_black_patches square black patches of a fixed size
  to the image where size is specified via the size_to_image_ratio parameter.

  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_black_patches: number of times that the function tries to add a
                       black box to the image.
    probability: at each try, what is the chance of adding a box.
    size_to_image_ratio: Determines the ratio of the size of the black patches
                         to the size of the image.
                         box_size = size_to_image_ratio *
                                    min(image_width, image_height)
    random_seed: random seed.

  Returns:
    image
  """

  def add_black_patch_to_image(image):
    """Function for adding one patch to the image.

    Args:
      image: image

    Returns:
      image with a randomly added black box
    """
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    box_size = tf.to_int32(
        tf.multiply(
            tf.minimum(tf.to_float(image_height), tf.to_float(image_width)), size_to_image_ratio))
    normalized_y_min = tf.random_uniform(
        [], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)
    normalized_x_min = tf.random_uniform(
        [], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)
    y_min = tf.to_int32(normalized_y_min * tf.to_float(image_height))
    x_min = tf.to_int32(normalized_x_min * tf.to_float(image_width))
    black_box = tf.ones([box_size, box_size, 3], dtype=tf.float32)
    mask = 1.0 - tf.image.pad_to_bounding_box(black_box, y_min, x_min, image_height, image_width)
    image = tf.multiply(image, mask)
    return image

  with tf.name_scope('RandomBlackPatchInImage', values=[image]):
    for _ in range(max_black_patches):
      random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=random_seed)
      image = tf.cond(
          tf.greater(random_prob, probability), lambda: image,
          lambda: add_black_patch_to_image(image))

    return image


class SegPreprocessorv2(Preprocessor):

  def __init__(self, prob_of_flip=0.5):
    self._prob_of_flip = prob_of_flip
    super(SegPreprocessorv2, self).__init__()

  def _flip_dim(self, tensor_list, prob=0.5, dim=1):
    """Randomly flips a dimension of the given tensor. The decision to randomly
    flip the `Tensors` is made together. In other words, all or none of the
    images pass in are flipped. Note that tf.random_flip_left_right and
    tf.random_flip_up_down isn't used so that we can control for the
    probability as well as ensure the same decision is applied across the
    images.

    Args:
      tensor_list: A list of `Tensors` with the same number of dimensions.
      prob: The probability of a left-right flip.
      dim: The dimension to flip, 0, 1, ..

    Returns:
      outputs: A list of the possibly flipped `Tensors` as well as an indicator
      `Tensor` at the end whose value is `True` if the inputs were flipped and
      `False` otherwise.

    Raises:
      ValueError: If dim is negative or greater than the dimension of a `Tensor`.
    """
    random_value = tf.random_uniform([])

    def flip():
      flipped = []
      for tensor in tensor_list:
        if dim < 0 or dim >= len(tensor.get_shape().as_list()):
          raise ValueError('dim must represent a valid dimension.')
        flipped.append(tf.reverse_v2(tensor, [dim]))
      return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
      outputs = [outputs]
    outputs.append(is_flipped)

    return outputs

  def _pad_to_bounding_box(self, image, offset_height, offset_width, target_height, target_width,
                           pad_value):
    """Pads the given image with the given pad_value. Works like
    tf.image.pad_to_bounding_box, except it can pad the image with any given
    arbitrary pad value and also handle images whose sizes are not known during
    graph construction.

    Args:
      image: 3-D tensor with shape [height, width, channels]
      offset_height: Number of rows of zeros to add on top.
      offset_width: Number of columns of zeros to add on the left.
      target_height: Height of output image.
      target_width: Width of output image.
      pad_value: Value to pad the image tensor with.

    Returns:
      3-D tensor of shape [target_height, target_width, channels].

    Raises:
      ValueError: If the shape of image is incompatible with the offset_* or
      target_* arguments.
    """
    image_rank = tf.rank(image)
    image_rank_assert = tf.Assert(
        tf.equal(image_rank, 3), ['Wrong image tensor rank [Expected] [Actual]', 3, image_rank])
    with tf.control_dependencies([image_rank_assert]):
      image -= pad_value
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]
    target_width_assert = tf.Assert(
        tf.greater_equal(target_width, width), ['target_width must be >= width'])
    target_height_assert = tf.Assert(
        tf.greater_equal(target_height, height), ['target_height must be >= height'])
    with tf.control_dependencies([target_width_assert]):
      after_padding_width = target_width - offset_width - width
    with tf.control_dependencies([target_height_assert]):
      after_padding_height = target_height - offset_height - height
    offset_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(after_padding_width, 0), tf.greater_equal(after_padding_height, 0)),
        ['target size not possible with the given target offsets'])

    height_params = tf.stack([offset_height, after_padding_height])
    width_params = tf.stack([offset_width, after_padding_width])
    channel_params = tf.stack([0, 0])
    with tf.control_dependencies([offset_assert]):
      paddings = tf.stack([height_params, width_params, channel_params])
    padded = tf.pad(image, paddings)
    return padded + pad_value

  def _crop(self, image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes. Note that
    the method doesn't assume we know the input image size but it does assume
    we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      The cropped (and resized) image.

    Raises:
      ValueError: if `image` doesn't have rank of 3.
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
      raise ValueError('input must have rank of 3')
    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
      cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
      image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image

  def _random_crop(self, image_list, crop_height, crop_width):
    """Crops the given list of images.
        The function applies the same crop to each image in the list. This can be
        effectively applied when there are multiple image inputs of the same
        dimension such as:
          image, depths, normals = random_crop([image, depths, normals], 120, 150)

        Args:
          image_list: a list of image tensors of the same dimension but possibly
            varying channel.
          crop_height: the new height.
          crop_width: the new width.

        Returns:
          the image_list with cropped images.

        Raises:
          ValueError: if there are multiple image inputs provided with different size
            or the images are smaller than the crop dimensions.
        """
    if not image_list:
      raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
      image_rank = tf.rank(image_list[i])
      rank_assert = tf.Assert(
          tf.equal(image_rank, 3),
          ['Wrong rank for tensor  %s [expected] [actual]', image_list[i].name, 3, image_rank])
      rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
      image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height), tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
      image = image_list[i]
      asserts.append(rank_assertions[i])
      with tf.control_dependencies([rank_assertions[i]]):
        shape = tf.shape(image)
      height = shape[0]
      width = shape[1]

      height_assert = tf.Assert(
          tf.equal(height, image_height),
          ['Wrong height for tensor %s [expected][actual]', image.name, height, image_height])
      width_assert = tf.Assert(
          tf.equal(width, image_width),
          ['Wrong width for tensor %s [expected][actual]', image.name, width, image_width])
      asserts.extend([height_assert, width_assert])

    with tf.control_dependencies(asserts):
      max_offset_height = tf.reshape(image_height - crop_height + 1, [])
      max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    return [
        self._crop(image, offset_height, offset_width, crop_height, crop_width)
        for image in image_list
    ]

  def _get_random_scale(self, min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.

    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.

    Returns:
      A random scale value selected between minimum and maximum value.

    Raises:
      ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
      raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
      return tf.to_float(min_scale_factor)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
      return tf.random_uniform([1], minval=min_scale_factor, maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]

  def _randomly_scale_image_and_label(self, image, label=None, scale=1.0):
    """Randomly scales image and label.

    Args:
      image: Image with shape [height, width, 3].
      label: Label with shape [height, width, 1].
      scale: The value to scale image and label.

    Returns:
      Scaled image and label.
    """
    # No random scaling if scale == 1.
    if scale == 1.0:
      return image, label
    image_shape = tf.shape(image)
    new_dim = tf.to_int32(tf.to_float([image_shape[0], image_shape[1]]) * scale)

    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = tf.squeeze(
        tf.image.resize_bilinear(tf.expand_dims(image, 0), new_dim, align_corners=True), [0])
    if label is not None:
      label = tf.squeeze(
          tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_dim, align_corners=True),
          [0])

    return image, label

  def _resolve_shape(self, tensor, rank=None, scope=None):
    """Fully resolves the shape of a Tensor. Use as much as possible the shape
    components already known during graph creation and resolve the remaining
    ones during runtime.

    Args:
      tensor: Input tensor whose shape we query.
      rank: The rank of the tensor, provided that we know it.
      scope: Optional name scope.

    Returns:
      shape: The full shape of the tensor.
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
      if rank is not None:
        shape = tensor.get_shape().with_rank(rank).as_list()
      else:
        shape = tensor.get_shape().as_list()

      if None in shape:
        shape_dynamic = tf.shape(tensor)
        for i in range(len(shape)):
          if shape[i] is None:
            shape[i] = shape_dynamic[i]

      return shape

  def _resize_to_range(self,
                       image,
                       label=None,
                       min_size=None,
                       max_size=None,
                       factor=None,
                       align_corners=True,
                       label_layout_is_chw=False,
                       scope=None,
                       method=tf.image.ResizeMethod.BILINEAR):
    """Resizes image or label so their sides are within the provided range.
        The output size can be described by two cases:
        1. If the image can be rescaled so its minimum size is equal to min_size
           without the other side exceeding max_size, then do so.
        2. Otherwise, resize so the largest side is equal to max_size.
        An integer in `range(factor)` is added to the computed sides so that the
        final dimensions are multiples of `factor` plus one.

        Args:
          image: A 3D tensor of shape [height, width, channels].
          label: (optional) A 3D tensor of shape [height, width, channels] (default)
            or [channels, height, width] when label_layout_is_chw = True.
          min_size: (scalar) desired size of the smaller image side.
          max_size: (scalar) maximum allowed size of the larger image side. Note
            that the output dimension is no larger than max_size and may be slightly
            smaller than min_size when factor is not None.
          factor: Make output size multiple of factor plus one.
          align_corners: If True, exactly align all 4 corners of input and output.
          label_layout_is_chw: If true, the label has shape [channel, height, width].
            We support this case because for some instance segmentation dataset, the
            instance segmentation is saved as [num_instances, height, width].
          scope: Optional name scope.
          method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.

        Returns:
          A 3-D tensor of shape [new_height, new_width, channels], where the image
          has been resized (with the specified method) so that
          min(new_height, new_width) == ceil(min_size) or
          max(new_height, new_width) == ceil(max_size).

        Raises:
          ValueError: If the image is not a 3D tensor.
        """
    with tf.name_scope(scope, 'resize_to_range', [image]):
      new_tensor_list = []
      min_size = tf.to_float(min_size)
      if max_size is not None:
        max_size = tf.to_float(max_size)
        # Modify the max_size to be a multiple of factor plus 1 and make sure the
        # max dimension after resizing is no larger than max_size.
        if factor is not None:
          max_size = (max_size + (factor - (max_size - 1) % factor) % factor - factor)

      [orig_height, orig_width, _] = self._resolve_shape(image, rank=3)
      orig_height = tf.to_float(orig_height)
      orig_width = tf.to_float(orig_width)
      orig_min_size = tf.minimum(orig_height, orig_width)

      # Calculate the larger of the possible sizes
      large_scale_factor = min_size / orig_min_size
      large_height = tf.to_int32(tf.ceil(orig_height * large_scale_factor))
      large_width = tf.to_int32(tf.ceil(orig_width * large_scale_factor))
      large_size = tf.stack([large_height, large_width])

      new_size = large_size
      if max_size is not None:
        # Calculate the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_size = tf.maximum(orig_height, orig_width)
        small_scale_factor = max_size / orig_max_size
        small_height = tf.to_int32(tf.ceil(orig_height * small_scale_factor))
        small_width = tf.to_int32(tf.ceil(orig_width * small_scale_factor))
        small_size = tf.stack([small_height, small_width])
        new_size = tf.cond(
            tf.to_float(tf.reduce_max(large_size)) > max_size, lambda: small_size,
            lambda: large_size)
      # Ensure that both output sides are multiples of factor plus one.
      if factor is not None:
        new_size += (factor - (new_size - 1) % factor) % factor
      new_tensor_list.append(
          tf.image.resize_images(image, new_size, method=method, align_corners=align_corners))
      if label is not None:
        if label_layout_is_chw:
          # Input label has shape [channel, height, width].
          resized_label = tf.expand_dims(label, 3)
          resized_label = tf.image.resize_nearest_neighbor(
              resized_label, new_size, align_corners=align_corners)
          resized_label = tf.squeeze(resized_label, 3)
        else:
          # Input label has shape [height, width, channel].
          resized_label = tf.image.resize_images(
              label,
              new_size,
              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
              align_corners=align_corners)
        new_tensor_list.append(resized_label)
      else:
        new_tensor_list.append(None)
      return new_tensor_list

  def _preprocess_image_label(self,
                              image,
                              label,
                              crop_height,
                              crop_width,
                              min_resize_value=None,
                              max_resize_value=None,
                              resize_factor=None,
                              min_scale_factor=1.,
                              max_scale_factor=1.,
                              scale_factor_step_size=0,
                              ignore_label=255,
                              is_training=True,
                              model_variant=None):
    """Preprocesses the image and label.

    Args:
      image: Input image.
      label: Ground truth annotation label.
      crop_height: The height value used to crop the image and label.
      crop_width: The width value used to crop the image and label.
      min_resize_value: Desired size of the smaller image side.
      max_resize_value: Maximum allowed size of the larger image side.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      ignore_label: The label value which will be ignored for training and
        evaluation.
      is_training: If the preprocessing is used for training or not.
      model_variant: Model variant (string) for choosing how to mean-subtract the
        images. See feature_extractor.network_map for supported model variants.
    Returns:
      original_image: Original image (could be resized).
      processed_image: Preprocessed image.
      label: Preprocessed ground truth segmentation label.
    Raises:
      ValueError: Ground truth label not provided during training.
    """
    if is_training and label is None:
      raise ValueError('During training, label must be provided.')
    if model_variant is None:
      tf.logging.warning('Default mean-subtraction is performed. Please specify '
                         'a model_variant. See feature_extractor.network_map for '
                         'supported model variants.')

    # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)

    if label is not None:
      label = tf.cast(label, tf.int32)

    # Resize image and label to the desired range.
    if min_resize_value is not None or max_resize_value is not None:
      [processed_image, label] = (self._resize_to_range(
          image=processed_image,
          label=label,
          min_size=min_resize_value,
          max_size=max_resize_value,
          factor=resize_factor,
          align_corners=True))
      # The `original_image` becomes the resized image.
      original_image = tf.identity(processed_image)

    # Data augmentation by randomly scaling the inputs.
    scale = self._get_random_scale(min_scale_factor, max_scale_factor, scale_factor_step_size)
    processed_image, label = self._randomly_scale_image_and_label(processed_image, label, scale)
    processed_image.set_shape([None, None, 3])

    # Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + \
        tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape(mean_pixel_fn(model_variant), [1, 1, 3])
    processed_image = self._pad_to_bounding_box(processed_image, 0, 0, target_height, target_width,
                                                mean_pixel)

    if label is not None:
      label = self._pad_to_bounding_box(label, 0, 0, target_height, target_width, ignore_label)

    # Randomly crop the image and label.
    if is_training and label is not None:
      processed_image, label = self._random_crop([processed_image, label], crop_height, crop_width)

    processed_image.set_shape([crop_height, crop_width, 3])

    if label is not None:
      label.set_shape([crop_height, crop_width, 1])

    if is_training:
      # Randomly left-right flip the image and label.
      processed_image, label, _ = self._flip_dim([processed_image, label], self._prob_of_flip, dim=1)

    return original_image, processed_image, label

  def preprocess_for_train(self,
                           image,
                           label,
                           crop_height,
                           crop_width,
                           min_resize_value=None,
                           max_resize_value=None,
                           resize_factor=None,
                           min_scale_factor=1.,
                           max_scale_factor=1.,
                           scale_factor_step_size=0,
                           ignore_label=255,
                           model_variant=None):
    return self.preprocess_image_label(
        image,
        label,
        crop_height,
        crop_width,
        min_resize_value=min_resize_value,
        max_resize_value=max_resize_value,
        resize_factor=resize_factor,
        min_scale_factor=min_scale_factor,
        max_scale_factor=max_scale_factor,
        scale_factor_step_size=scale_factor_step_size,
        ignore_label=ignore_label,
        is_training=True,
        model_variant=model_variant)

  def preprocess_for_eval(self,
                          image,
                          label,
                          crop_height,
                          crop_width,
                          min_resize_value=None,
                          max_resize_value=None,
                          resize_factor=None,
                          min_scale_factor=1.,
                          max_scale_factor=1.,
                          scale_factor_step_size=0,
                          ignore_label=255,
                          model_variant=None):
    return self.preprocess_image_label(
        image,
        label,
        crop_height,
        crop_width,
        min_resize_value=min_resize_value,
        max_resize_value=max_resize_value,
        resize_factor=resize_factor,
        min_scale_factor=min_scale_factor,
        max_scale_factor=max_scale_factor,
        scale_factor_step_size=scale_factor_step_size,
        ignore_label=ignore_label,
        is_training=False,
        model_variant=model_variant)


class RexPreprocessor(Preprocessor):

  def __init__(self, width, height, standardizer):
    self.width = width
    self.height = height
    self.standardizer = standardizer

  def process(self, frame):
    roi_height, roi_width = frame.shape[0], int(frame.shape[1] * 0.68)
    processed = np.zeros((roi_height, roi_width))

    roi = frame[:, :roi_width, 0]
    all_obstacles_idx = roi > 50
    processed[all_obstacles_idx] = 1
    unharmful_obstacles_idx = roi > 200
    processed[unharmful_obstacles_idx] = 0

    processed = np.asarray(imresize(processed, (self.height, self.width, 1)), dtype=np.float)
    return processed

  def preprocess_for_train(self, inputs):
    processed = self.process(inputs)
    return self.standardizer(processed, True)

  def preprocess_for_eval(self, inputs):
    processed = self.process(inputs)
    return self.standardizer(processed, False)

  def get_initial_state(self, first_frame):
    self.state = np.array([first_frame, first_frame, first_frame, first_frame])
    return self.state

  def get_updated_state(self, next_frame):
    self.state = np.concatenate((self.state[-3:], np.expand_dims(next_frame, axis=0)), axis=0)
    return self.state
