import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Preprocessor(object):

    def __init__(self):
        super(Preprocessor, self).__init__()

    @abc.abstractmethod
    def preprocess_for_train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess_for_eval(self, *args, **kwargs):
        raise NotImplementedError

    def preprocess_image(self, image, output_height, output_width, is_training,
                         resize_side_min,
                         resize_side_max):
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
            return preprocess_for_train(image, output_height, output_width,
                                        resize_side_min, resize_side_max)
        else:
            return preprocess_for_eval(image, output_height, output_width,
                                       resize_side_min)


class VggPreprocessor(Preprocessor):

    def __init__(self):
        self._R_MEAN = 123.68
        self._G_MEAN = 116.78
        self._B_MEAN = 103.94
        self._RESIZE_SIDE_MIN = 256
        self._RESIZE_SIDE_MAX = 512

    def _crop(self, image, offset_height, offset_width, crop_height, crop_width):
        """Crops the given image using the provided offsets and sizes.
        Note that the method doesn't assume we know the input image size but it does
        assume we know the input image rank.

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

        rank_assertion = tf.Assert(
            tf.equal(tf.rank(image), 3),
            ['Rank of image must be equal to 3.'])
        with tf.control_dependencies([rank_assertion]):
            cropped_shape = tf.stack(
                [crop_height, crop_width, original_shape[2]])

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
          image, depths, normals = _random_crop([image, depths, normals], 120, 150)

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
                ['Wrong rank for tensor  %s [expected] [actual]',
                 image_list[i].name, 3, image_rank])
            rank_assertions.append(rank_assert)

        with tf.control_dependencies([rank_assertions[0]]):
            image_shape = tf.shape(image_list[0])
        image_height = image_shape[0]
        image_width = image_shape[1]
        crop_size_assert = tf.Assert(
            tf.logical_and(
                tf.greater_equal(image_height, crop_height),
                tf.greater_equal(image_width, crop_width)),
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
                ['Wrong height for tensor %s [expected][actual]',
                 image.name, height, image_height])
            width_assert = tf.Assert(
                tf.equal(width, image_width),
                ['Wrong width for tensor %s [expected][actual]',
                 image.name, width, image_width])
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
        offset_height = tf.random_uniform(
            [], maxval=max_offset_height, dtype=tf.int32)
        offset_width = tf.random_uniform(
            [], maxval=max_offset_width, dtype=tf.int32)

        return [self._crop(image, offset_height, offset_width,
                           crop_height, crop_width) for image in image_list]

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

            outputs.append(self._crop(image, offset_height, offset_width,
                                      crop_height, crop_width))
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

        channels = tf.split(
            axis=2, num_or_size_splits=num_channels, value=image)
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

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
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
        new_height, new_width = self._smallest_size_at_least(
            height, width, smallest_side)
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                                 align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return resized_image

    def preprocess_for_train(self, image,
                             output_height,
                             output_width,
                             resize_side_min,
                             resize_side_max):
        """Preprocesses the given image for training.
        Note that the actual resizing scale is sampled from
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


def normalize_image(image, original_minval, original_maxval, target_minval,
                    target_maxval):
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
        image = tf.multiply(image, (target_maxval - target_minval) /
                            (original_maxval - original_minval))
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
            tf.greater(do_gray_random, probability), lambda: image,
            lambda: _image_to_gray(image))

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

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    align_corners=False):
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
            small_height = tf.to_int32(
                tf.round(orig_height * small_scale_factor))
            small_width = tf.to_int32(
                tf.round(orig_width * small_scale_factor))
            small_size = tf.stack([small_height, small_width])

            new_size = tf.cond(
                tf.to_float(tf.reduce_max(large_size)) > max_dimension,
                lambda: small_size, lambda: large_size)
        else:
            new_size = large_size

        new_image = tf.image.resize_images(image, new_size,
                                           align_corners=align_corners)

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

            masks = tf.cond(num_instances > 0,
                            resize_masks_branch,
                            reshape_masks_branch)
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
                tf.minimum(tf.to_float(image_height),
                           tf.to_float(image_width)),
                size_to_image_ratio))
        normalized_y_min = tf.random_uniform(
            [], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)
        normalized_x_min = tf.random_uniform(
            [], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)
        y_min = tf.to_int32(normalized_y_min * tf.to_float(image_height))
        x_min = tf.to_int32(normalized_x_min * tf.to_float(image_width))
        black_box = tf.ones([box_size, box_size, 3], dtype=tf.float32)
        mask = 1.0 - tf.image.pad_to_bounding_box(black_box, y_min, x_min,
                                                  image_height, image_width)
        image = tf.multiply(image, mask)
        return image

    with tf.name_scope('RandomBlackPatchInImage', values=[image]):
        for _ in range(max_black_patches):
            random_prob = tf.random_uniform([], minval=0.0, maxval=1.0,
                                            dtype=tf.float32, seed=random_seed)
            image = tf.cond(
                tf.greater(random_prob, probability), lambda: image,
                lambda: add_black_patch_to_image(image))

        return image
