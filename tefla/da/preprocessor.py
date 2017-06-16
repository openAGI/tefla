import tensorflow as tf


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
