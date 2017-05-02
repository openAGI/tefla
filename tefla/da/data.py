# Original code from: https://github.com/sveitser/kaggle_diabetic
# Original MIT license:
# https://github.com/sveitser/kaggle_diabetic/blob/master/LICENSE
"""data augmentation.

The code for data augmentation originally comes from
https://github.com/benanne/kaggle-ndsb/blob/master/data.py

Enhanced by Mrinal Haloi
"""
from __future__ import division, print_function

from PIL import Image
from PIL import ImageEnhance

import skimage
import skimage.transform
from skimage.transform._warps_cy import _warp_fast

from .standardizer import *
from ..core.data_load_ops import *

no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}


def fast_warp(img, tf, output_shape, mode='constant', mode_cval=0, order=0):
    """Warp an image according to a given coordinate transformation.

        This wrapper function is faster than skimage.transform.warp
    Args:
        img: `ndarray`, input image
        tf: For 2-D images, you can directly pass a transformation object
            e.g. skimage.transform.SimilarityTransform, or its inverse.
        output_shape: tuple, (rows, cols)
        mode: mode for transformation
            available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
        mode_cval: float, Used in conjunction with mode `constant`, the value outside the image boundaries
        order: int, The order of interpolation. The order has to be in the range 0-5:
            0: Nearest-neighbor
            1: Bi-linear (default)
            2: Bi-quadratic
            3: Bi-cubic
            4: Bi-quartic
            5: Bi-quintic

    Returns:
        warped, double `ndarray`
    """
    m = tf.params
    t_img = np.zeros((img.shape[0],) + output_shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, output_shape=output_shape,
                              mode=mode, cval=mode_cval, order=order)
    return t_img


def contrast_transform(img, contrast_min=0.8, contrast_max=1.2):
    """Transform input image contrast

    Transform the input image contrast by a factor returned by a unifrom
    distribution with `contarst_min` and `contarst_max` as params

    Args:
        img: `ndarray`, input image
        contrast_min: float, minimum contrast for transformation
        contrast_max: float, maximum contrast for transformation

    Returns:
        `ndarray`, contrast enhanced image
    """
    if isinstance(img, (np.ndarray)):
        img = Image.fromarray(img)
    contrast_param = np.random.uniform(contrast_min, contrast_max)
    t_img = ImageEnhance.Contrast(img).enhance(contrast_param)

    return np.array(t_img)


def brightness_transform(img, brightness_min=0.93, brightness_max=1.4):
    """Transform input image brightness

    Transform the input image brightness by a factor returned by a unifrom
    distribution with `brightness_min` and `brightness_max` as params

    Args:
        img: `ndarray`, input image
        brightness_min: float, minimum contrast for transformation
        brightness_max: float, maximum contrast for transformation

    Returns:
        `ndarray`, brightness transformed image
    """
    if isinstance(img, (np.ndarray)):
        img = Image.fromarray(img)
    brightness_param = np.random.uniform(brightness_min, brightness_max)
    t_img = ImageEnhance.Brightness(img).enhance(brightness_param)

    return np.array(t_img)


def build_rescale_transform_slow(downscale_factor, image_shape, target_shape):
    """Rescale Transform

    This mimics the skimage.transform.resize function.
    The resulting image is centered.

    Args:
        downscale_factor: float, >1
        image_shape: tuple(rows, cols), input image shape
        target_shape: tuple(rows, cols), output image shape

    Returns:
        rescaled centered image transform instance
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    col_scale = row_scale = downscale_factor
    src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
    dst_corners = np.zeros(src_corners.shape, dtype=np.double)
    # take into account that 0th pixel is at position (0.5, 0.5)
    dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
    dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

    tform_ds = skimage.transform.AffineTransform()
    tform_ds.estimate(src_corners, dst_corners)

    # centering
    shift_x = cols / (2.0 * downscale_factor) - tcols / 2.0
    shift_y = rows / (2.0 * downscale_factor) - trows / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(
        translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds


def build_rescale_transform_fast(downscale_factor, image_shape, target_shape):
    """Rescale Transform

    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't
    100% correct, but it shouldn't matter much in practice.
    The resulting image is centered.

    Args:
        downscale_factor: float, >1
        image_shape: tuple(rows, cols), input image shape
        target_shape: tuple(rows, cols), output image shape

    Returns:
        rescaled and centering transform instance
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    tform_ds = skimage.transform.AffineTransform(
        scale=(downscale_factor, downscale_factor))
    # centering
    shift_x = cols / (2.0 * downscale_factor) - tcols / 2.0
    shift_y = rows / (2.0 * downscale_factor) - trows / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(
        translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds


def build_centering_transform(image_shape, target_shape):
    """Image cetering transform

    Args:
        image_shape: tuple(rows, cols), input image shape
        target_shape: tuple(rows, cols), output image shape

    Returns:
        a centering transform instance

    """
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def build_center_uncenter_transforms(image_shape):
    """Center Unceter transform

    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.

    Args:
        image_shape: tuple(rows, cols), input image shape

    Returns:
        a center and an uncenter transform instance
    """
    center_shift = np.array(
        [image_shape[1], image_shape[0]]) / 2.0 - 0.5  # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(
        translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(
        translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False):
    """Augmentation transform

    It performs zooming, rotation, shear, translation and flip operation
    Affine Transformation on the input image

    Args:
        zoom: a tuple(zoom_rows, zoom_cols)
        rotation: float, Rotation angle in counter-clockwise direction as radians.
        shear: float, shear angle in counter-clockwise direction as radians
        translation: tuple(trans_rows, trans_cols)
        flip: bool, flip an image

    Returns:
        augment tranform instance
    """
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1 / zoom[0], 1 / zoom[1]), rotation=np.deg2rad(rotation),
                                                      shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True,
                                  allow_stretch=False, rng=np.random):
    """Random perturbation

    It perturbs the image randomly

    Args:
        zoom_range: a tuple(min_zoom, max_zoom)
            e.g.: (1/1.15, 1.15)
        rotation_range: a tuple(min_angle, max_angle)
            e.g.: (0. 360)
        shear_range: a tuple(min_shear, max_shear)
            e.g.: (0, 15)
        translation_range: a tuple(min_shift, max_shift)
            e.g.: (-15, 15)
        do_flip: bool, flip an image
        allow_stretch: bool, stretch an image
        rng: an instance

    Returns:
        augment transform instance
    """
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0)  # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True:  # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead
    # of [0.9, 1.1] makes more sense.

    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)


def definite_crop(img, bbox):
    """crop an image

    Args:
        img: `ndarray`, input image
        bbox: list, with crop co-ordinates and width and height
            e.g.: [x, y, width, height]

    Returns:
        returns cropped image
    """
    img = img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return img


def perturb(img, augmentation_params, target_shape, rng=np.random, mode='constant', mode_cval=0):
    """Perturb image

    It perturbs an image with augmentation transform

    Args:
        img: a `ndarray`, input image
        augmentation_paras: a dict, with augmentation name as keys and values as params
        target_shape: a tuple(rows, cols), output image shape
        rng: an instance for random number generation
        mode: mode for transformation
            available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
        mode_cval: float, Used in conjunction with mode `constant`,
            the value outside the image boundaries

    Returns:
        a `ndarray` of transformed image
    """
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = random_perturbation_transform(
        rng=rng, **augmentation_params)
    # shift to center, augment, shift back (for the rotation/shearing)
    tform_augment = tform_uncenter + tform_augment + tform_center
    return fast_warp(img, tform_centering + tform_augment,
                     output_shape=target_shape,
                     mode=mode, mode_cval=mode_cval)


def perturb_rescaled(img, scale, augmentation_params, target_shape=(224, 224), rng=np.random, mode='constant', mode_cval=0):
    """Perturb image rescaled

    It perturbs an image with augmentation transform

    Args:
        img: a `ndarray`, input image
        scale: float, >1, downscaling factor.
        augmentation_paras: a dict, with augmentation name as keys and values as params
        target_shape: a tuple(rows, cols), output image shape
        rng: an instance for random number generation
        mode: mode for transformation
            available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
        mode_cval: float, Used in conjunction with mode `constant`,
            the value outside the image boundaries

    Returns:
        a `ndarray` of transformed image
    """
    tform_rescale = build_rescale_transform(
        scale, img.shape, target_shape)  # also does centering
    tform_center, tform_uncenter = build_center_uncenter_transforms(img.shape)
    tform_augment = random_perturbation_transform(
        rng=rng, **augmentation_params)
    # shift to center, augment, shift back (for the rotation/shearing)
    tform_augment = tform_uncenter + tform_augment + tform_center
    return fast_warp(img, tform_rescale + tform_augment, output_shape=target_shape, mode=mode, mode_cval=mode_cval).astype('float32')


# for test-time augmentation
def perturb_fixed(img, tform_augment, target_shape=(50, 50), mode='constant', mode_cval=0):
    """Perturb image Determinastic

    It perturbs an image with augmentation transform with determinastic params
    used for validation/testing data

    Args:
        img: a `ndarray`, input image
        augmentation_paras: a dict, with augmentation name as keys and values as params
        target_shape: a tuple(rows, cols), output image shape
        mode: mode for transformation
            available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
        mode_cval: float, Used in conjunction with mode `constant`,
            the value outside the image boundaries

    Returns:
        a `ndarray` of transformed image
    """
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    # shift to center, augment, shift back (for the rotation/shearing)
    tform_augment = tform_uncenter + tform_augment + tform_center
    return fast_warp(img, tform_centering + tform_augment,
                     output_shape=target_shape, mode=mode, mode_cval=mode_cval)


def load_perturbed(fname):
    img = load_image(fname).astype(np.float32)
    return perturb(img)


def load_augmented_images(fnames, preprocessor, w, h, is_training, aug_params=no_augmentation_params, transform=None,
                          bbox=None, fill_mode='constant', fill_mode_cval=0, standardizer=None, save_to_dir=None):
    return np.array(
        [load_augment(f, preprocessor, w, h, is_training, aug_params, transform, bbox, fill_mode, fill_mode_cval,
                      standardizer, save_to_dir) for f in fnames])


def load_augment(fname, preprocessor, w, h, is_training, aug_params=no_augmentation_params, transform=None, bbox=None,
                 fill_mode='constant', fill_mode_cval=0, standardizer=None, save_to_dir=None):
    """Load augmented image with output shape (w, h).

    Default arguments return non augmented image of shape (w, h).
    To apply a fixed transform (color augmentation) specify transform
    (color_vec).
    To generate a random augmentation specify aug_params and sigma.

    Args:
        fname: string, image filename
        preprocessor: real-time image processing/crop
        w: int, width of target image
        h: int, height of target image
        is_training: bool, if True then training else validation
        aug_params: a dict, augmentation params
        transform: transform instance
        bbox: object bounding box
        fll_mode: mode for transformation
            available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
        fill_mode_cval: float, Used in conjunction with mode `constant`,
            the value outside the image boundaries
        standardizer: image standardizer, zero mean, unit variance image
             e.g.: samplewise standardized each image based on its own value
        save_to_dir: a string, path to save image, save output image to a dir

    Returns:
        augmented image
    """
    img = load_image(fname, preprocessor)

    # target shape should be (h, w) i.e. (rows, cols). need to revisit when we
    # do non-square shapes

    if bbox is not None:
        img = definite_crop(img, bbox)
        # print(img.shape)
        # import cv2
        # cv2.imshow("test", np.asarray(img[1,:,:], dtype=np.uint8))
        # cv2.waitKey(0)
        if bbox[4] == 1:
            img = img[:, :, ::-1]
    elif transform is not None:
        img = perturb_fixed(img, tform_augment=transform, target_shape=(w, h), mode=fill_mode,
                            mode_cval=fill_mode_cval)
    else:
        img = perturb(img, augmentation_params=aug_params, target_shape=(w, h), mode=fill_mode,
                      mode_cval=fill_mode_cval)
    # img = brightness_transform(img, brightness_min=0.93, brightness_max=1.4)

    if save_to_dir is not None:
        file_full_name = os.path.basename(fname)
        file_name, file_ext = os.path.splitext(file_full_name)
        fname2 = "%s/%s_DA_%d%s" % (save_to_dir,
                                    file_name, np.random.randint(1e4), file_ext)
        save_image(img, fname2)

    if standardizer is not None:
        img = standardizer(img, is_training)

    # convert to tf format
    return img.transpose(1, 2, 0)


def image_no_preprocessing(fname):
    """Open Image

    Args:
        fname: Image filename

    Returns:
        PIL formatted image

    """
    return Image.open(fname)


def load_images(imgs, preprocessor=image_no_preprocessing):
    """Load batch of images

    Args:
        imgs: a list of image filenames
        preprocessor: image processing function

    Returns:
        a `ndarray` with a batch of images

    """
    return np.array([load_image(f, preprocessor) for f in imgs])


def load_image(img, preprocessor=image_no_preprocessing):
    """Load image

    Args:
        img: a image filename
        preprocessor: image processing function

    Returns:
        a processed image

    """
    if isinstance(img, basestring):
        p_img = preprocessor(img)
        return np.array(p_img, dtype=np.float32).transpose(2, 1, 0)
    elif isinstance(img, np.ndarray):
        return preprocessor(img)
    else:
        raise AssertionError("Unknown image type")


def save_image(x, fname):
    """Save image

    Args:
        x: input array
        fname: filename of the output image

    """
    x = x.transpose(2, 1, 0)
    img = Image.fromarray(x.astype('uint8'), 'RGB')
    print("Saving file: %s" % fname)
    img.save(fname)


def balance_per_class_indices(y, weights):
    """Data balancing utility

    Args:
        y: class labels
        weights: sampling weights per class

    Returns:
        balanced batch as per weights

    """
    y = np.array(y)
    weights = np.array(weights, dtype=float)
    p = np.zeros(len(y))
    for i, weight in enumerate(weights):
        p[y == i] = weight
    return np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                            p=np.array(p) / p.sum())
