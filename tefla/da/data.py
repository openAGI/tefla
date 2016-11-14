# Original code from: https://github.com/sveitser/kaggle_diabetic
# Original MIT license: https://github.com/sveitser/kaggle_diabetic/blob/master/LICENSE
"""data augmentation.

The code for data augmentation originally comes from
https://github.com/benanne/kaggle-ndsb/blob/master/data.py
"""
from __future__ import division, print_function

from PIL import Image

import skimage
import skimage.transform
from skimage.transform._warps_cy import _warp_fast

from standardizer import *
from tefla.core.data_load_ops import *

no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}


def fast_warp(img, tf, output_shape, mode='constant', mode_cval=0, order=0):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params
    t_img = np.zeros((img.shape[0],) + output_shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, output_shape=output_shape,
                              mode=mode, cval=mode_cval, order=order)
    return t_img


def build_centering_transform(image_shape, target_shape):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array(
        [image_shape[1], image_shape[0]]) / 2.0 - 0.5  # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False):
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
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)


def definite_crop(img, bbox):
    img = img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return img


def perturb(img, augmentation_params, target_shape, rng=np.random, mode='constant', mode_cval=0):
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 0.5
    # img[-1, :] = 0.5
    # img[:, 0] = 0.5
    # img[:, -1] = 0.5
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = random_perturbation_transform(rng=rng, **augmentation_params)
    # shift to center, augment, shift back (for the rotation/shearing)
    tform_augment = tform_uncenter + tform_augment + tform_center
    return fast_warp(img, tform_centering + tform_augment,
                     output_shape=target_shape,
                     mode=mode, mode_cval=mode_cval)


# for test-time augmentation
def perturb_fixed(img, tform_augment, target_shape=(50, 50), mode='constant', mode_cval=0):
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
    """
    img = load_image(fname, preprocessor)

    # target shape should be (h, w) i.e. (rows, cols). need to revisit when we do non-square shapes

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

    if save_to_dir:
        file_full_name = os.path.basename(fname)
        file_name, file_ext = os.path.splitext(file_full_name)
        fname2 = "%s/%s_DA_%d%s" % (save_to_dir, file_name, np.random.randint(1e4), file_ext)
        save_image(img, fname2)

    if standardizer:
        img = standardizer(img, is_training)

    # convert to tf format
    return img.transpose(1, 2, 0)


def image_no_preprocessing(fname):
    return Image.open(fname)


def load_images(imgs, preprocessor=image_no_preprocessing):
    return np.array([load_image(f, preprocessor) for f in imgs])


def load_image(img, preprocessor=image_no_preprocessing):
    if isinstance(img, basestring):
        p_img = preprocessor(img)
        return np.array(p_img, dtype=np.float32).transpose(2, 1, 0)
    elif isinstance(img, np.ndarray):
        return preprocessor(img)
    else:
        raise AssertionError("Unknown image type")


def save_image(x, fname):
    x = x.transpose(2, 1, 0)
    img = Image.fromarray(x.astype('uint8'), 'RGB')
    print("Saving file: %s" % fname)
    img.save(fname)


def balance_per_class_indices(y, weights):
    y = np.array(y)
    weights = np.array(weights, dtype=float)
    p = np.zeros(len(y))
    for i, weight in enumerate(weights):
        p[y == i] = weight
    return np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                            p=np.array(p) / p.sum())
