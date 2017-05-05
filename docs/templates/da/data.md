# Warp an image according to a given coordinate transformation

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L33 target="_blank"><b>tefla.da.data.fast_warp</b></a></span>  (img,  tf,  output_shape,  mode='constant',  mode_cval=0,  order=0)</span>

This wrapper function is faster than skimage.transform.warp
<h3>Args</h3>


 - **img**: `ndarray`, input image
 - **tf**: For 2-D images, you can directly pass a transformation object
e.g. skimage.transform.SimilarityTransform, or its inverse.
 - **output_shape**: tuple, (rows, cols)
 - **mode**: mode for transformation
available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
 - **mode_cval**: float, Used in conjunction with mode `constant`, the value outside the image boundaries
 - **order**: int, The order of interpolation. The order has to be in the range 0-5:
0: Nearest-neighbor
1: Bi-linear (default)
2: Bi-quadratic
3: Bi-cubic
4: Bi-quartic
5: Bi-quintic

<h3>Returns</h3>


warped, double `ndarray`

 ---------- 

# Transform input image contrast

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L64 target="_blank"><b>tefla.da.data.contrast_transform</b></a></span>  (img,  contrast_min=0.8,  contrast_max=1.2)</span>

Transform the input image contrast by a factor returned by a unifrom
distribution with `contarst_min` and `contarst_max` as params

<h3>Args</h3>


 - **img**: `ndarray`, input image
 - **contrast_min**: float, minimum contrast for transformation
 - **contrast_max**: float, maximum contrast for transformation

<h3>Returns</h3>


`ndarray`, contrast enhanced image

 ---------- 

# Transform input image brightness

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L86 target="_blank"><b>tefla.da.data.brightness_transform</b></a></span>  (img,  brightness_min=0.93,  brightness_max=1.4)</span>

Transform the input image brightness by a factor returned by a unifrom
distribution with `brightness_min` and `brightness_max` as params

<h3>Args</h3>


 - **img**: `ndarray`, input image
 - **brightness_min**: float, minimum contrast for transformation
 - **brightness_max**: float, maximum contrast for transformation

<h3>Returns</h3>


`ndarray`, brightness transformed image

 ---------- 

# Rescale Transform

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L108 target="_blank"><b>tefla.da.data.build_rescale_transform_slow</b></a></span>  (downscale_factor,  image_shape,  target_shape)</span>

This mimics the skimage.transform.resize function.
The resulting image is centered.

<h3>Args</h3>


 - **downscale_factor**: float, >1
 - **image_shape**: tuple(rows, cols), input image shape
 - **target_shape**: tuple(rows, cols), output image shape

<h3>Returns</h3>


rescaled centered image transform instance

 ---------- 

# Rescale Transform

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L142 target="_blank"><b>tefla.da.data.build_rescale_transform_fast</b></a></span>  (downscale_factor,  image_shape,  target_shape)</span>

estimating the correct rescaling transform is slow, so just use the
downscale_factor to define a transform directly. This probably isn't
100% correct, but it shouldn't matter much in practice.
The resulting image is centered.

<h3>Args</h3>


 - **downscale_factor**: float, >1
 - **image_shape**: tuple(rows, cols), input image shape
 - **target_shape**: tuple(rows, cols), output image shape

<h3>Returns</h3>


rescaled and centering transform instance

 ---------- 

# Image cetering transform

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L170 target="_blank"><b>tefla.da.data.build_centering_transform</b></a></span>  (image_shape,  target_shape)</span>

<h3>Args</h3>


 - **image_shape**: tuple(rows, cols), input image shape
 - **target_shape**: tuple(rows, cols), output image shape

<h3>Returns</h3>


a centering transform instance

 ---------- 

# Center Unceter transform

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L188 target="_blank"><b>tefla.da.data.build_center_uncenter_transforms</b></a></span>  (image_shape)</span>

These are used to ensure that zooming and rotation happens around the center of the image.
Use these transforms to center and uncenter the image around such a transform.

<h3>Args</h3>


 - **image_shape**: tuple(rows, cols), input image shape

<h3>Returns</h3>


a center and an uncenter transform instance

 ---------- 

# Augmentation transform

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L209 target="_blank"><b>tefla.da.data.build_augmentation_transform</b></a></span>  (zoom=  (1.0,  1.0),  rotation=0,  shear=0,  translation=  (0,  0),  flip=False)</span>

It performs zooming, rotation, shear, translation and flip operation
Affine Transformation on the input image

<h3>Args</h3>


 - **zoom**: a tuple(zoom_rows, zoom_cols)
 - **rotation**: float, Rotation angle in counter-clockwise direction as radians.
 - **shear**: float, shear angle in counter-clockwise direction as radians
 - **translation**: tuple(trans_rows, trans_cols)
 - **flip**: bool, flip an image

<h3>Returns</h3>


augment tranform instance

 ---------- 

# Random perturbation

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L236 target="_blank"><b>tefla.da.data.random_perturbation_transform</b></a></span>  (zoom_range,  rotation_range,  shear_range,  translation_range,  do_flip=True,  allow_stretch=False,  rng=<module  'numpy.random'  from  '/home/artelus_server/work/venv/caffe_v/local/lib/python2.7/site-packages/numpy/random/__init__.pyc'>)</span>

It perturbs the image randomly

<h3>Args</h3>


 - **zoom_range**: a tuple(min_zoom, max_zoom)
e.g.: (1/1.15, 1.15)
 - **rotation_range**: a tuple(min_angle, max_angle)
e.g.: (0. 360)
 - **shear_range**: a tuple(min_shear, max_shear)
e.g.: (0, 15)
 - **translation_range**: a tuple(min_shift, max_shift)
e.g.: (-15, 15)
 - **do_flip**: bool, flip an image
 - **allow_stretch**: bool, stretch an image
 - **rng**: an instance

<h3>Returns</h3>


augment transform instance

 ---------- 

# crop an image

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L289 target="_blank"><b>tefla.da.data.definite_crop</b></a></span>  (img,  bbox)</span>

<h3>Args</h3>


 - **img**: `ndarray`, input image
 - **bbox**: list, with crop co-ordinates and width and height
e.g.: [x, y, width, height]

<h3>Returns</h3>


returns cropped image

 ---------- 

# Perturb image

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L304 target="_blank"><b>tefla.da.data.perturb</b></a></span>  (img,  augmentation_params,  target_shape,  rng=<module  'numpy.random'  from  '/home/artelus_server/work/venv/caffe_v/local/lib/python2.7/site-packages/numpy/random/__init__.pyc'>,  mode='constant',  mode_cval=0)</span>

It perturbs an image with augmentation transform

<h3>Args</h3>


 - **img**: a `ndarray`, input image
 - **augmentation_paras**: a dict, with augmentation name as keys and values as params
 - **target_shape**: a tuple(rows, cols), output image shape
 - **rng**: an instance for random number generation
 - **mode**: mode for transformation
available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
 - **mode_cval**: float, Used in conjunction with mode `constant`,
the value outside the image boundaries

<h3>Returns</h3>


a `ndarray` of transformed image

 ---------- 

# Perturb image rescaled

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L334 target="_blank"><b>tefla.da.data.perturb_rescaled</b></a></span>  (img,  scale,  augmentation_params,  target_shape=  (224,  224),  rng=<module  'numpy.random'  from  '/home/artelus_server/work/venv/caffe_v/local/lib/python2.7/site-packages/numpy/random/__init__.pyc'>,  mode='constant',  mode_cval=0)</span>

It perturbs an image with augmentation transform

<h3>Args</h3>


 - **img**: a `ndarray`, input image
 - **scale**: float, >1, downscaling factor.
 - **augmentation_paras**: a dict, with augmentation name as keys and values as params
 - **target_shape**: a tuple(rows, cols), output image shape
 - **rng**: an instance for random number generation
 - **mode**: mode for transformation
available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
 - **mode_cval**: float, Used in conjunction with mode `constant`,
the value outside the image boundaries

<h3>Returns</h3>


a `ndarray` of transformed image

 ---------- 

# Perturb image Determinastic

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L364 target="_blank"><b>tefla.da.data.perturb_fixed</b></a></span>  (img,  tform_augment,  target_shape=  (50,  50),  mode='constant',  mode_cval=0)</span>

It perturbs an image with augmentation transform with determinastic params
used for validation/testing data

<h3>Args</h3>


 - **img**: a `ndarray`, input image
 - **augmentation_paras**: a dict, with augmentation name as keys and values as params
 - **target_shape**: a tuple(rows, cols), output image shape
 - **mode**: mode for transformation
available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
 - **mode_cval**: float, Used in conjunction with mode `constant`,
the value outside the image boundaries

<h3>Returns</h3>


a `ndarray` of transformed image

 ---------- 

# Load augmented image with output shape (w, h)

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L403 target="_blank"><b>tefla.da.data.load_augment</b></a></span>  (fname,  preprocessor,  w,  h,  is_training,  aug_params={'zoom_range':(1.0,  1.0),  'translation_range':(0,  0),  'shear_range':(0,  0),  'do_flip':  False,  'allow_stretch':  False,  'rotation_range':(0,  0)},  transform=None,  bbox=None,  fill_mode='constant',  fill_mode_cval=0,  standardizer=None,  save_to_dir=None)</span>

Default arguments return non augmented image of shape (w, h).
To apply a fixed transform (color augmentation) specify transform
(color_vec).
To generate a random augmentation specify aug_params and sigma.

<h3>Args</h3>


 - **fname**: string, image filename
 - **preprocessor**: real-time image processing/crop
 - **w**: int, width of target image
 - **h**: int, height of target image
 - **is_training**: bool, if True then training else validation
 - **aug_params**: a dict, augmentation params
 - **transform**: transform instance
 - **bbox**: object bounding box
 - **fll_mode**: mode for transformation
available modes: {`constant`, `edge`, `symmetric`, `reflect`, `wrap`}
 - **fill_mode_cval**: float, Used in conjunction with mode `constant`,
the value outside the image boundaries
 - **standardizer**: image standardizer, zero mean, unit variance image
 e.g.: samplewise standardized each image based on its own value
 - **save_to_dir**: a string, path to save image, save output image to a dir

<h3>Returns</h3>


augmented image

 ---------- 

# Open Image

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L467 target="_blank"><b>tefla.da.data.image_no_preprocessing</b></a></span>  (fname)</span>

<h3>Args</h3>


 - **fname**: Image filename

<h3>Returns</h3>


PIL formatted image

 ---------- 

# Load batch of images

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L480 target="_blank"><b>tefla.da.data.load_images</b></a></span>  (imgs,  preprocessor=<function  image_no_preprocessing  at  0x7ff0f5545c08>)</span>

<h3>Args</h3>


 - **imgs**: a list of image filenames
 - **preprocessor**: image processing function

<h3>Returns</h3>


a `ndarray` with a batch of images

 ---------- 

# Load image

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L494 target="_blank"><b>tefla.da.data.load_image</b></a></span>  (img,  preprocessor=<function  image_no_preprocessing  at  0x7ff0f5545c08>)</span>

<h3>Args</h3>


 - **img**: a image filename
 - **preprocessor**: image processing function

<h3>Returns</h3>


a processed image

 ---------- 

# Save image

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L514 target="_blank"><b>tefla.da.data.save_image</b></a></span>  (x,  fname)</span>

<h3>Args</h3>


 - **x**: input array
 - **fname**: filename of the output image

 ---------- 

# Data balancing utility

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/data.py#L528 target="_blank"><b>tefla.da.data.balance_per_class_indices</b></a></span>  (y,  weights)</span>

<h3>Args</h3>


 - **y**: class labels
 - **weights**: sampling weights per class

<h3>Returns</h3>


balanced batch as per weights

 ---------- 

