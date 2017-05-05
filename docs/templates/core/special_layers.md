# Spatial Transformer Layer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L10 target="_blank"><b>tefla.core.special_layers.spatialtransformer</b></a></span>  (U,  theta,  batch_size=64,  downsample_factor=1.0,  num_transform=1,  name='SpatialTransformer',  **kwargs)</span>

Implements a spatial transformer layer as described in [1]_.
It's based on lasagne implementation in [2]_, modified by Mrinal Haloi

<h3>Args</h3>


 - **U**: float
The output of a convolutional net should have the
shape [batch_size, height, width, num_channels].
 - **theta**: float
The output of the localisation network should be [batch_size, num_transform, 6] or [batch_size, 6] if num_transform=1
```python`theta`` to : - identity = np.array([[1., 0., 0.], -  [0., 1., 0.]]) - identity = identity.flatten() - theta = tf.Variable(initial_value=identity)
```
 - **downsample_factor**: a float, determines output shape, downsample input shape by downsample_factor

<h3>Returns</h3>


spatial transformed output of the network

 ---------- 

# Subsamples the input along the spatial dimensions

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L164 target="_blank"><b>tefla.core.special_layers.subsample</b></a></span>  (inputs,  factor,  name=None)</span>

<h3>Args</h3>


 - **inputs**: A `Tensor` of size [batch, height_in, width_in, channels].
 - **factor**: The subsampling factor.
 - **name**: Optional variable_scope.

<h3>Returns</h3>


output: A `Tensor` of size [batch, height_out, width_out, channels] with the
input, either intact (if factor == 1) or subsampled (if factor > 1).

 ---------- 

# Strided 2-D convolution with 'SAME' padding

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L182 target="_blank"><b>tefla.core.special_layers.conv2d_same</b></a></span>  (inputs,  num_outputs,  kernel_size,  stride,  rate=1,  name=None,  **kwargs)</span>

When stride > 1, then we do explicit zero-padding, followed by conv2d with
'VALID' padding.

Note that

   net = conv2d_same(inputs, num_outputs, 3, stride=stride)

is equivalent to

   net = conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
   net = subsample(net, factor=stride)

whereas

   net = conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

is different when the input's height or width is even, which is why we add the
current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

<h3>Args</h3>


 - **inputs**: A 4-D tensor of size [batch, height_in, width_in, channels].
 - **num_outputs**: An integer, the number of output filters.
 - **kernel_size**: An int with the kernel_size of the filters.
 - **stride**: An integer, the output stride.
 - **rate**: An integer, rate for atrous convolution.
 - **name**: name.

<h3>Returns</h3>


output: A 4-D tensor of size [batch, height_out, width_out, channels] with
the convolution output.

 ---------- 

# Bottleneck residual unit variant with BN before convolutions

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L230 target="_blank"><b>tefla.core.special_layers.bottleneck_v1</b></a></span>  (inputs,  depth,  depth_bottleneck,  stride,  rate=1,  name=None,  **kwargs)</span>

This is the full preactivation residual unit variant proposed in [2]. See
Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
variant which has an extra bottleneck layer.

When putting together two consecutive ResNet blocks that use this unit, one
should use stride = 2 in the last unit of the first block.

<h3>Args</h3>


 - **inputs**: A tensor of size [batch, height, width, channels].
 - **depth**: The depth of the ResNet unit output.
 - **depth_bottleneck**: The depth of the bottleneck layers.
 - **stride**: The ResNet unit's stride. Determines the amount of downsampling of
the units output compared to its input.
 - **rate**: An integer, rate for atrous convolution.
 - **outputs_collections**: Collection to add the ResNet unit output.
 - **name**: Optional variable_scope.

<h3>Returns</h3>


The ResNet unit's output.

 ---------- 

# Bottleneck residual unit variant with BN before convolutions

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L277 target="_blank"><b>tefla.core.special_layers.bottleneck_v2</b></a></span>  (inputs,  depth,  depth_bottleneck,  stride,  rate=1,  name=None,  **kwargs)</span>

This is the full preactivation residual unit variant proposed in [2]. See
Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
variant which has an extra bottleneck layer.

When putting together two consecutive ResNet blocks that use this unit, one
should use stride = 2 in the last unit of the first block.

<h3>Args</h3>


 - **inputs**: A tensor of size [batch, height, width, channels].
 - **depth**: The depth of the ResNet unit output.
 - **depth_bottleneck**: The depth of the bottleneck layers.
 - **stride**: The ResNet unit's stride. Determines the amount of downsampling of
the units output compared to its input.
 - **rate**: An integer, rate for atrous convolution.
 - **outputs_collections**: Collection to add the ResNet unit output.
 - **name**: Optional variable_scope.

<h3>Returns</h3>


The ResNet unit's output.

 ---------- 

# DenseCRF over unnormalised predictions

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L405 target="_blank"><b>tefla.core.special_layers.dense_crf</b></a></span>  (probs,  img=None,  n_classes=15,  n_iters=10,  sxy_gaussian=  (1,  1),  compat_gaussian=4,  kernel_gaussian=<KernelType.DIAG_KERNEL:  1>,  normalisation_gaussian=<NormalizationType.NORMALIZE_SYMMETRIC:  3>,  sxy_bilateral=  (49,  49),  compat_bilateral=2,  srgb_bilateral=  (13,  13,  13),  kernel_bilateral=<KernelType.DIAG_KERNEL:  1>,  normalisation_bilateral=<NormalizationType.NORMALIZE_SYMMETRIC:  3>)</span>
   More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

<h3>Args</h3>


 - **probs**: class probabilities per pixel.
 - **img**: if given, the pairwise bilateral potential on raw RGB values will be computed.
 - **n_iters**: number of iterations of MAP inference.
 - **sxy_gaussian**: standard deviations for the location component
of the colour-independent term.
 - **compat_gaussian**: label compatibilities for the colour-independent
term (can be a number, a 1D array, or a 2D array).
 - **kernel_gaussian**: kernel precision matrix for the colour-independent
term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
 - **normalisation_gaussian**: normalisation for the colour-independent term
(possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
 - **sxy_bilateral**: standard deviations for the location component of the colour-dependent term.
 - **compat_bilateral**: label compatibilities for the colour-dependent
term (can be a number, a 1D array, or a 2D array).
 - **srgb_bilateral**: standard deviations for the colour component
of the colour-dependent term.
 - **kernel_bilateral**: kernel precision matrix for the colour-dependent term
(can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
 - **normalisation_bilateral**: normalisation for the colour-dependent term
(possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

<h3>Returns</h3>


Refined predictions after MAP inference.

 ---------- 

# ResNeXt Block

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L487 target="_blank"><b>tefla.core.special_layers.resnext_block</b></a></span>  (inputs,  nb_blocks,  out_channels,  is_training,  reuse,  cardinality,  downsample=False,  downsample_strides=2,  activation=<function  relu  at  0x7ff0da7df8c0>,  batch_norm=None,  batch_norm_args=None,  name='ResNeXtBlock',  **kwargs)</span>
resnext paper https://arxiv.org/pdf/1611.05431.pdf

<h3>Args</h3>


 - **inputs**: `Tensor`. Inputs 4-D Layer.
 - **nb_blocks**: `int`. Number of layer blocks.
 - **out_channels**: `int`. The number of convolutional filters of the
layers surrounding the bottleneck layer.
 - **cardinality**: `int`. Number of aggregated residual transformations.
 - **downsample**: `bool`. If True, apply downsampling using
'downsample_strides' for strides.
 - **downsample_strides**: `int`. The strides to use when downsampling.
 - **activation**: `function` (returning a `Tensor`).
 - **batch_norm**: `bool`. If True, apply batch normalization.
 - use_ bias: `bool`. If True, a bias is used.
 - **w_init**: `function`, Weights initialization.
 - **b_init**: `tf.Tensor`. Bias initialization.
 - **w_regularizer**: `function`. Add a regularizer to this
 - **weight_decay**: `float`. Regularizer decay parameter. Default: 0.001.
 - **trainable**: `bool`. If True, weights will be trainable.
 - **reuse**: `bool`. If True and 'scope' is provided, this layer variables
will be reused (shared).
override name.
 - **name**: A name for this layer (optional). Default: 'ResNeXtBlock'.

<h3>Returns</h3>


4-D Tensor [batch, new height, new width, out_channels].

 ---------- 

