# Spatial Transformer Layer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L7 target="_blank"><b>tefla.core.special_layers.spatialtransformer</b></a></span>  (U,  theta,  batch_size=64,  downsample_factor=1.0,  num_transform=1,  name='SpatialTransformer',  **kwargs)</span>

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

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L161 target="_blank"><b>tefla.core.special_layers.subsample</b></a></span>  (inputs,  factor,  name=None)</span>

<h3>Args</h3>


  inputs: A `Tensor` of size [batch, height_in, width_in, channels].
  factor: The subsampling factor.
  name: Optional variable_scope.

<h3>Returns</h3>


  output: A `Tensor` of size [batch, height_out, width_out, channels] with the
input, either intact (if factor == 1) or subsampled (if factor > 1).

 ---------- 

# Strided 2-D convolution with 'SAME' padding

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L179 target="_blank"><b>tefla.core.special_layers.conv2d_same</b></a></span>  (inputs,  num_outputs,  kernel_size,  stride,  rate=1,  name=None,  **kwargs)</span>

When stride > 1, then we do explicit zero-padding, followed by conv2d with
'VALID' padding.

Note that

   net = conv2d_same(inputs, num_outputs, 3, stride=stride)

is equivalent to

   net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
   net = subsample(net, factor=stride)

whereas

   net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

is different when the input's height or width is even, which is why we add the
current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

<h3>Args</h3>


  inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
  num_outputs: An integer, the number of output filters.
  kernel_size: An int with the kernel_size of the filters.
  stride: An integer, the output stride.
  rate: An integer, rate for atrous convolution.
  name: name.

<h3>Returns</h3>


  output: A 4-D tensor of size [batch, height_out, width_out, channels] with
the convolution output.

 ---------- 

# Bottleneck residual unit variant with BN before convolutions

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L274 target="_blank"><b>tefla.core.special_layers.bottleneck_v2</b></a></span>  (inputs,  depth,  depth_bottleneck,  stride,  rate=1,  name=None,  **kwargs)</span>

This is the full preactivation residual unit variant proposed in [2]. See
Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
variant which has an extra bottleneck layer.

When putting together two consecutive ResNet blocks that use this unit, one
should use stride = 2 in the last unit of the first block.

<h3>Args</h3>


  inputs: A tensor of size [batch, height, width, channels].
  depth: The depth of the ResNet unit output.
  depth_bottleneck: The depth of the bottleneck layers.
  stride: The ResNet unit's stride. Determines the amount of downsampling of
 - the units output compared to its input.
  rate: An integer, rate for atrous convolution.
  outputs_collections: Collection to add the ResNet unit output.
  name: Optional variable_scope.

<h3>Returns</h3>


  The ResNet unit's output.

 ---------- 

