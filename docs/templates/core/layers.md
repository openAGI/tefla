# Define input layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.input</b></span>  (shape,  name='inputs',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **shape**: A `Tensor`, define the input shape
e.g. for image input [batch_size, height, width, depth]
 - **name**: A optional score/name for this op
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A placeholder for the input

 ---------- 

# Adds a fully connected layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.fully_connected</b></span>  (x,  n_output,  is_training,  reuse,  trainable=True,  w_init=<function  _initializer  at  0x7f97acecf668>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7f97dcfee2a8>,  name='fc',  batch_norm=None,  batch_norm_args=None,  activation=None,  outputs_collections=None,  use_bias=True)</span>

`fully_connected` creates a variable called `weights`, representing a fully
connected weight matrix, which is multiplied by the `x` to produce a
`Tensor` of hidden units. If a `batch_norm` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `batch_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank greater than 2, then `x` is flattened
prior to the initial matrix multiply by `weights`.

<h3>Args</h3>


 - **x**: A `Tensor` of with at least rank 2 and value for the last dimension,
i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
 - **is_training**: Bool, training or testing
 - **n_output**: Integer or long, the number of output units in the layer.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **activation**: activation function, set to None to skip it and maintain
a linear activation.
 - **batch_norm**: normalization function to use. If
 -`batch_norm` is `True` then google original implementation is used and
if another function is provided then it is applied.
default set to None for no normalizer function
 - **batch_norm_args**: normalization function parameters.
 - **w_init**: An initializer for the weights.
 - **w_regularizer**: Optional regularizer for the weights.
 - **b_init**: An initializer for the biases. If None skip biases.
 - **outputs_collections**: The collections to which the outputs are added.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
 - **name**: Optional name or scope for variable_scope/name_scope.
 - **use_bias**: Whether to add bias or not

<h3>Returns</h3>


The 2-D `Tensor` variable representing the result of the series of operations.
e.g: 2-D `Tensor` [batch, n_output].

 ---------- 

# Adds a 2D convolutional layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.conv2d</b></span>  (x,  n_output_channels,  is_training,  reuse,  trainable=True,  filter_size=  (3,  3),  stride=  (1,  1),  padding='SAME',  w_init=<function  _initializer  at  0x7f97acecf758>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7f97dcfee2a8>,  untie_biases=False,  name='conv2d',  batch_norm=None,  batch_norm_args=None,  activation=None,  use_bias=True,  outputs_collections=None)</span>

`convolutional layer` creates a variable called `weights`, representing a conv
weight matrix, which is multiplied by the `x` to produce a
`Tensor` of hidden units. If a `batch_norm` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `batch_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank 4

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of with at least rank 2 and value for the last dimension,
i.e. `[batch_size, in_height, in_width, depth]`,
 - **is_training**: Bool, training or testing
 - **n_output**: Integer or long, the number of output units in the layer.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.

<h3>Returns</h3>


The 4-D `Tensor` variable representing the result of the series of operations.
e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

 ---------- 

# Adds a 2D dilated convolutional layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.dilated_conv2d</b></span>  (x,  n_output_channels,  is_training,  reuse,  trainable=True,  filter_size=  (3,  3),  dilation=1,  padding='SAME',  w_init=<function  _initializer  at  0x7f97acecf848>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7f97dcfee2a8>,  untie_biases=False,  name='dilated_conv2d',  batch_norm=None,  batch_norm_args=None,  activation=None,  use_bias=True,  outputs_collections=None)</span>

also known as convolution with holes or atrous convolution.
If the rate parameter is equal to one, it performs regular 2-D convolution.
If the rate parameter
is greater than one, it performs convolution with holes, sampling the input
values every rate pixels in the height and width dimensions.
`convolutional layer` creates a variable called `weights`, representing a conv
weight matrix, which is multiplied by the `x` to produce a
`Tensor` of hidden units. If a `batch_norm` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `batch_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank 4

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of with rank 4 and value for the last dimension,
i.e. `[batch_size, in_height, in_width, depth]`,
 - **is_training**: Bool, training or testing
 - **n_output**: Integer or long, the number of output units in the layer.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.

<h3>Returns</h3>


The 4-D `Tensor` variable representing the result of the series of operations.
e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

 ---------- 

# Adds a 2D seperable convolutional layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.separable_conv2d</b></span>  (x,  n_output_channels,  is_training,  reuse,  trainable=True,  filter_size=  (3,  3),  stride=  (1,  1),  depth_multiplier=8,  padding='SAME',  w_init=<function  _initializer  at  0x7f97acecf938>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7f97dcfee2a8>,  untie_biases=False,  name='separable_conv2d',  batch_norm=None,  batch_norm_args=None,  activation=None,  use_bias=True,  outputs_collections=None)</span>

Performs a depthwise convolution that acts separately on channels followed by
a pointwise convolution that mixes channels. Note that this is separability between
dimensions [1, 2] and 3, not spatial separability between dimensions 1 and 2.
`convolutional layer` creates two variable called `depthwise_W` and `pointwise_W`,
`depthwise_W` is multiplied by `x` to produce depthwise conolution, which is multiplied by
the `pointwise_W` to produce a output `Tensor`
If a `batch_norm` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `batch_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank 4

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of with rank 4 and value for the last dimension,
i.e. `[batch_size, in_height, in_width, depth]`,
 - **is_training**: Bool, training or testing
 - **n_output**: Integer or long, the number of output units in the layer.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **filter_size**: a list or tuple of 2 positive integers specifying the spatial
 - dimensions of of the filters.
 - **depth_multiplier**:  A positive int32. the number of depthwise convolution output channels for
each input channel. The total number of depthwise convolution output
channels will be equal to `num_filters_in * depth_multiplier
 - **padding**: one of `"VALID"` or `"SAME"`.
 - **activation**: activation function, set to None to skip it and maintain
a linear activation.
 - **batch_norm**: normalization function to use. If
`batch_norm` is `True` then google original implementation is used and
if another function is provided then it is applied.
default set to None for no normalizer function
 - **batch_norm_args**: normalization function parameters.
 - **w_init**: An initializer for the weights.
 - **w_regularizer**: Optional regularizer for the weights.
 - **untie_biases**: spatial dimensions wise baises
 - **b_init**: An initializer for the biases. If None skip biases.
 - **outputs_collections**: The collections to which the outputs are added.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
 - **name**: Optional name or scope for variable_scope/name_scope.
 - **use_bias**: Whether to add bias or not

<h3>Returns</h3>


The 4-D `Tensor` variable representing the result of the series of operations.
e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

 ---------- 

# Adds a 2D sdepthwise convolutional layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.depthwise_conv2d</b></span>  (x,  is_training,  reuse,  trainable=True,  filter_size=  (3,  3),  stride=  (1,  1),  depth_multiplier=8,  padding='SAME',  w_init=<function  _initializer  at  0x7f97acecfa28>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7f97dcfee2a8>,  untie_biases=False,  name='depthwise_conv2d',  batch_norm=None,  batch_norm_args=None,  activation=None,  use_bias=True,  outputs_collections=None)</span>

Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter
tensor of shape [filter_height, filter_width, in_channels, channel_multiplier] containing
in_channels convolutional filters of depth 1, depthwise_conv2d applies a different filter
to each input channel (expanding from 1 channel to channel_multiplier channels for each),
then concatenates the results together. The output has in_channels * channel_multiplier channels.
If a `batch_norm` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `batch_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank 4

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of with rank 4 and value for the last dimension,
i.e. `[batch_size, in_height, in_width, depth]`,
 - **is_training**: Bool, training or testing
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **filter_size**: a list or tuple of 2 positive integers specifying the spatial
dimensions of of the filters.
 - **depth_multiplier**:  A positive int32. the number of depthwise convolution output channels for
each input channel. The total number of depthwise convolution output
channels will be equal to `num_filters_in * depth_multiplier
 - **padding**: one of `"VALID"` or `"SAME"`.
 - **activation**: activation function, set to None to skip it and maintain
a linear activation.
 - **batch_norm**: normalization function to use. If
`batch_norm` is `True` then google original implementation is used and
if another function is provided then it is applied.
default set to None for no normalizer function
 - **batch_norm_args**: normalization function parameters.
 - **w_init**: An initializer for the weights.
 - **w_regularizer**: Optional regularizer for the weights.
 - **untie_biases**: spatial dimensions wise baises
 - **b_init**: An initializer for the biases. If None skip biases.
 - **outputs_collections**: The collections to which the outputs are added.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
 - **name**: Optional name or scope for variable_scope/name_scope.
 - **use_bias**: Whether to add bias or not

<h3>Returns</h3>


The tensor variable representing the result of the series of operations.
e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

 ---------- 

# Adds a 2D upsampling or deconvolutional layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.upsample2d</b></span>  (input_,  output_shape,  is_training,  reuse,  trainable=True,  filter_size=  (5,  5),  stride=  (2,  2),  w_init=<function  _initializer  at  0x7f97acecfb18>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7f97dcfee2a8>,  batch_norm=None,  activation=None,  name='deconv2d',  use_bias=True,  with_w=False,  outputs_collections=None,  **unused)</span>

his operation is sometimes called "deconvolution" after Deconvolutional Networks,
but is actually the transpose (gradient) of conv2d rather than an actual deconvolution.
If a `batch_norm` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `batch_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank 4

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of with at least rank 2 and value for the last dimension,
i.e. `[batch_size, in_height, in_width, depth]`,
 - **is_training**: Bool, training or testing
 - **output_shape**: 4D tensor, the output shape
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **filter_size**: a list or tuple of 2 positive integers specifying the spatial
dimensions of of the filters.
 - **stride**: a tuple or list of 2 positive integers specifying the stride at which to
compute output.
 - **padding**: one of `"VALID"` or `"SAME"`.
 - **activation**: activation function, set to None to skip it and maintain
a linear activation.
 - **batch_norm**: normalization function to use. If
`batch_norm` is `True` then google original implementation is used and
if another function is provided then it is applied.
default set to None for no normalizer function
 - **batch_norm_args**: normalization function parameters.
 - **w_init**: An initializer for the weights.
 - **w_regularizer**: Optional regularizer for the weights.
 - **b_init**: An initializer for the biases. If None skip biases.
 - **outputs_collections**: The collections to which the outputs are added.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
 - **name**: Optional name or scope for variable_scope/name_scope.
 - **use_bias**: Whether to add bias or not

<h3>Returns</h3>


The tensor variable representing the result of the series of operations.
e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

 ---------- 

# Adds a 2D highway convolutional layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.highway_conv2d</b></span>  (x,  n_output,  is_training,  reuse,  trainable=True,  filter_size=  (3,  3),  stride=  (1,  1),  padding='SAME',  w_init=<function  _initializer  at  0x7f97acecfcf8>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7f97dcfee2a8>,  name='highway_conv2d',  activation=None,  use_bias=True,  outputs_collections=None)</span>

https://arxiv.org/abs/1505.00387
If a `batch_norm` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `batch_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank 4

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of with at least rank 2 and value for the last dimension,
i.e. `[batch_size, in_height, in_width, depth]`,
 - **is_training**: Bool, training or testing
 - **n_output**: Integer or long, the number of output units in the layer.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **filter_size**: a list or tuple of 2 positive integers specifying the spatial
dimensions of of the filters.
 - **stride**: a tuple or list of 2 positive integers specifying the stride at which to
compute output.
 - **padding**: one of `"VALID"` or `"SAME"`.
 - **activation**: activation function, set to None to skip it and maintain
a linear activation.
 - **batch_norm**: normalization function to use. If
`batch_norm` is `True` then google original implementation is used and
if another function is provided then it is applied.
default set to None for no normalizer function
 - **batch_norm_args**: normalization function parameters.
 - **w_init**: An initializer for the weights.
 - **w_regularizer**: Optional regularizer for the weights.
 - **untie_biases**: spatial dimensions wise baises
 - **b_init**: An initializer for the biases. If None skip biases.
 - **outputs_collections**: The collections to which the outputs are added.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
 - **name**: Optional name or scope for variable_scope/name_scope.
 - **use_bias**: Whether to add bias or not

<h3>Returns</h3>


The `Tensor` variable representing the result of the series of operations.
e.g.: 4-D `Tensor` [batch, new_height, new_width, n_output].

 ---------- 

# Adds a fully connected highway layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.highway_fc2d</b></span>  (x,  n_output,  is_training,  reuse,  trainable=True,  filter_size=  (3,  3),  w_init=<function  _initializer  at  0x7f97acecfde8>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7f97dcfee2a8>,  name='highway_fc2d',  activation=None,  use_bias=True,  outputs_collections=None)</span>

https://arxiv.org/abs/1505.00387
If a `batch_norm` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `batch_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank greater than 2, then `x` is flattened
prior to the initial matrix multiply by `weights`.

<h3>Args</h3>


 - **x**: A 2-D/4-D `Tensor` of with at least rank 2 and value for the last dimension,
i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
 - **is_training**: Bool, training or testing
 - **n_output**: Integer or long, the number of output units in the layer.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **activation**: activation function, set to None to skip it and maintain
a linear activation.
 - **batch_norm**: normalization function to use. If
`batch_norm` is `True` then google original implementation is used and
if another function is provided then it is applied.
default set to None for no normalizer function
 - **batch_norm_args**: normalization function parameters.
 - **w_init**: An initializer for the weights.
 - **w_regularizer**: Optional regularizer for the weights.
 - **b_init**: An initializer for the biases. If None skip biases.
 - **outputs_collections**: The collections to which the outputs are added.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
 - **name**: Optional name or scope for variable_scope/name_scope.
 - **use_bias**: Whether to add bias or not

<h3>Returns</h3>


The 2-D `Tensor` variable representing the result of the series of operations.
e.g.: 2-D `Tensor` [batch_size, n_output]

 ---------- 

# Max pooling layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.max_pool</b></span>  (x,  filter_size=  (3,  3),  stride=  (2,  2),  padding='SAME',  name='pool',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: A 4-D 'Tensor` of shape `[batch_size, height, width, channels]`
 - **filter_size**: A list of length 2: [kernel_height, kernel_width] of the
pooling kernel over which the op is computed. Can be an int if both
values are the same.
 - **stride**: A list of length 2: [stride_height, stride_width].
 - **padding**: The padding method, either 'VALID' or 'SAME'.
 - **outputs_collections**: The collections to which the outputs are added.
 - **name**: Optional scope/name for name_scope.

<h3>Returns</h3>


A `Tensor` representing the results of the pooling operation.
e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

 ---------- 

# Fractional pooling layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.fractional_pool</b></span>  (x,  pooling_ratio=[1.0,  1.44,  1.73,  1.0],  pseudo_random=None,  determinastic=None,  overlapping=None,  name='fractional_pool',  seed=None,  seed2=None,  type='avg',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
 - **pooling_ratio**: A list of floats that has length >= 4. Pooling ratio for each
dimension of value, currently only supports row and col dimension and should
be >= 1.0. For example, a valid pooling ratio looks like [1.0, 1.44, 1.73, 1.0].
The first and last elements must be 1.0 because we don't allow pooling on batch and
channels dimensions. 1.44 and 1.73 are pooling ratio on height and width
dimensions respectively.
 - **pseudo_random**: An optional bool. Defaults to False. When set to True, generates
the pooling sequence in a pseudorandom fashion, otherwise, in a random fashion.
Check paper Benjamin Graham, Fractional Max-Pooling for difference between
pseudorandom and random.
 - **overlapping**: An optional bool. Defaults to False. When set to True, it means when pooling,
the values at the boundary of adjacent pooling cells are used by both cells.
For example: index 0 1 2 3 4
value 20 5 16 3 7; If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used
twice. The result would be [41/3, 26/3] for fractional avg pooling.
 - **deterministic**: An optional bool. Defaults to False. When set to True, a fixed pooling
region will be used when iterating over a FractionalAvgPool node in the computation
graph. Mainly used in unit test to make FractionalAvgPool deterministic.
 - **seed**: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero,
the random number generator is seeded by the given seed. Otherwise,
it is seeded by a random seed.
 - **seed2**: An optional int. Defaults to 0. An second seed to avoid seed collision.
 - **outputs_collections**: The collections to which the outputs are added.
 - **type**: avg or max pool
 - **name**: Optional scope/name for name_scope.

<h3>Returns</h3>


A 4-D `Tensor` representing the results of the pooling operation.
e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

 ---------- 

# RMS pooling layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.rms_pool_2d</b></span>  (x,  filter_size=  (3,  3),  stride=  (2,  2),  padding='SAME',  name='pool',  epsilon=1e-12,  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
 - **filter_size**: A list of length 2: [kernel_height, kernel_width] of the
pooling kernel over which the op is computed. Can be an int if both
values are the same.
 - **stride**: A list of length 2: [stride_height, stride_width].
 - **padding**: The padding method, either 'VALID' or 'SAME'.
 - **outputs_collections**: The collections to which the outputs are added.
 - **name**: Optional scope/name for name_scope.
 - **epsilon**: prevents divide by zero

<h3>Returns</h3>


A 4-D `Tensor` representing the results of the pooling operation.
e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

 ---------- 

# Avg pooling layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.avg_pool_2d</b></span>  (x,  filter_size=  (3,  3),  stride=  (2,  2),  padding='SAME',  name=None,  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
 - **filter_size**: A list of length 2: [kernel_height, kernel_width] of the
pooling kernel over which the op is computed. Can be an int if both
values are the same.
 - **stride**: A list of length 2: [stride_height, stride_width].
 - **padding**: The padding method, either 'VALID' or 'SAME'.
 - **outputs_collections**: The collections to which the outputs are added.
 - **name**: Optional scope/name for name_scope.

<h3>Returns</h3>


A 4-D `Tensor` representing the results of the pooling operation.
e.g.: 4-D `Tensor` [batch, new_height, new_width, channels].

 ---------- 

# Gloabl pooling layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.global_avg_pool</b></span>  (x,  name='global_avg_pool',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`
 - **outputs_collections**: The collections to which the outputs are added.
 - **name**: Optional scope/name for name_scope.

<h3>Returns</h3>


A 4-D `Tensor` representing the results of the pooling operation.
e.g.: 4-D `Tensor` [batch, 1, 1, channels].

 ---------- 

# Feature max pooling layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.feature_max_pool_1d</b></span>  (x,  stride=2,  name='pool',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: A 2-D tensor of shape `[batch_size, channels]`
 - **stride**: A int.
 - **outputs_collections**: The collections to which the outputs are added.
 - **name**: Optional scope/name for name_scope.

<h3>Returns</h3>


A 2-D `Tensor` representing the results of the pooling operation.
e.g.: 2-D `Tensor` [batch_size, new_channels]

 ---------- 

# Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.batch_norm_tf</b></span>  (x,  name='bn',  scale=False,  updates_collections=None,  **kwargs)</span>
"Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift", Sergey Ioffe, Christian Szegedy
Can be used as a normalizer function for conv2d and fully_connected.
Note: When is_training is True the moving_mean and moving_variance need to be
updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
they need to be added as a dependency to the `train_op`, example:
`update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)`
`if update_ops:`
`updates = tf.group(*update_ops)`
`total_loss = control_flow_ops.with_dependencies([updates], total_loss)`
One can set updates_collections=None to force the updates in place, but that
can have speed penalty, specially in distributed settings.
<h3>Args</h3>


 - **x**: a `Tensor` with 2 or more dimensions, where the first dimension has
`batch_size`. The normalization is over all but the last dimension if
`data_format` is `NHWC` and the second dimension if `data_format` is
`NCHW`.
 - **decay**: decay for the moving average. Reasonable values for `decay` are close
to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
Lower `decay` value (recommend trying `decay`=0.9) if model experiences
reasonably good training performance but poor validation and/or test
performance. Try zero_debias_moving_mean=True for improved stability.
 - **center**: If True, subtract `beta`. If False, `beta` is ignored.
 - **scale**: If True, multiply by `gamma`. If False, `gamma` is
not used. When the next layer is linear (also e.g. `nn.relu`), this can be
disabled since the scaling can be done by the next layer.
 - **epsilon**: small float added to variance to avoid dividing by zero.
 - **activation_fn**: activation function, default set to None to skip it and
maintain a linear activation.
 - **param_initializers**: optional initializers for beta, gamma, moving mean and
moving variance.
 - **updates_collections**: collections to collect the update ops for computation.
The updates_ops need to be executed with the train_op.
If None, a control dependency would be added to make sure the updates are
computed in place.
 - **is_training**: whether or not the layer is in training mode. In training mode
it would accumulate the statistics of the moments into `moving_mean` and
`moving_variance` using an exponential moving average with the given
`decay`. When it is not in training mode then it would use the values of
the `moving_mean` and the `moving_variance`.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
outputs_collections: collections to add the outputs.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
 - **batch_weights**: An optional tensor of shape `[batch_size]`,
containing a frequency weight for each batch item. If present,
then the batch normalization uses weighted mean and
variance. (This can be used to correct for bias in training
example selection.)
 - **fused**:  Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
 - **name**: Optional scope/name for `variable_scope`.

<h3>Returns</h3>


A `Tensor` representing the output of the operation.

 ---------- 

# Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.batch_norm_lasagne</b></span>  (x,  is_training,  reuse,  trainable=True,  decay=0.9,  epsilon=0.0001,  name='bn',  updates_collections='update_ops',  outputs_collections=None)</span>
Instead of storing and updating moving variance, this layer store and
update moving inverse standard deviation
"Batch Normalization: Accelerating Deep Network Training by Reducin Internal Covariate Shift"
Sergey Ioffe, Christian Szegedy
Can be used as a normalizer function for conv2d and fully_connected.
Note: When is_training is True the moving_mean and moving_variance need to be
updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
they need to be added as a dependency to the `train_op`, example:
`update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)`
`if update_ops:`
`updates = tf.group(*update_ops)`
`total_loss = control_flow_ops.with_dependencies([updates], total_loss)`
One can set updates_collections=None to force the updates in place, but that
can have speed penalty, specially in distributed settings.

<h3>Args</h3>


 - **x**: a tensor with 2 or more dimensions, where the first dimension has
`batch_size`. The normalization is over all but the last dimension if
`data_format` is `NHWC` and the second dimension if `data_format` is
`NCHW`.
 - **decay**: decay for the moving average. Reasonable values for `decay` are close
to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
Lower `decay` value (recommend trying `decay`=0.9) if model experiences
reasonably good training performance but poor validation and/or test
performance. Try zero_debias_moving_mean=True for improved stability.
 - **epsilon**: small float added to variance to avoid dividing by zero.
 - **updates_collections**: collections to collect the update ops for computation.
The updates_ops need to be executed with the train_op.
If None, a control dependency would be added to make sure the updates are
computed in place.
 - **is_training**: whether or not the layer is in training mode. In training mode
it would accumulate the statistics of the moments into `moving_mean` and
`moving_variance` using an exponential moving average with the given
`decay`. When it is not in training mode then it would use the values of
the `moving_mean` and the `moving_variance`.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **outputs_collections**: collections to add the outputs.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
 - **name**: Optional scope/name for `variable_scope`.

<h3>Returns</h3>


A `Tensor` representing the output of the operation.

 ---------- 

# Prametric rectifier linear layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.prelu</b></span>  (x,  reuse,  alpha_init=0.2,  trainable=True,  name='prelu',  outputs_collections=None)</span>

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **alpha_init**: initalization value for alpha
 - **trainable**: a bool, training or fixed value
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the prelu activation operation.

 ---------- 

# Rectifier linear layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.relu</b></span>  (x,  name='relu',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the relu activation operation.

 ---------- 

# Rectifier linear relu6 layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.relu6</b></span>  (x,  name='relu6',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the relu6 activation operation.

 ---------- 

# Softpluas layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.softplus</b></span>  (x,  name='softplus',  outputs_collections=None,  **unused)</span>
Computes softplus: log(exp(features) + 1).

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the activation operation.

 ---------- 

# Computes Concatenated ReLU

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.crelu</b></span>  (x,  name='crelu',  outputs_collections=None,  **unused)</span>
Concatenates a ReLU which selects only the positive part of the activation with
a ReLU which selects only the negative part of the activation. Note that
at as a result this non-linearity doubles the depth of the activations.
Source: https://arxiv.org/abs/1603.05201

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the activation operation.

 ---------- 

# Computes exponential linear: exp(features) - 1 if < 0, features otherwise

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.elu</b></span>  (x,  name='elu',  outputs_collections=None,  **unused)</span>
See "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the activation operation.

 ---------- 

# Computes reaky relu

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.leaky_relu</b></span>  (x,  alpha=0.01,  name='leaky_relu',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **aplha**: the conatant fro scalling the activation
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the activation operation.

 ---------- 

# Computes reaky relu lasagne style

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.lrelu</b></span>  (x,  leak=0.2,  name='lrelu',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **leak**: the conatant fro scalling the activation
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the activation operation.

 ---------- 

# Computes maxout activation

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.maxout</b></span>  (x,  k=2,  name='maxout',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **k**: output channel splitting factor
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the activation operation.

 ---------- 

# Computes maxout activation

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.offset_maxout</b></span>  (x,  k=2,  name='maxout',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **k**: output channel splitting factor
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the activation operation.

 ---------- 

# Computes softmax activation

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.softmax</b></span>  (x,  name='softmax',  outputs_collections=None,  **unused)</span>

<h3>Args</h3>


 - **x**: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, int16`, or `int8`.
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the activation operation.

 ---------- 

# Dropout layer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.dropout</b></span>  (x,  is_training,  drop_p=0.5,  name='dropout',  outputs_collections=None,  **unused)</span>
<h3>Args</h3>


 - **x**: a `Tensor`.
 - **is_training**: a bool, training or validation
 - **drop_p**: probability of droping unit
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the dropout operation.

 ---------- 

# Repeat op

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.repeat</b></span>  (x,  repetitions,  layer,  name='repeat',  outputs_collections=None,  *args,  **kwargs)</span>

<h3>Args</h3>


 - **x**: a `Tensor`.
 - **repetitions**: a int, number of times to apply the same operation
 - **layer**: the layer function with arguments to repeat
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the repetition operation.

 ---------- 

# Merge op

<span class="extra_h1"><span style="color:black;"><b>tefla.core.layers.merge</b></span>  (tensors_list,  mode,  axis=1,  name='merge',  outputs_collections=None,  **kwargs)</span>

<h3>Args</h3>


 - **tensor_list**: A list `Tensors` to merge
 - **mode**: str, available modes are
['concat', 'elemwise_sum', 'elemwise_mul', 'sum', 'mean', 'prod', 'max', 'min', 'and', 'or']
 - **name**: a optional scope/name of the layer
 - **outputs_collections**: The collections to which the outputs are added.

<h3>Returns</h3>


A `Tensor` representing the results of the repetition operation.

 ---------- 

