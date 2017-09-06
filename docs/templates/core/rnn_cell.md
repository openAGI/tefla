# The most basic RNN cell

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L15 target="_blank"><b>tefla.core.rnn_cell.BasicRNNCell</b></a></span>  (num_units,  reuse,  trainable=True,  w_init=<function  _initializer  at  0x7fc13b6c6b18>,  use_bias=False,  input_size=None,  activation=<function  tanh  at  0x7fc1916f80c8>,  layer_norm=None,  layer_norm_args=None)</span>

<h3>Args</h3>


 - **num_units**: int, The number of units in the LSTM cell.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **input_size**: Deprecated and unused.
 - **activation**: Activation function of the states.
 - **layer_norm**: If `True`, layer normalization will be applied.
 - **layer_norm_args**: optional dict, layer_norm arguments
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/#L344 target="_blank"><b>add_variable</b></a></span>  (name,  shape,  dtype=None,  initializer=None,  regularizer=None,  trainable=True)</span>

Arguments:
  name: variable name.
  shape: variable shape.
  dtype: The type of the variable. Defaults to `self.dtype`.
  initializer: initializer instance (callable).
  regularizer: regularizer instance (callable).
  trainable: whether the variable should be part of the layer's
"trainable_variables" (e.g. variables, biases)
or "non_trainable_variables" (e.g. BatchNorm mean, stddev).

<h5>Returns</h5>


  The created variable.

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/#L479 target="_blank"><b>apply</b></a></span>  (inputs,  *args,  **kwargs)</span>

This simply wraps `self.__call__`.

Arguments:
  inputs: Input tensor(s).
  *args: additional positional arguments to be passed to `self.call`.
  **kwargs: additional keyword arguments to be passed to `self.call`.

<h5>Returns</h5>


  Output tensor(s).

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/#L299 target="_blank"><b>call</b></a></span>  (inputs,  **kwargs)</span>

Arguments:
  inputs: input tensor(s).
 **kwargs: additional keyword arguments.

<h5>Returns</h5>


  Output tensor(s).

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/#L271 target="_blank"><b>get_losses_for</b></a></span>  (inputs)</span>

Arguments:
  inputs: Input tensor or list/tuple of input tensors.
Must match the `inputs` argument passed to the `__call__`
method at the time the losses were created.
If you pass `inputs=None`, unconditional losses are returned,
such as weight regularization losses.

<h5>Returns</h5>


  List of loss tensors of the layer that depend on `inputs`.

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/#L207 target="_blank"><b>get_updates_for</b></a></span>  (inputs)</span>

Arguments:
  inputs: Input tensor or list/tuple of input tensors.
Must match the `inputs` argument passed to the `__call__` method
at the time the updates were created.
If you pass `inputs=None`, unconditional updates are returned.

<h5>Returns</h5>


  List of update ops of the layer that depend on `inputs`.

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/#L212 target="_blank"><b>zero_state</b></a></span>  (batch_size,  dtype)</span>

<h5>Args</h5>


  batch_size: int, float, or unit Tensor representing the batch size.
  dtype: the data type to use for the state.

<h5>Returns</h5>


  If `state_size` is an int or TensorShape, then the return value is a
  `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

 --------- 

# LSTM unit

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L61 target="_blank"><b>tefla.core.rnn_cell.LSTMCell</b></a></span>  (num_units,  reuse,  trainable=True,  w_init=<function  _initializer  at  0x7fc13b6c6d70>,  forget_bias=1.0,  use_bias=False,  input_size=None,  activation=<function  tanh  at  0x7fc1916f80c8>,  inner_activation=<function  sigmoid  at  0x7fc1916f2f50>,  keep_prob=1.0,  dropout_seed=None,  cell_clip=None,  layer_norm=None,  layer_norm_args=None)</span>

This class adds layer normalization and recurrent dropout to a
basic LSTM unit. Layer normalization implementation is based on:
https://arxiv.org/abs/1607.06450.
"Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
and is applied before the internal nonlinearities.
Recurrent dropout is base on:
https://arxiv.org/abs/1603.05118
"Recurrent Dropout without Memory Loss"
Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.

<h3>Args</h3>


 - **num_units**: int, The number of units in the LSTM cell.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **forget_bias**: float, The bias added to forget gates (see above).
 - **input_size**: Deprecated and unused.
 - **activation**: Activation function of the states.
 - **inner_activation**: Activation function of the inner states.
 - **layer_norm**: If `True`, layer normalization will be applied.
 - **layer_norm_args**: optional dict, layer_norm arguments
 - **cell_clip**: (optional) A float value, if provided the cell state is clipped
by this value prior to the cell output activation.
 - **keep_prob**: unit Tensor or float between 0 and 1 representing the
recurrent dropout probability value. If float and 1.0, no dropout will
be applied.
 - **dropout_seed**: (optional) integer, the randomness seed.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).

 --------- 

# Basic attention cell

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L164 target="_blank"><b>tefla.core.rnn_cell.AttentionCell</b></a></span>  (cell,  attn_length,  reuse,  w_init=<function  _initializer  at  0x7fc13b6d5050>,  use_bias=False,  trainable=True,  attn_size=None,  attn_vec_size=None,  input_size=None,  layer_norm=None,  layer_norm_args=None)</span>

Implementation based on https://arxiv.org/abs/1409.0473.
Create a cell with attention.

<h3>Args</h3>


 - **cell**: an RNNCell, an attention is added to it.
e.g.: a LSTMCell
 - **attn_length**: integer, the size of an attention window.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **attn_size**: integer, the size of an attention vector. Equal to
cell.output_size by default.
 - **attn_vec_size**: integer, the number of convolutional features calculated
on attention state and a size of the hidden layer built from
base cell state. Equal attn_size to by default.
 - **input_size**: integer, the size of a hidden linear layer,
 - **layer_norm**: If `True`, layer normalization will be applied.
 - **layer_norm_args**: optional dict, layer_norm arguments
built from inputs and attention. Derived from the input tensor by default.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).

 --------- 

# Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L257 target="_blank"><b>tefla.core.rnn_cell.GRUCell</b></a></span>  (num_units,  reuse,  w_init=<function  _initializer  at  0x7fc13b6d5320>,  use_bias=False,  trainable=True,  input_size=None,  activation=<function  tanh  at  0x7fc1916f80c8>,  inner_activation=<function  sigmoid  at  0x7fc1916f2f50>,  b_init=1.0,  layer_norm=None,  layer_norm_args=None)</span>

<h3>Args</h3>


 - **num_units**: int, The number of units in the LSTM cell.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **input_size**: Deprecated and unused.
 - **activation**: Activation function of the states.
 - **inner_activation**: Activation function of the inner states.
 - **layer_norm**: If `True`, layer normalization will be applied.
 - **layer_norm_args**: optional dict, layer_norm arguments
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).

 --------- 

# RNN cell composed sequentially of multiple simple cells

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L321 target="_blank"><b>tefla.core.rnn_cell.MultiRNNCell</b></a></span>  (cells,  state_is_tuple=True)</span>

Create a RNN cell composed sequentially of a number of RNNCells.
<h3>Args</h3>


 - **cells**: list of RNNCells that will be composed in this order.

 --------- 

# Operator adding dropout to inputs and outputs of the given cell

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L620 target="_blank"><b>tefla.core.rnn_cell.DropoutWrapper</b></a></span>  (cell,  is_training,  input_keep_prob=1.0,  output_keep_prob=1.0,  seed=None)</span>

Create a cell with added input and/or output dropout.
Dropout is never used on the state.

<h3>Args</h3>


 - **cell**: an RNNCell, a projection to output_size is added to it.
 - **is_training**: a bool, training if true else validation/testing
 - **input_keep_prob**: unit Tensor or float between 0 and 1, input keep
probability; if it is float and 1, no input dropout will be added.
 - **output_keep_prob**: unit Tensor or float between 0 and 1, output keep
probability; if it is float and 1, no output dropout will be added.
 - **seed**: (optional) integer, the randomness seed.

 --------- 

# Adds a fully connected layer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L673 target="_blank"><b>tefla.core.rnn_cell._linear</b></a></span>  (x,  n_output,  reuse,  trainable=True,  w_init=<function  _initializer  at  0x7fc13b6c6758>,  b_init=0.0,  w_regularizer=<function  l2_loss  at  0x7fc1918327d0>,  name='fc',  layer_norm=None,  layer_norm_args=None,  activation=None,  outputs_collections=None,  use_bias=True)</span>

`fully_connected` creates a variable called `weights`, representing a fully
connected weight matrix, which is multiplied by the `x` to produce a
`Tensor` of hidden units. If a `layer_norm` is provided (such as
`layer_norm`), it is then applied. Otherwise, if `layer_norm` is
None and a `b_init` and `use_bias` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation` is not `None`,
it is applied to the hidden units as well.
Note: that if `x` have a rank greater than 2, then `x` is flattened
prior to the initial matrix multiply by `weights`.

<h3>Args</h3>


 - **x**: A `Tensor` of with at least rank 2 and value for the last dimension,
i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
 - **n_output**: Integer or long, the number of output units in the layer.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **activation**: activation function, set to None to skip it and maintain
a linear activation.
 - **layer_norm**: normalization function to use. If
 -`batch_norm` is `True` then google original implementation is used and
if another function is provided then it is applied.
default set to None for no normalizer function
 - **layer_norm_args**: normalization function parameters.
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

# Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L771 target="_blank"><b>tefla.core.rnn_cell.layer_norm</b></a></span>  (x,  reuse,  center=True,  scale=True,  trainable=True,  epsilon=1e-12,  name='bn',  outputs_collections=None)</span>
"Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
Can be used as a normalizer function for conv2d and fully_connected.

<h3>Args</h3>


 - **x**: a tensor with 2 or more dimensions, where the first dimension has
`batch_size`. The normalization is over all but the last dimension if
`data_format` is `NHWC` and the second dimension if `data_format` is
`NCHW`.
 - **center**: If True, subtract `beta`. If False, `beta` is ignored.
 - **scale**: If True, multiply by `gamma`. If False, `gamma` is
not used. When the next layer is linear (also e.g. `nn.relu`), this can be
disabled since the scaling can be done by the next layer.
 - **epsilon**: small float added to variance to avoid dividing by zero.
 - **reuse**: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
 - **outputs_collections**: collections to add the outputs.
 - **trainable**: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
 - **name**: Optional scope/name for `variable_scope`.

<h3>Returns</h3>


A `Tensor` representing the output of the operation.

 ---------- 

# LSTM

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L908 target="_blank"><b>tefla.core.rnn_cell.lstm</b></a></span>  (inputs,  n_units,  reuse,  is_training,  activation=<function  tanh  at  0x7fc1916f80c8>,  inner_activation=<function  sigmoid  at  0x7fc1916f2f50>,  dropout=None,  use_bias=True,  w_init=<function  _initializer  at  0x7fc13b6d7140>,  forget_bias=1.0,  return_seq=False,  return_state=False,  initial_state=None,  dynamic=True,  trainable=True,  scope='lstm')</span>
Long Short Term Memory Recurrent Layer.

<h3>Args</h3>


 - **inputs**: `Tensor`. Inputs 3-D Tensor [samples, timesteps, input_dim].
 - **n_units**: `int`, number of units for this layer.
 - **reuse**: `bool`. If True and 'scope' is provided, this layer variables
will be reused (shared).
 - **is_training**: `bool`, training if True
 - **activation**: `function` (returning a `Tensor`).
 - **inner_activation**: `function` (returning a `Tensor`).
 - **dropout**: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
input and output keep probability.
 - **use_bias**: `bool`. If True, a bias is used.
 - **w_init**: `function` (returning a `Tensor`). Weights initialization.
 - **forget_bias**: `float`. Bias of the forget gate. Default: 1.0.
 - **return_seq**: `bool`. If True, returns the full sequence instead of
last sequence output only.
 - **return_state**: `bool`. If True, returns a tuple with output and
states: (output, states).
 - **initial_state**: `Tensor`. An initial state for the RNN.  This must be
a tensor of appropriate type and shape [batch_size x cell.state_size].
 - **dynamic**: `bool`. If True, dynamic computation is performed. It will not
compute RNN steps above the sequence length. Note that because TF
requires to feed sequences of same length, 0 is used as a mask.
So a sequence padded with 0 at the end must be provided. When
computation is performed, it will stop when it meets a step with
a value of 0.
 - **trainable**: `bool`. If True, weights will be trainable.
 - **scope**: `str`. Define this layer scope (optional). A scope can be
used to share variables between layers. Note that scope will
override name.

<h3>Returns</h3>


if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
else: 2-D Tensor [samples, output dim].

 ---------- 

# GRU

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L961 target="_blank"><b>tefla.core.rnn_cell.gru</b></a></span>  (inputs,  n_units,  reuse,  is_training,  activation=<function  tanh  at  0x7fc1916f80c8>,  inner_activation=<function  sigmoid  at  0x7fc1916f2f50>,  dropout=None,  use_bias=True,  w_init=<function  _initializer  at  0x7fc13b6d7230>,  forget_bias=1.0,  return_seq=False,  return_state=False,  initial_state=None,  dynamic=True,  trainable=True,  scope='gru')</span>
Gated Recurrent Layer.

<h3>Args</h3>


 - **inputs**: `Tensor`. Inputs 3-D Tensor [samples, timesteps, input_dim].
 - **n_units**: `int`, number of units for this layer.
 - **reuse**: `bool`. If True and 'scope' is provided, this layer variables
will be reused (shared).
 - **is_training**: `bool`, training if True
 - **activation**: `function` (returning a `Tensor`).
 - **inner_activation**: `function` (returning a `Tensor`).
 - **dropout**: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
input and output keep probability.
 - **use_bias**: `bool`. If True, a bias is used.
 - **w_init**: `function` (returning a `Tensor`). Weights initialization.
 - **forget_bias**: `float`. Bias of the forget gate. Default: 1.0.
 - **return_seq**: `bool`. If True, returns the full sequence instead of
last sequence output only.
 - **return_state**: `bool`. If True, returns a tuple with output and
states: (output, states).
 - **initial_state**: `Tensor`. An initial state for the RNN.  This must be
a tensor of appropriate type and shape [batch_size x cell.state_size].
 - **dynamic**: `bool`. If True, dynamic computation is performed. It will not
compute RNN steps above the sequence length. Note that because TF
requires to feed sequences of same length, 0 is used as a mask.
So a sequence padded with 0 at the end must be provided. When
computation is performed, it will stop when it meets a step with
a value of 0.
 - **trainable**: `bool`. If True, weights will be trainable.
 - **scope**: `str`. Define this layer scope (optional). A scope can be
used to share variables between layers. Note that scope will
override name.

<h3>Returns</h3>


if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
else: 2-D Tensor [samples, output dim].

 ---------- 

# Bidirectional RNN

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/rnn_cell.py#L1014 target="_blank"><b>tefla.core.rnn_cell.bidirectional_rnn</b></a></span>  (inputs,  rnncell_fw,  rnncell_bw,  reuse,  is_training,  dropout_fw=None,  dropout_bw=None,  return_seq=False,  return_states=False,  initial_state_fw=None,  initial_state_bw=None,  dynamic=False,  scope='BiRNN',  outputs_collections=None)</span>
Build a bidirectional recurrent neural network, it requires 2 RNN Cells
to process sequence in forward and backward order. Any RNN Cell can be
used i.e. SimpleRNN, LSTM, GRU... with its own parameters. But the two
cells number of units must match.

<h3>Args</h3>


 - **inputs**: `Tensor`. The 3D inputs Tensor [samples, timesteps, input_dim].
 - **rnncell_fw**: `RNNCell`. The RNN Cell to use for foward computation.
 - **rnncell_bw**: `RNNCell`. The RNN Cell to use for backward computation.
 - **reuse**: `bool`. If True and 'scope' is provided, this layer variables
will be reused (shared).
 - **is_training**: `bool`, training if True
 - **dropout_fw**: `tuple` of `float`: (input_keep_prob, output_keep_prob). the
input and output keep probability.
 - **dropout_bw**: `tuple` of `float`: (input_keep_prob, output_keep_prob). the
input and output keep probability.
 - **return_seq**: `bool`. If True, returns the full sequence instead of
last sequence output only.
 - **return_states**: `bool`. If True, returns a tuple with output and
states: (output, states).
 - **initial_state_fw**: `Tensor`. An initial state for the forward RNN.
This must be a tensor of appropriate type and shape [batch_size
x cell.state_size].
 - **initial_state_bw**: `Tensor`. An initial state for the backward RNN.
This must be a tensor of appropriate type and shape [batch_size
x cell.state_size].
 - **dynamic**: `bool`. If True, dynamic computation is performed. It will not
compute RNN steps above the sequence length. Note that because TF
requires to feed sequences of same length, 0 is used as a mask.
So a sequence padded with 0 at the end must be provided. When
computation is performed, it will stop when it meets a step with
a value of 0.
 - **scope**: `str`. Define this layer scope (optional). A scope can be
used to share variables between layers. Note that scope will
override name.

<h3>Returns</h3>


if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
else: 2-D Tensor Layer [samples, output dim].

 ---------- 

