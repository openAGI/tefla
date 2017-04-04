# Device chooser for variables

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L388 target="_blank"><b>tefla.utils.util.VariableDeviceChooser</b></a></span>  (num_parameter_servers=0,  ps_device='/job:ps',  placement='CPU:0')</span>
When using a parameter server it will assign them in a round-robin fashion.
When not using a parameter server it allows GPU:0 placement otherwise CPU:0.
Initialize VariableDeviceChooser.

<h3>Args</h3>


 - **num_parameter_servers**: number of parameter servers.
 - **ps_device**: string representing the parameter server device.
 - **placement**: string representing the placement of the variable either CPU:0
or GPU:0. When using parameter servers forced to CPU:0.

 --------- 

# Valid types for loss, variables and gradients

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L202 target="_blank"><b>tefla.utils.util.valid_dtypes</b></a></span>  ()</span>
Subclasses should override to allow other float types.
<h3>Returns</h3>


Valid types for loss, variables and gradients.

 ---------- 

# Asserts tensors are all valid types (see `_valid_dtypes`)

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L211 target="_blank"><b>tefla.utils.util.assert_valid_dtypes</b></a></span>  (tensors)</span>
<h3>Args</h3>


 - **tensors**: Tensors to check.
<h3>Raises</h3>


 - **ValueError**: If any tensor is not a valid type.

 ---------- 

# Returns value if value_or_tensor_or_var has a constant value

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L226 target="_blank"><b>tefla.utils.util.constant_value</b></a></span>  (value_or_tensor_or_var,  dtype=None)</span>

<h3>Args</h3>


 - **value_or_tensor_or_var**: A value, a `Tensor` or a `Variable`.
 - **dtype**: Optional `tf.dtype`, if set it would check it has the right
 -   dtype.

<h3>Returns</h3>


The constant value or None if it not constant.

 ---------- 

# Return either fn1() or fn2() based on the boolean value of `pred`

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L255 target="_blank"><b>tefla.utils.util.static_cond</b></a></span>  (pred,  fn1,  fn2)</span>

Same signature as `control_flow_ops.cond()` but requires pred to be a bool.

<h3>Args</h3>


 - **pred**: A value determining whether to return the result of `fn1` or `fn2`.
 - **fn1**: The callable to be performed if pred is true.
 - **fn2**: The callable to be performed if pred is false.

<h3>Returns</h3>


Tensors returned by the call to either `fn1` or `fn2`.

 ---------- 

# Return either fn1() or fn2() based on the boolean predicate/value `pred`

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L281 target="_blank"><b>tefla.utils.util.smart_cond</b></a></span>  (pred,  fn1,  fn2,  name=None)</span>

If `pred` is bool or has a constant value it would use `static_cond`,
 otherwise it would use `tf.cond`.

<h3>Args</h3>


 - **pred**: A scalar determining whether to return the result of `fn1` or `fn2`.
 - **fn1**: The callable to be performed if pred is true.
 - **fn2**: The callable to be performed if pred is false.
 - **name**: Optional name prefix when using tf.cond
<h3>Returns</h3>


 - Tensors returned by the call to either `fn1` or `fn2`.

<h3>Returns</h3>


Tensors returned by the call to either `fn1` or `fn2`.

 ---------- 

# Transform numeric labels into onehot_labels

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L333 target="_blank"><b>tefla.utils.util.one_hot</b></a></span>  (labels,  num_classes,  name='one_hot')</span>
<h3>Args</h3>


 - **labels**: [batch_size] target labels.
 - **num_classes**: total number of classes.
 - **scope**: Optional scope for op_scope.
<h3>Returns</h3>


 - one hot encoding of the labels.

<h3>Returns</h3>


one hot encoding of the labels.

 ---------- 

# Returns a true if its input is a collections.Sequence (except strings)

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L353 target="_blank"><b>tefla.utils.util.is_sequence</b></a></span>  (seq)</span>

<h3>Args</h3>


 - **seq**: an input sequence.

<h3>Returns</h3>


True if the sequence is a not a string and is a collections.Sequence.

 ---------- 

# Returns a flat sequence from a given nested structure

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L365 target="_blank"><b>tefla.utils.util.flatten_sq</b></a></span>  (nest_sq)</span>
If `nest` is not a sequence, this returns a single-element list: `[nest]`.

<h3>Args</h3>


 - **nest**: an arbitrarily nested structure or a scalar object.
Note, numpy arrays are considered scalars.

<h3>Returns</h3>


A Python list, the flattened version of the input.

 ---------- 

# Returns the last dimension of shape while checking it has min_rank

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L417 target="_blank"><b>tefla.utils.util.last_dimension</b></a></span>  (shape,  min_rank=1)</span>

<h3>Args</h3>


 - **shape**: A `TensorShape`.
 - **min_rank**: Integer, minimum rank of shape.

<h3>Returns</h3>


The value of the last dimension.

 ---------- 

# Load Graph from frozen weights and model

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L443 target="_blank"><b>tefla.utils.util.load_frozen_graph</b></a></span>  (frozen_graph)</span>

<h3>Args</h3>


 - **frozen_graph**: binary pb file

<h3>Returns</h3>


loaded graph

 ---------- 

# Normalize a input layer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L471 target="_blank"><b>tefla.utils.util.normalize</b></a></span>  (input_layer)</span>

<h3>Args</h3>


 - **inmput_layer**: input layer tp normalize

<h3>Returns</h3>


normalized layer

 ---------- 

# DeNormalize a input layer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L483 target="_blank"><b>tefla.utils.util.denormalize</b></a></span>  (input_layer)</span>

<h3>Args</h3>


 - **input_layer**: input layer to de normalize

<h3>Returns</h3>


denormalized layer

 ---------- 

