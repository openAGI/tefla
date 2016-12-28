# Valid types for loss, variables and gradients

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L194 target="_blank"><b>tefla.utils.util.valid_dtypes</b></a></span>  ()</span>
Subclasses should override to allow other float types.
<h3>Returns</h3>


Valid types for loss, variables and gradients.

 ---------- 

# Asserts tensors are all valid types (see `_valid_dtypes`)

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L203 target="_blank"><b>tefla.utils.util.assert_valid_dtypes</b></a></span>  (tensors)</span>
<h3>Args</h3>


 - **tensors**: Tensors to check.
<h3>Raises</h3>


 - **ValueError**: If any tensor is not a valid type.

 ---------- 

# Returns value if value_or_tensor_or_var has a constant value

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L217 target="_blank"><b>tefla.utils.util.constant_value</b></a></span>  (value_or_tensor_or_var,  dtype=None)</span>

<h3>Args</h3>


 - **value_or_tensor_or_var**: A value, a `Tensor` or a `Variable`.
 - **dtype**: Optional `tf.dtype`, if set it would check it has the right
 -   dtype.

<h3>Returns</h3>


The constant value or None if it not constant.

 ---------- 

# Return either fn1() or fn2() based on the boolean value of `pred`

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L245 target="_blank"><b>tefla.utils.util.static_cond</b></a></span>  (pred,  fn1,  fn2)</span>

Same signature as `control_flow_ops.cond()` but requires pred to be a bool.

<h3>Args</h3>


 - **pred**: A value determining whether to return the result of `fn1` or `fn2`.
 - **fn1**: The callable to be performed if pred is true.
 - **fn2**: The callable to be performed if pred is false.

<h3>Returns</h3>


Tensors returned by the call to either `fn1` or `fn2`.

 ---------- 

# Return either fn1() or fn2() based on the boolean predicate/value `pred`

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L271 target="_blank"><b>tefla.utils.util.smart_cond</b></a></span>  (pred,  fn1,  fn2,  name=None)</span>

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

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/utils/util.py#L321 target="_blank"><b>tefla.utils.util.one_hot_encoding</b></a></span>  (labels,  num_classes,  name='one_hot_encoding')</span>
<h3>Args</h3>


 - **labels**: [batch_size] target labels.
 - **num_classes**: total number of classes.
 - **scope**: Optional scope for op_scope.
<h3>Returns</h3>


 - one hot encoding of the labels.

<h3>Returns</h3>


one hot encoding of the labels.

 ---------- 

