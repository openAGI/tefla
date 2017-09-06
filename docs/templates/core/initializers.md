# He Normal initializer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/initializers.py#L11 target="_blank"><b>tefla.core.initializers.he_normal</b></a></span>  (seed=None,  scale=1.0,  dtype=tf.float32)</span>
Kaiming He et al. (2015): Delving deep into rectifiers: Surpassing human-level
performance on imagenet classification. arXiv preprint arXiv:1502.01852.

<h3>Args</h3>


 - **scale**: float
   Scaling factor for the weights. Set this to ``1.0`` for linear and
   sigmoid units, to ``sqrt(2)`` for rectified linear units, and
   to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
   leakiness ``alpha``. Other transfer functions may need different factors.

 ---------- 

# He Uniform initializer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/initializers.py#L28 target="_blank"><b>tefla.core.initializers.he_uniform</b></a></span>  (seed=None,  scale=1.0,  dtype=tf.float32)</span>

<h3>Args</h3>


 - **scale**: float
   Scaling factor for the weights. Set this to ``1.0`` for linear and
   sigmoid units, to ``sqrt(2)`` for rectified linear units, and
   to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
   leakiness ``alpha``. Other transfer functions may need different factors.

 ---------- 

# Random Normal initializer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/initializers.py#L43 target="_blank"><b>tefla.core.initializers.random_normal</b></a></span>  (seed=None,  mean=0.0,  stddev=1.0,  dtype=tf.float32,  name=None)</span>

<h3>Args</h3>


 - **mean**: a `float`
 - **stddev**: a `float`

 ---------- 

# Returns an initializer that generates tensors without scaling variance

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/initializers.py#L71 target="_blank"><b>tefla.core.initializers.variance_scaling_initializer_v2</b></a></span>  (factor=2.0,  mode='FAN_IN',  uniform=False,  seed=None,  dtype=tf.float32,  mean=0.0,  stddev=1.0,  normal_type=None,  name=None)</span>
When initializing a deep network, it is in principle advantageous to keep
the scale of the input variance constant, so it does not explode or diminish
by reaching the final layer. This initializer use the following formula:
```python
  if mode='FAN_IN': # Count only number of input connections.
n = fan_in
  elif mode='FAN_OUT': # Count only number of output connections.
n = fan_out
  elif mode='FAN_AVG': # Average number of inputs and output connections.
n = (fan_in + fan_out)/2.0
truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
```
* To get [Delving Deep into Rectifiers](
   http://arxiv.org/pdf/1502.01852v1.pdf), use (Default):<br/>
  `factor=2.0 mode='FAN_IN' uniform=False`
* To get [Convolutional Architecture for Fast Feature Embedding](
   http://arxiv.org/abs/1408.5093), use:<br/>
  `factor=1.0 mode='FAN_IN' uniform=True`
* To get [Understanding the difficulty of training deep feedforward neural
  networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
  use:<br/>
  `factor=1.0 mode='FAN_AVG' uniform=True.`
* To get `xavier_initializer` use either:<br/>
  `factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
  `factor=1.0 mode='FAN_AVG' uniform=False`.
<h3>Args</h3>


  factor: Float.  A multiplicative factor.
  mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
  uniform: Whether to use uniform or normal distributed random initialization.
  seed: A Python integer. Used to create random seeds. See
 - [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
 - for behavior.
  dtype: The data type. Only floating point types are supported.
<h3>Returns</h3>


  An initializer that generates tensors with unit variance.
<h3>Raises</h3>


  ValueError: if `dtype` is not a floating point type.
  TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].

<h3>Returns</h3>


  An initializer that generates tensors with unit variance.
<h3>Raises</h3>


  ValueError: if `dtype` is not a floating point type.
  TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].

 ---------- 

# Bilinear initialization for up sampling operation

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/initializers.py#L166 target="_blank"><b>tefla.core.initializers.bilinear</b></a></span>  (f_shape)</span>

<h3>Args</h3>


 - **f_shape**: shape of the variable

<h3>Returns</h3>


bilinear initializer

 ---------- 

# Variable initializer that produces a random orthonormal matrix

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/initializers.py#L194 target="_blank"><b>tefla.core.initializers.random_orthonormal_initializer</b></a></span>  (shape,  dtype=tf.float32,  partition_info=None)</span>

<h3>Args</h3>


 - **shape**: shape of the variable

<h3>Returns</h3>


random_orthogonal_matrix for initialization.

 ---------- 

