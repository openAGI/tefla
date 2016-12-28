# Define a log loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L11 target="_blank"><b>tefla.core.losses.log_loss_custom</b></a></span>  (predictions,  labels,  eps=1e-07,  name='log')</span>
<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D or array tensor, [batch_size, num_classes]  ground truth labels or target labels.
 - **eps**: a constant to set upper or lower limit for labels, smoothening factor
 - **name**: Optional scope/name for op_scope.
<h3>Returns</h3>


 - A tensor with the log loss.

<h3>Returns</h3>


A tensor with the log loss.

 ---------- 

# Define a kappa loss, Its a continuous differentiable approximation of discrete kappa loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L30 target="_blank"><b>tefla.core.losses.kappa_loss</b></a></span>  (predictions,  labels,  y_pow=1,  eps=1e-15,  num_ratings=5,  batch_size=32,  name='kappa')</span>
<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
 - **y_pow**: int, to whcih the labels should be raised; useful if model diverge. e.g. y_pow=2
 - **num_ratings**: numbers of rater to used, typically num_classes of the model
 - **batch_size**: batch_size of the training or validation ops
 - **eps**: a float, prevents divide by zero 
 - **name**: Optional scope/name for op_scope.
<h3>Returns</h3>


 - A tensor with the kappa loss.

<h3>Returns</h3>


A tensor with the kappa loss.

 ---------- 

# Define a joint kappa and log loss, Kappa is a continuous differentiable approximation of discrete kappa loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L69 target="_blank"><b>tefla.core.losses.kappa_log_loss</b></a></span>  (predictions,  labels,  label_smoothing=0.0,  y_pow=1,  batch_size=32,  log_scale=0.5,  log_offset=0.5,  name='kappa_log')</span>
<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
 - **label_smoothing**: a float, used to smooth the labels for better generalization if greater than 0 then smooth the labels.
 - **y_pow**: int, to whcih the labels should be raised; useful if model diverge. e.g. y_pow=2
 - **num_ratings**: numbers of rater to used, typically num_classes of the model
 - **batch_size**: batch_size of the training or validation ops
 - **log_scale**: a float, used to multiply the clipped log loss, e.g: 0.5
 - **log_offset**:a float minimum log loss offset to substract from original log loss; e.g. 0.50
 - **name**: Optional scope/name for op_scope.
<h3>Returns</h3>


 - A tensor with the kappa log loss.

<h3>Returns</h3>


A tensor with the kappa log loss.

 ---------- 

# Define a joint kappa and log loss; log loss is clipped by a defined min value; Kappa is a continuous differentiable approximation of discrete kappa loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L97 target="_blank"><b>tefla.core.losses.kappa_log_loss_clipped</b></a></span>  (predictions,  labels,  label_smoothing=0.0,  y_pow=1,  batch_size=32,  log_scale=0.5,  log_cutoff=0.8,  name='kappa_log_clipped')</span>
<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
 - **label_smoothing**: a float, used to smooth the labels for better generalization if greater than 0 then smooth the labels.
 - **y_pow**: int, to whcih the labels should be raised; useful if model diverge. e.g. y_pow=2
 - **num_ratings**: numbers of rater to used, typically num_classes of the model
 - **batch_size**: batch_size of the training or validation ops
 - **log_scale**: a float, used to multiply the clipped log loss, e.g: 0.5
 - **log_cutoff**:a float, minimum log loss value; e.g. 0.50
 - **name**: Optional scope/name for op_scope.
<h3>Returns</h3>


 - A tensor with the clipped kappa log loss.

<h3>Returns</h3>


A tensor with the clipped kappa log loss.

 ---------- 

# Define a cross entropy loss with label smoothing

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L125 target="_blank"><b>tefla.core.losses.cross_entropy_loss</b></a></span>  (logits,  labels,  label_smoothing=0.0,  weight=1.0,  name='cross_entropy_loss')</span>
<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
 - **label_smoothing**: a float, used to smooth the labels for better generalizationif greater than 0 then smooth the labels.
 - **weight**: scale the loss by this factor.
 - **name**: Optional scope/name for op_scope.
<h3>Returns</h3>


 - A tensor with the cross entropy loss.

<h3>Returns</h3>


A tensor with the cross entropy loss.

 ---------- 

# Define a L2Loss, useful for regularize, i.e. weight decay

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L151 target="_blank"><b>tefla.core.losses.l1_l2_regularizer</b></a></span>  (var,  weight_l1=1.0,  weight_l2=1.0,  name='l1_l2_regularizer')</span>
<h3>Args</h3>


 - **var**: tensor to regularize.
 - **weight_l1**: an optional weight to modulate the l1 loss.
 - **weight_l2**: an optional weight to modulate the l2 loss.
 - **name**: Optional scope/name for op_scope.
<h3>Returns</h3>


 - the l1+L2 loss op.

<h3>Returns</h3>


the l1+L2 loss op.

 ---------- 

