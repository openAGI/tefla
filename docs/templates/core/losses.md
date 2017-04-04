# Define a log loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L11 target="_blank"><b>tefla.core.losses.log_loss_custom</b></a></span>  (predictions,  labels,  eps=1e-07,  name='log')</span>

<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D or array tensor, [batch_size, num_classes]  ground truth labels or target labels.
 - **eps**: a constant to set upper or lower limit for labels, smoothening factor
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


A tensor with the log loss.

 ---------- 

# Define a kappa loss, Its a continuous differentiable approximation of discrete kappa loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L42 target="_blank"><b>tefla.core.losses.kappa_loss</b></a></span>  (predictions,  labels,  y_pow=1,  eps=1e-15,  num_ratings=5,  batch_size=32,  name='kappa')</span>

<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
 - **y_pow**: int, to whcih the labels should be raised; useful if model diverge. e.g. y_pow=2
 - **num_ratings**: numbers of rater to used, typically num_classes of the model
 - **batch_size**: batch_size of the training or validation ops
 - **eps**: a float, prevents divide by zero
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


A tensor with the kappa loss.

 ---------- 

# Define a joint kappa and log loss, Kappa is a continuous differentiable approximation of discrete kappa loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L87 target="_blank"><b>tefla.core.losses.kappa_log_loss</b></a></span>  (predictions,  labels,  label_smoothing=0.0,  y_pow=1,  batch_size=32,  log_scale=0.5,  log_offset=0.5,  name='kappa_log')</span>

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


A tensor with the kappa log loss.

 ---------- 

# Define a joint kappa and log loss; log loss is clipped by a defined min value; Kappa is a continuous differentiable approximation of discrete kappa loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L118 target="_blank"><b>tefla.core.losses.kappa_log_loss_clipped</b></a></span>  (predictions,  labels,  label_smoothing=0.0,  y_pow=1,  batch_size=32,  log_scale=0.5,  log_cutoff=0.8,  num_classes=5,  name='kappa_log_clipped')</span>

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


A tensor with the clipped kappa log loss.

 ---------- 

# Define a cross entropy loss with label smoothing

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L149 target="_blank"><b>tefla.core.losses.cross_entropy_loss</b></a></span>  (logits,  labels,  label_smoothing=0.0,  weight=1.0,  name='cross_entropy_loss')</span>

<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D tensor or array,[batch_size, num_classes]  ground truth labels or target labels.
 - **label_smoothing**: a float, used to smooth the labels for better generalizationif greater than 0 then smooth the labels.
 - **weight**: scale the loss by this factor.
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


A tensor with the cross entropy loss.

 ---------- 

# Define a L2Loss, useful for regularize, i.e. weight decay

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L179 target="_blank"><b>tefla.core.losses.l1_l2_regularizer</b></a></span>  (var,  weight_l1=1.0,  weight_l2=1.0,  name='l1_l2_regularizer')</span>

<h3>Args</h3>


 - **var**: tensor to regularize.
 - **weight_l1**: an optional weight to modulate the l1 loss.
 - **weight_l2**: an optional weight to modulate the l2 loss.
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


the l1+L2 loss op.

 ---------- 

# log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L202 target="_blank"><b>tefla.core.losses.discretized_mix_logistic_loss</b></a></span>  (inputs,  predictions,  sum_all=True,  name='disretized_mix_logistic_loss')</span>

<h3>Args</h3>


 - **predictions**: 4D tensor or array, [batch_size, width, height, out_channels] predictions of the network .
 - **inputs**: 4D tensor or array, [batch_size, width, height, num_classes] ground truth labels or target labels.
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


A tensor with the discretized mix logistic loss.

 ---------- 

# Pull Away loss calculation

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L264 target="_blank"><b>tefla.core.losses.pullaway_loss</b></a></span>  (embeddings,  name='pullaway_loss')</span>

<h3>Args</h3>


 - **embeddings**: The embeddings to be orthogonalized for varied faces. Shape [batch_size, embeddings_dim]

 ---------- 

# Calculate the loss from the logits and the labels

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L298 target="_blank"><b>tefla.core.losses.segment_loss</b></a></span>  (logits,  labels,  num_classes,  head=None)</span>
<h3>Args</h3>


  logits: tensor, float - [batch_size * width * height, num_classes].
 -   Use vgg_fcn.up as logits.
  labels: Labels tensor, int32 - [batch_size * width * height, num_classes].
 -   The ground truth of your data.
  head: numpy array - [num_classes]
 -   Weighting the loss of each class
 -   Optional: Prioritize some classes
<h3>Returns</h3>


  loss: Loss tensor of type float.

<h3>Returns</h3>


  loss: Loss tensor of type float.

 ---------- 

# Calculate the triplet loss according to the FaceNet paper

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L331 target="_blank"><b>tefla.core.losses.triplet_loss</b></a></span>  (anchor,  positive,  negative,  alpha=0.2,  name='triplet_loss')</span>

<h3>Args</h3>


  anchor: 2-D `tensor` [batch_size, embedding_size], the embeddings for the anchor images.
  positive: 2-D `tensor` [batch_size, embedding_size], the embeddings for the positive images.
  negative: 2-D `tensor` [batch_size, embedding_size], the embeddings for the negative images.
  alpha: positive to negative triplet distance margin

<h3>Returns</h3>


  the triplet loss.

 ---------- 

# Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L351 target="_blank"><b>tefla.core.losses.decov_loss</b></a></span>  (xs,  name='decov_loss')</span>
'Reducing Overfitting In Deep Networks by Decorrelating Representation'

<h3>Args</h3>


 - **xs**: 4-D `tensor` [batch_size, height, width, channels], input

<h3>Returns</h3>


a `float` decov loss

 ---------- 

# Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L372 target="_blank"><b>tefla.core.losses.center_loss</b></a></span>  (features,  label,  alpha,  num_classes,  name='center_loss')</span>
   (http://ydwen.github.io/papers/WenECCV16.pdf)

<h3>Args</h3>


 - **features**: 2-D `tensor` [batch_size, feature_length], input features
 - **label**: 1-D `tensor` [batch_size], input label
 - **alpha**: center loss parameter
 - **num_classes**: a `int` numof classes for training

<h3>Returns</h3>


a `float`, center loss

 ---------- 

