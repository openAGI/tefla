# Define a log loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L16 target="_blank"><b>tefla.core.losses.log_loss_custom</b></a></span>  (predictions,  labels,  eps=1e-07,  name='log')</span>

<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D or array tensor, [batch_size, num_classes]  ground truth labels or target labels.
 - **eps**: a constant to set upper or lower limit for labels, smoothening factor
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


A tensor with the log loss.

 ---------- 

# Define a log loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L37 target="_blank"><b>tefla.core.losses.log_loss_tf</b></a></span>  (predictions,  labels,  eps=1e-07,  weights=1.0,  name='log_loss')</span>

<h3>Args</h3>


 - **predictions**: 2D tensor or array, [batch_size, num_classes] predictions of the network .
 - **labels**: 2D or array tensor, [batch_size, num_classes]  ground truth labels or target labels.
 - **eps**: a constant to set upper or lower limit for labels, smoothening factor
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


A tensor with the log loss.

 ---------- 

# Define a kappa loss, Its a continuous differentiable approximation of discrete kappa loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L58 target="_blank"><b>tefla.core.losses.kappa_loss</b></a></span>  (predictions,  labels,  y_pow=1,  eps=1e-15,  num_ratings=5,  batch_size=32,  name='kappa')</span>

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

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L103 target="_blank"><b>tefla.core.losses.kappa_log_loss</b></a></span>  (predictions,  labels,  label_smoothing=0.0,  y_pow=1,  batch_size=32,  log_scale=0.5,  num_classes=5,  log_offset=0.5,  name='kappa_log')</span>

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

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L134 target="_blank"><b>tefla.core.losses.kappa_log_loss_clipped</b></a></span>  (predictions,  labels,  label_smoothing=0.0,  y_pow=1,  batch_size=32,  log_scale=0.5,  log_cutoff=0.8,  num_classes=5,  name='kappa_log_clipped')</span>

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

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L165 target="_blank"><b>tefla.core.losses.cross_entropy_loss</b></a></span>  (logits,  labels,  label_smoothing=0.0,  weight=1.0,  name='cross_entropy_loss')</span>

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

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L195 target="_blank"><b>tefla.core.losses.l1_l2_regularizer</b></a></span>  (var,  weight_l1=1.0,  weight_l2=1.0,  name='l1_l2_regularizer')</span>

<h3>Args</h3>


 - **var**: tensor to regularize.
 - **weight_l1**: an optional weight to modulate the l1 loss.
 - **weight_l2**: an optional weight to modulate the l2 loss.
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


the l1+L2 loss op.

 ---------- 

# Returns a function that can be used to apply L1 regularization to weights

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L218 target="_blank"><b>tefla.core.losses.l1_regularizer</b></a></span>  (scale,  name='l1_regularizer')</span>
L1 regularization encourages sparsity.

<h3>Args</h3>


  scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
  name: An optional name/scope name.

<h3>Returns</h3>


  A function with signature `l1(weights)` that apply L1 regularization.

 ---------- 

# Returns a function that can be used to apply L2 regularization to weights

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L255 target="_blank"><b>tefla.core.losses.l2_regularizer</b></a></span>  (scale,  name='l2_regularizer')</span>
Small values of L2 can help prevent overfitting the training data.

<h3>Args</h3>


  scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
  name: An optional name/scope name.

<h3>Returns</h3>


  A function with signature `l2(weights)` that applies L2 regularization.

 ---------- 

# log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L289 target="_blank"><b>tefla.core.losses.discretized_mix_logistic_loss</b></a></span>  (inputs,  predictions,  sum_all=True,  name='disretized_mix_logistic_loss')</span>

<h3>Args</h3>


 - **predictions**: 4D tensor or array, [batch_size, width, height, out_channels] predictions of the network .
 - **inputs**: 4D tensor or array, [batch_size, width, height, num_classes] ground truth labels or target labels.
 - **name**: Optional scope/name for op_scope.

<h3>Returns</h3>


A tensor with the discretized mix logistic loss.

 ---------- 

# Pull Away loss calculation

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L351 target="_blank"><b>tefla.core.losses.pullaway_loss</b></a></span>  (embeddings,  name='pullaway_loss')</span>

<h3>Args</h3>


 - **embeddings**: The embeddings to be orthogonalized for varied faces. Shape [batch_size, embeddings_dim]

 ---------- 

# Calculate the loss from the logits and the labels

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L385 target="_blank"><b>tefla.core.losses.segment_loss</b></a></span>  (logits,  labels,  num_classes,  head=None)</span>

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

 ---------- 

# Calculate the triplet loss according to the FaceNet paper

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L420 target="_blank"><b>tefla.core.losses.triplet_loss</b></a></span>  (anchor,  positive,  negative,  alpha=0.2,  name='triplet_loss')</span>

<h3>Args</h3>


  anchor: 2-D `tensor` [batch_size, embedding_size], the embeddings for the anchor images.
  positive: 2-D `tensor` [batch_size, embedding_size], the embeddings for the positive images.
  negative: 2-D `tensor` [batch_size, embedding_size], the embeddings for the negative images.
  alpha: positive to negative triplet distance margin

<h3>Returns</h3>


  the triplet loss.

 ---------- 

# Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L440 target="_blank"><b>tefla.core.losses.decov_loss</b></a></span>  (xs,  name='decov_loss')</span>
'Reducing Overfitting In Deep Networks by Decorrelating Representation'

<h3>Args</h3>


 - **xs**: 4-D `tensor` [batch_size, height, width, channels], input

<h3>Returns</h3>


a `float` decov loss

 ---------- 

# Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L461 target="_blank"><b>tefla.core.losses.center_loss</b></a></span>  (features,  label,  alpha,  num_classes,  name='center_loss')</span>
   (http://ydwen.github.io/papers/WenECCV16.pdf)

<h3>Args</h3>


 - **features**: 2-D `tensor` [batch_size, feature_length], input features
 - **label**: 1-D `tensor` [batch_size], input label
 - **alpha**: center loss parameter
 - **num_classes**: a `int` numof classes for training

<h3>Returns</h3>


a `float`, center loss

 ---------- 

# Adds a similarity loss term, the correlation between two representations

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L486 target="_blank"><b>tefla.core.losses.correlation_loss</b></a></span>  (source_samples,  target_samples,  weight,  name='corr_loss')</span>

<h3>Args</h3>


 - **source_samples**: a tensor of shape [num_samples, num_features]
 - **target_samples**: a tensor of shape [num_samples, num_features]
 - **weight**: a scalar weight for the loss.
 - **scope**: optional name scope for summary tags.

<h3>Returns</h3>


a scalar tensor representing the correlation loss value.

 ---------- 

# Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L516 target="_blank"><b>tefla.core.losses.maximum_mean_discrepancy</b></a></span>  (x,  y,  kernel=<function  gaussian_kernel_matrix  at  0x7fc2376eade8>,  name='maximum_mean_discrepancy')</span>

Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
the distributions of x and y. Here we use the kernel two sample estimate
using the empirical mean of the two distributions.

MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2= \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

where K = <\phi(x), \phi(y)>,
  is the desired kernel function, in this case a radial basis kernel.

<h3>Args</h3>


 - **x**: a tensor of shape [num_samples, num_features]
 - **y**: a tensor of shape [num_samples, num_features]
 - **kernel**: a function which computes the kernel in MMD. Defaults to theGaussianKernelMatrix.

<h3>Returns</h3>


a scalar denoting the squared maximum mean discrepancy loss.

 ---------- 

# Adds a similarity loss term, the MMD between two representations

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L549 target="_blank"><b>tefla.core.losses.mmd_loss</b></a></span>  (source_samples,  target_samples,  weight,  name='mmd_loss')</span>

This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
different Gaussian kernels.

<h3>Args</h3>


  source_samples: a tensor of shape [num_samples, num_features].
  target_samples: a tensor of shape [num_samples, num_features].
  weight: the weight of the MMD loss.
  scope: optional name scope for summary tags.

<h3>Returns</h3>


  a scalar tensor representing the MMD loss value.

 ---------- 

# Adds the domain adversarial (DANN) loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L582 target="_blank"><b>tefla.core.losses.dann_loss</b></a></span>  (source_samples,  target_samples,  weight,  name='dann_loss')</span>

<h3>Args</h3>


  source_samples: a tensor of shape [num_samples, num_features].
  target_samples: a tensor of shape [num_samples, num_features].
  weight: the weight of the loss.
  scope: optional name scope for summary tags.

<h3>Returns</h3>


  a scalar tensor representing the correlation loss value.

 ---------- 

# Adds the difference loss between the private and shared representations

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L624 target="_blank"><b>tefla.core.losses.difference_loss</b></a></span>  (private_samples,  shared_samples,  weight=1.0,  name='difference_loss')</span>

<h3>Args</h3>


  private_samples: a tensor of shape [num_samples, num_features].
  shared_samples: a tensor of shape [num_samples, num_features].
  weight: the weight of the incoherence loss.
  name: the name of the tf summary.

 ---------- 

# A helper function to compute the error between quaternions

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L652 target="_blank"><b>tefla.core.losses.log_quaternion_loss_batch</b></a></span>  (predictions,  labels,  name='log_quaternion_batch_loss')</span>

<h3>Args</h3>


  predictions: A Tensor of size [batch_size, 4].
  labels: A Tensor of size [batch_size, 4].
  params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.

<h3>Returns</h3>


  A Tensor of size [batch_size], denoting the error between the quaternions.

 ---------- 

# A helper function to compute the mean error between batches of quaternions

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L678 target="_blank"><b>tefla.core.losses.log_quaternion_loss</b></a></span>  (predictions,  labels,  batch_size,  name='log_quaternion_loss')</span>

The caller is expected to add the loss to the graph.

<h3>Args</h3>


  predictions: A Tensor of size [batch_size, 4].
  labels: A Tensor of size [batch_size, 4].
  params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.

<h3>Returns</h3>


  A Tensor of size 1, denoting the mean error between batches of quaternions.

 ---------- 

# Adds noise to embeddings and recomputes classification loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L699 target="_blank"><b>tefla.core.losses.random_perturbation_loss</b></a></span>  (embedded,  length,  loss_fn,  perturb_norm_length=0.1)</span>

<h3>Args</h3>


 - **embedded**: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim]
 - **length**: a `int`, length of the mask
 - **loss_fn**: a callable, that returns loss
 - **perturb_norm_length**: a `float`, Norm length of adversarial perturbation to be optimized with validatio

<h3>Returns</h3>


perturbation loss

 ---------- 

# Adds gradient to embedding and recomputes classification loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L716 target="_blank"><b>tefla.core.losses.adversarial_loss</b></a></span>  (embedded,  loss,  loss_fn,  perturb_norm_length=0.1)</span>

<h3>Args</h3>


 - **embedded**: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim]
 - **loss**: `float`, loss
 - **loss_fn**: a callable, that returns loss
 - **perturb_norm_length**: a `float`, Norm length of adversarial perturbation to be optimized with validatio

<h3>Returns</h3>


adversial loss

 ---------- 

# Virtual adversarial loss

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L737 target="_blank"><b>tefla.core.losses.virtual_adversarial_loss</b></a></span>  (logits,  embedded,  labels,  length,  logits_from_embedding_fn,  num_classes,  num_power_iteration=1,  small_constant_for_finite_diff=0.001,  perturb_norm_length=0.1)</span>
Computes virtual adversarial perturbation by finite difference method and
power iteration, adds it to the embedding, and computes the KL divergence
between the new logits and the original logits.

<h3>Args</h3>


 - **logits**: 2-D float `Tensor`, [num_timesteps*batch_size, m], where m=1 if
num_classes=2, otherwise m=num_classes.
 - **embedded**: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim].
 - **labels**: 1-D `Tensor`, input labels
 - **length**: a `int`, input length
 - **logits_from_embedding_fn**: callable that takes embeddings and returns
classifier logits.
 - **num_classes**: num_classes for training
 - **vocab_size**: a `int`, vocabular size of the problem
 - **num_power_iteration**: a `int`, the number of power iteration
 - **small_constant_for_finite_diff**: a `float`, Small constant for finite difference method
 - **perturb_norm_length**: a `float`, Norm length of adversarial perturbation to be optimized with validatio

<h3>Returns</h3>


a `float` `scalar`, KL divergence.

 ---------- 

# Adds noise to embeddings and recomputes classification loss fir bidirectional rnn models

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L781 target="_blank"><b>tefla.core.losses.random_perturbation_loss_brnn</b></a></span>  (embedded,  length,  loss_fn,  perturb_norm_length=0.1)</span>

<h3>Args</h3>


 - **embedded**: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim]
 - **length**: a `int`, length of the mask
 - **loss_fn**: a callable, that returns loss
 - **perturb_norm_length**: a `float`, Norm length of adversarial perturbation to be optimized with validatio

<h3>Returns</h3>


perturbation loss

 ---------- 

# Adds gradient to embeddings and recomputes classification loss for bidirectional rnn models

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L799 target="_blank"><b>tefla.core.losses.adversarial_loss_brnn</b></a></span>  (embedded,  loss,  loss_fn,  perurb_norm_length=0.1)</span>

<h3>Args</h3>


 - **embedded**: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim]
 - **loss**: `float`, loss
 - **loss_fn**: a callable, that returns loss
 - **perturb_norm_length**: a `float`, Norm length of adversarial perturbation to be optimized with validatio

<h3>Returns</h3>


adversial loss

 ---------- 

# Virtual adversarial loss for bidirectional models

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L818 target="_blank"><b>tefla.core.losses.virtual_adversarial_loss_brnn</b></a></span>  (logits,  embedded,  labels,  length,  logits_from_embedding_fn,  vocab_size,  num_classes,  num_power_iteration=1,  small_constant_for_finite_diff=0.001,  perturb_norm_length=0.1)</span>
Computes virtual adversarial perturbation by finite difference method and
power iteration, adds it to the embedding, and computes the KL divergence
between the new logits and the original logits.

<h3>Args</h3>


 - **logits**: 2-D float `Tensor`, [num_timesteps*batch_size, m], where m=1 if
num_classes=2, otherwise m=num_classes.
 - **embedded**: 3-D float `Tensor`, [batch_size, num_timesteps, embedding_dim].
 - **labels**: 1-D `Tensor`, input labels
 - **length**: a `int`, input length
 - **logits_from_embedding_fn**: callable that takes embeddings and returns
classifier logits.
 - **num_classes**: num_classes for training
 - **vocab_size**: a `int`, vocabular size of the problem
 - **num_power_iteration**: a `int`, the number of power iteration
 - **small_constant_for_finite_diff**: a `float`, Small constant for finite difference method
 - **perturb_norm_length**: a `float`, Norm length of adversarial perturbation to be optimized with validatio

<h3>Returns</h3>


a `float` `scalar`, KL divergence.

 ---------- 

# Generate a mask for the EOS token (1.0 on EOS, 0.0 otherwise)

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L879 target="_blank"><b>tefla.core.losses._end_of_seq_mask</b></a></span>  (tokens,  vocab_size)</span>

<h3>Args</h3>


 - **tokens**: 1-D integer `Tensor` [num_timesteps*batch_size]. Each element is an
id from the vocab.
 - **vocab_size**: a `int`, vocabular size of the problem

<h3>Returns</h3>


Float 1-D `Tensor` same shape as tokens, whose values are 1.0 on the end of
sequence and 0.0 on the others.

 ---------- 

# Returns weighted KL divergence between distributions q and p

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L895 target="_blank"><b>tefla.core.losses._kl_divergence_with_logits</b></a></span>  (q_logits,  p_logits,  weights,  num_classes)</span>

<h3>Args</h3>


 - **q_logits**: logits for 1st argument of KL divergence shape
  [num_timesteps * batch_size, num_classes] if num_classes > 2, and
  [num_timesteps * batch_size] if num_classes == 2.
 - **p_logits**: logits for 2nd argument of KL divergence with same shape q_logits.
 - **weights**: 1-D `float` tensor with shape [num_timesteps * batch_size].
 Elements should be 1.0 only on end of sequences
 - **num_classes**: a `int`, number of training classes

<h3>Returns</h3>


a `float` `scalar`, KL divergence.

 ---------- 

# Calculates the per-example cross-entropy loss for a sequence of logits and

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/losses.py#L931 target="_blank"><b>tefla.core.losses.cross_entropy_sequence_loss</b></a></span>  (logits,  targets,  sequence_length)</span>
masks out all losses passed the sequence length.

<h3>Args</h3>


 - **logits**: Logits of shape `[T, B, vocab_size]`
 - **targets**: Target classes of shape `[T, B]`
 - **sequence_length**: An int32 tensor of shape `[B]` corresponding
 -to the length of each input

<h3>Returns</h3>


A tensor of shape [T, B] that contains the loss per example, per time step.

 ---------- 

