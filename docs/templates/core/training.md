# Supervised Trainer class

<span class="extra_h1"><span style="color:black;"><b>tefla.core.training.SupervisedTrainer</b></span>  (model,  cnf,  training_iterator=<tefla.da.iterator.BatchIterator  object  at  0x7f97bccd0e50>,  validation_iterator=<tefla.da.iterator.BatchIterator  object  at  0x7f97bccd0e90>,  start_epoch=1,  resume_lr=0.01,  classification=True,  clip_norm=True,  n_iters_per_epoch=1094,  gpu_memory_fraction=0.94,  is_summary=False)</span>

<h3>Args</h3>


 - **model**: model definition 
 - **cnf**: dict, training configs
 - **training_iterator**: iterator to use for training data access, processing and augmentations
 - **validation_iterator**: iterator to use for validation data access, processing and augmentations
 - **start_epoch**: int, training start epoch; for resuming training provide the last 
 - epoch number to resume training from, its a required parameter for training data balancing
 - **resume_lr**: float, learning rate to use for new training
 - **classification**: bool, classificattion or regression
 - **clip_norm**: bool, to clip gradient using gradient norm, stabilizes the training
 - **n_iters_per_epoch**: int,  number of iteratiosn for each epoch; 
e.g: total_training_samples/batch_size
 - **gpu_memory_fraction**: amount of gpu memory to use
 - **is_summary**: bool, to write summary or not

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black"><b>fit</b></span>  (data_set,  weights_from=None,  start_epoch=1,  summary_every=10,  verbose=0)</span>

<h5>Args</h5>


 - **data_set**: dataset instance to use to access data for training/validation
 - **weights_from**: str, if not None, initializes model from exisiting weights
 - **start_epoch**: int,  epoch number to start training from
e.g. for retarining set the epoch number you want to resume training from
 - **summary_every**: int, epoch interval to write summary; higher value means lower frequency
of summary writing
 - **verbose**: log level

 --------- 

# Clips the gradients by the given value

<span class="extra_h1"><span style="color:black;"><b>tefla.core.training.clip_grad_global_norms</b></span>  (tvars,  loss,  opt,  global_norm=1,  gate_gradients=1,  gradient_noise_scale=4.0,  GATE_GRAPH=2,  grad_loss=None,  agre_method=None,  col_grad_ops=False)</span>

<h3>Args</h3>


 - **tvars**: trainable variables used for gradint updates
 - **loss**: total loss of the network
 - **opt**: optimizer
 - **global_norm**: the maximum global norm

<h3>Returns</h3>


A list of clipped gradient to variable pairs.
 

 ---------- 

# Multiply specified gradients

<span class="extra_h1"><span style="color:black;"><b>tefla.core.training.multiply_gradients</b></span>  (grads_and_vars,  gradient_multipliers)</span>

<h3>Args</h3>


 - **grads_and_vars**: A list of gradient to variable pairs (tuples).
 - **gradient_multipliers**: A map from either `Variables` or `Variable` op names
 -   to the coefficient by which the associated gradient should be scaled.

<h3>Returns</h3>


The updated list of gradient to variable pairs.

 ---------- 

# Adds scaled noise from a 0-mean normal distribution to gradients

<span class="extra_h1"><span style="color:black;"><b>tefla.core.training.add_scaled_noise_to_gradients</b></span>  (grads_and_vars,  gradient_noise_scale=10.0)</span>

<h3>Args</h3>


 - **grads_and_vars**: list of gradient and variables
 - **gardient_noise_scale**: value of noise factor

<h3>Returns</h3>


noise added gradients

 ---------- 

