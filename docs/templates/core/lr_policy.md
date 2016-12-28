# Training learning rate schedule based on  inputs dict with epoch number as keys

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/lr_policy.py#L86 target="_blank"><b>tefla.core.lr_policy.StepDecayPolicy</b></a></span>  (schedule,  start_epoch=1)</span>

<h3>Args</h3>


 - **schedule**: a dict, epoch number as keys and learning rate as values
 - **start_epoch**: training start epoch number

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/lr_policy.py#L98 target="_blank"><b>epoch_update</b></a></span>  (learning_rate,  training_history)</span>

<h5>Args</h5>


 - **learning_rate**: previous epoch learning rate
 - **training_histoty**: a dict with epoch, training loss, validation loss as keys

<h5>Returns</h5>


updated learning rate

 --------- 

# Polynomial learning rate decay policy

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/lr_policy.py#L229 target="_blank"><b>tefla.core.lr_policy.PolyDecayPolicy</b></a></span>  (base_lr,  power=10.0,  max_epoch=500,  n_iters_per_epoch=1094)</span>

the effective learning rate follows a polynomial decay, to be
zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)

<h3>Args</h3>


 - **base_lr**: a float, starting learning rate
 - **power**: a float, decay factor
 - **max_epoch**: a int, max training epoch
 - **n_iters_per_epoch**: number of interations per epoch, e.g. total_training_samples/batch_size

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/lr_policy.py#L248 target="_blank"><b>batch_update</b></a></span>  (learning_rate,  iter_idx)</span>

it follows a polynomial decay policy

<h5>Args</h5>


 - **learning_rate**: current batch learning rate
 - **iter_idx**: iteration number,
e.g. number_of_iterations_per_batch*epoch+current_batch_iteration_number

<h5>Returns</h5>


updated_lr

 --------- 

