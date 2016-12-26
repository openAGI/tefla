# Semi Supervised Trainer

<span class="extra_h1"><span style="color:black;"><b>tefla.core.learning_ss.SemiSupervisedTrainer</b></span>  (model,  cnf,  clip_by_global_norm=False,  **kwargs)</span>


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



<span class="extra_h2"><span style="color:black"><b>fit</b></span>  (data_set,  weights_from=None,  start_epoch=1,  summary_every=199,  model_name='multiclass_ss',  weights_dir='weights')</span>

<h5>Args</h5>


 - **data_set**: dataset instance to use to access data for training/validation
 - **weights_from**: str, if not None, initializes model from exisiting weights
 - **start_epoch**: int,  epoch number to start training from
e.g. for retarining set the epoch number you want to resume training from
 - **summary_every**: int, epoch interval to write summary; higher value means lower frequency
of summary writing

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black"><b>sigmoid_kl_with_logits</b></span>  (logits,  targets)</span>
<h5>Args</h5>


 - **logits**: logits
 - **targets**: smooth targets

<h5>Returns</h5>


cross entropy loss

 --------- 

