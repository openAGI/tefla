# Supervised Trainer, support data parallelism, multi GPU

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/learning.py#L28 target="_blank"><b>tefla.core.learning.SupervisedLearner</b></a></span>  (model,  cnf,  clip_by_global_norm=False,  **kwargs)</span>


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



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/learning.py#L54 target="_blank"><b>fit</b></a></span>  (data_set,  weights_from=None,  weights_dir='weights',  start_epoch=1,  summary_every=10,  keep_moving_averages=False,  **kwargs)</span>

<h5>Args</h5>


 - **data_set**: dataset instance to use to access data for training/validation
 - **weights_from**: str, if not None, initializes model from exisiting weights
 - **start_epoch**: int,  epoch number to start training from
e.g. for retarining set the epoch number you want to resume training from
 - **summary_every**: int, epoch interval to write summary; higher value means lower frequency
of summary writing
 - **keep_moving_averages**: a bool, keep moving averages of trainable variables

 --------- 

