# Supervised Learner, support data parallelism, multi GPU, accept TFRecords data as input

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/learningv2.py#L27 target="_blank"><b>tefla.core.learning_v2.SupervisedLearner</b></a></span>  (model,  cnf,  clip_by_global_norm=False,  **kwargs)</span>


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



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/learningv2.py#L53 target="_blank"><b>fit</b></a></span>  (data_dir,  data_dir_val=None,  features_keys=None,  weights_from=None,  start_epoch=1,  summary_every=10,  training_set_size=None,  val_set_size=None,  dataset_name='cifar10',  keep_moving_averages=False)</span>

<h5>Args</h5>


 - **data_dir**: str, training dataset directory (where tfrecords are staored for training)
 - **data_dir_val**: str optional, validation dataset directory (where tfrecords are stored for validation)
 - **features_keys**: a dict, tfrecords keys to datum features
 - e.g.:
 - features_keys = {
'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),
'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
 - }
 - **weights_from**: str, if not None, initializes model from exisiting weights
 - **training_set_size**: int, number of training examples
 - **val_set_size**: int, set if data_dir_val not None, number of validation examples
 - **dataset_name**: a optional, Name of the dataset
 - **start_epoch**: int,  epoch number to start training from
e.g. for retarining set the epoch number you want to resume training from
 - **summary_every**: int, epoch interval to write summary; higher value means lower frequency
of summary writing
 - **keep_moving_averages**: a bool, keep moving averages of trainable variables

 --------- 

