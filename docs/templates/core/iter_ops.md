# Creates training iterator to access and augment the dataset

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/iter_ops.py#L12 target="_blank"><b>tefla.core.iter_ops.create_training_iters</b></a></span>  (cnf,  data_set,  standardizer,  crop_size,  epoch,  parallel=True)</span>

<h3>Args</h3>


 - **cnf**: configs dict with all training and augmentation params
 - **data_set**: an instance of the dataset class
 - **standardizer**: data samples standardization; either samplewise or aggregate
 - **crop_size**: training time crop_size of the data samples
 - **epoch**: the current epoch number; used for data balancing
 - **parallel**: iterator type; either parallel or queued

 ---------- 

# Creates prediction iterator to access and augment the dataset

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/iter_ops.py#L68 target="_blank"><b>tefla.core.iter_ops.create_prediction_iter</b></a></span>  (cnf,  standardizer,  crop_size,  preprocessor=None,  sync=False)</span>

<h3>Args</h3>


 - **cnf**: configs dict with all training and augmentation params
 - **standardizer**: data samples standardization; either samplewise or aggregate
 - **crop_size**: training time crop_size of the data samples
 - **preprocessor**: data processing or cropping function
 - **sync**: a bool, if False, used parallel iterator

 ---------- 

