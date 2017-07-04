# base mixin class for prediction

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/prediction.py#L14 target="_blank"><b>tefla.core.prediction.PredictSession</b></a></span>  (weights_from,  gpu_memory_fraction=None)</span>

<h3>Args</h3>


 - **weights_from**: path to the weights file
 - **gpu_memory_fraction**: fraction of gpu memory to use, if not cpu prediction

 --------- 

# One crop Predictor, it predict network out put from a single crop of an input image

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/prediction.py#L46 target="_blank"><b>tefla.core.prediction.OneCropPredictor</b></a></span>  (model,  cnf,  weights_from,  prediction_iterator)</span>

<h3>Args</h3>


 - **model**: model definition file
 - **cnf**: prediction configs
 - **weights_from**: location of the model weights file
 - **prediction_iterator**: iterator to access and augment the data for prediction
 - **gpu_memory_fraction**: fraction of gpu memory to use, if not cpu prediction

 --------- 

# Quasi transform predictor

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/prediction.py#L86 target="_blank"><b>tefla.core.prediction.QuasiPredictor</b></a></span>  (model,  cnf,  weights_from,  prediction_iterator,  number_of_transforms)</span>

<h3>Args</h3>


 - **model**: model definition file
 - **cnf**: prediction configs
 - **weights_from**: location of the model weights file
 - **prediction_iterator**: iterator to access and augment the data for prediction
 - **number_of_transform**: number of determinastic augmentaions to be performed on the input data
resulted predictions are averaged over the augmentated transformation prediction outputs
 - **gpu_memory_fraction**: fraction of gpu memory to use, if not cpu prediction

 --------- 

# Multiples non Data augmented crops predictor

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/prediction.py#L124 target="_blank"><b>tefla.core.prediction.CropPredictor</b></a></span>  (model,  cnf,  weights_from,  prediction_iterator,  im_size,  crop_size)</span>

<h3>Args</h3>


 - **model**: model definition file
 - **cnf**: prediction configs
 - **weights_from**: location of the model weights file
 - **prediction_iterator**: iterator to access and augment the data for prediction
 - **crop_size**: crop size for network input
 - **im_size**: original image size
 - **number_of_crops**: total number of crops to extract from the input image
 - **gpu_memory_fraction**: fraction of gpu memory to use, if not cpu prediction
 - 

 --------- 

# Returns predcitions from multiples models

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/prediction.py#L159 target="_blank"><b>tefla.core.prediction.EnsemblePredictor</b></a></span>  (predictors)</span>

Ensembled predictions from multiples models using ensemble type

<h3>Args</h3>


 - **predictors**: predictor instances

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/prediction.py#L171 target="_blank"><b>predict</b></a></span>  (X,  ensemble_type='mean')</span>

<h5>Args</h5>


 - **X**: 4D tensor, inputs
 - **ensemble_type**: operation to combine models probabilitiesavailable type: ['mean', 'gmean', 'log_mean']

 --------- 

