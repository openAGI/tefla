# One crop Predictor, it predict network out put from a single crop of an input image

<span class="extra_h1"><span style="color:black;"><b>tefla.core.prediction.OneCropPredictor</b></span>  (model,  cnf,  weights_from,  prediction_iterator)</span>

<h3>Args</h3>


 - **model**: model definition file
 - **cnf**: prediction configs
 - **weights_from**: location of the model weights file
 - **prediction_iterator**: iterator to access and augment the data for prediction

 --------- 

# Quasi transform predictor

<span class="extra_h1"><span style="color:black;"><b>tefla.core.prediction.QuasiPredictor</b></span>  (model,  cnf,  weights_from,  prediction_iterator,  number_of_transforms)</span>

<h3>Args</h3>


 - **model**: model definition file
 - **cnf**: prediction configs
 - **weights_from**: location of the model weights file
 - **prediction_iterator**: iterator to access and augment the data for prediction
 - **number_of_transform**: number of determinastic augmentaions to be performed on the input data
resulted predictions are averaged over the augmentated transformation prediction outputs

 --------- 

# Multiples non Data augmented crops predictor

<span class="extra_h1"><span style="color:black;"><b>tefla.core.prediction.CropPredictor</b></span>  (model,  cnf,  weights_from,  prediction_iterator,  crop_size,  im_size,  number_of_crops=10)</span>

<h3>Args</h3>


 - **model**: model definition file
 - **cnf**: prediction configs
 - **weights_from**: location of the model weights file
 - **prediction_iterator**: iterator to access and augment the data for prediction
 - **crop_size**: crop size for network input
 - **im_size**: original image size
 - **number_of_crops**: total number of crops to extract from the input image
 - 

 --------- 

# Returns predcitions from multiples models

<span class="extra_h1"><span style="color:black;"><b>tefla.core.prediction.EnsemblePredictor</b></span>  (predictors)</span>

Ensembled predictions from multiples models using ensemble type

<h3>Args</h3>


 - **predictors**: predictor instances

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black"><b>predict</b></span>  (X,  ensemble_type='mean')</span>

<h5>Args</h5>


 - **X**: 4D tensor, inputs
 - **ensemble_type**: operation to combine models probabilitiesavailable type: ['mean', 'gmean', 'log_mean']

 --------- 

