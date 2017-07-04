# Samplewise Standardizer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/standardizer.py#L29 target="_blank"><b>tefla.da.standardizer.SamplewiseStandardizer</b></a></span>  (clip,  channel_wise=False)</span>

<h3>Args</h3>


 - **clip**: max/min allowed value in the output image
e.g.: 6
 - **channel_wise**: perform standarization separately accross channels

 --------- 

# Samplewise Standardizer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/standardizer.py#L60 target="_blank"><b>tefla.da.standardizer.SamplewiseStandardizerTF</b></a></span>  (clip,  channel_wise=False)</span>

<h3>Args</h3>


 - **clip**: max/min allowed value in the output image
e.g.: 6
 - **channel_wise**: perform standarization separately accross channels

 --------- 

# Aggregate Standardizer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/standardizer.py#L88 target="_blank"><b>tefla.da.standardizer.AggregateStandardizer</b></a></span>  (mean,  std,  u,  ev,  sigma=0.0,  color_vec=None)</span>

Creates a standardizer based on whole training dataset

<h3>Args</h3>


 - **mean**: 1-D array, aggregate mean array
e.g.: mean is calculated for each color channel, R, G, B
 - **std**: 1-D array, aggregate standard deviation array
e.g.: std is calculated for each color channel, R, G, B
 - **u**: 2-D array, eigenvector for the color channel variation
 - **ev**: 1-D array, eigenvalues
 - **sigma**: float, noise factor
 - **color_vec**: an optional color vector

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/standardizer.py#L128 target="_blank"><b>augment_color</b></a></span>  (img,  sigma=0.0,  color_vec=None)</span>

<h5>Args</h5>


 - **img**: input image
 - **sigma**: a float, noise factor
 - **color_vec**: an optional color vec

 --------- 

# Aggregate Standardizer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/standardizer.py#L148 target="_blank"><b>tefla.da.standardizer.AggregateStandardizerTF</b></a></span>  (mean,  std,  u,  ev,  sigma=0.0,  color_vec=None)</span>

Creates a standardizer based on whole training dataset

<h3>Args</h3>


 - **mean**: 1-D array, aggregate mean array
e.g.: mean is calculated for each color channel, R, G, B
 - **std**: 1-D array, aggregate standard deviation array
e.g.: std is calculated for each color channel, R, G, B
 - **u**: 2-D array, eigenvector for the color channel variation
 - **ev**: 1-D array, eigenvalues
 - **sigma**: float, noise factor
 - **color_vec**: an optional color vector

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/da/standardizer.py#L188 target="_blank"><b>augment_color</b></a></span>  (img,  sigma=0.0,  color_vec=None)</span>

<h5>Args</h5>


 - **img**: input image
 - **sigma**: a float, noise factor
 - **color_vec**: an optional color vec

 --------- 

