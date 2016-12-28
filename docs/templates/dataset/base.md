# A simple class for handling data sets,

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/base.py#L17 target="_blank"><b>tefla.dataset.base.Dataset</b></a></span>  (name,  decoder,  data_dir=None,  num_classes=10,  num_examples_per_epoch=1,  items_to_descriptions=None,  **kwargs)</span>

<h3>Args</h3>


 - **name**: a string, Name of the class instance
 - **decoder**: object instance, tfrecords object decoding and image encoding and decoding
 - **data_dir**: a string, path to the data folder
 - **num_classes**: num of classes of the dataset
 - **num_examples_per_epoch**: total number of examples per epoch
 - **items_to_description**: a string descriving the items of the dataset

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/base.py#L61 target="_blank"><b>data_files</b></a></span>  (self)</span>

<h5>Returns</h5>


python list of all (sharded) data set files.

 --------- 

