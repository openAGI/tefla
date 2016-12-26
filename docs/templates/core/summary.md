# Add summary to a tensor, scalar summary if the tensor is 1D, else scalar and histogram summary

<span class="extra_h1"><span style="color:black;"><b>tefla.core.summary.summary_metric</b></span>  (tensor,  name=None,  collections=None)</span>

<h3>Args</h3>


 - **tensor**: a tensor to add summary
 - **name**: name of the tensor
 - **collections**: training or validation collections

 ---------- 

# Add summary to a tensor, scalar summary if the tensor is 1D, else  scalar and histogram summary

<span class="extra_h1"><span style="color:black;"><b>tefla.core.summary.summary_activation</b></span>  (tensor,  name=None,  collections=None)</span>

<h3>Args</h3>


 - **tensor**: a tensor to add summary
 - **name**: name of the tensor
 - **collections**: training or validation collections

 ---------- 

# creates the summar writter for training and validation

<span class="extra_h1"><span style="color:black;"><b>tefla.core.summary.create_summary_writer</b></span>  (summary_dir,  sess)</span>

<h3>Args</h3>


 - **summary_dir**: the directory to write summary
 - **sess**: the session to sun the ops

<h3>Returns</h3>


training and vaidation summary writter

 ---------- 

# Add summary as per the ops mentioned

<span class="extra_h1"><span style="color:black;"><b>tefla.core.summary.summary_param</b></span>  (op,  tensor,  ndims,  name,  collections=None)</span>

<h3>Args</h3>


 - **op**: name of the summary op; e.g. 'stddev'
available ops: ['scalar', 'histogram', 'sparsity', 'mean', 'rms', 'stddev', 'norm', 'max', 'min']
 - **tensor**: the tensor to add summary
 - **ndims**: dimension of the tensor
 - **name**: name of the op
 - **collections**: training or validation collections

 ---------- 

# Add summary to all trainable tensors

<span class="extra_h1"><span style="color:black;"><b>tefla.core.summary.summary_trainable_params</b></span>  (summary_types,  collections=None)</span>

<h3>Args</h3>


 - **summary_type**: a list of all sumary types to add
e.g.: ['scalar', 'histogram', 'sparsity', 'mean', 'rms', 'stddev', 'norm', 'max', 'min']
 - **collections**: training or validation collections

 ---------- 

# Add summary to all gradient tensors

<span class="extra_h1"><span style="color:black;"><b>tefla.core.summary.summary_gradients</b></span>  (grad_vars,  summary_types,  collections=None)</span>

<h3>Args</h3>


 - **grads_vars**: grads and vars list
 - **summary_type**: a list of all sumary types to add
e.g.: ['scalar', 'histogram', 'sparsity', 'mean', 'rms', 'stddev', 'norm', 'max', 'min']
 - **collections**: training or validation collections

 ---------- 

# Add image summary to a image tensor

<span class="extra_h1"><span style="color:black;"><b>tefla.core.summary.summary_image</b></span>  (tensor,  name=None,  max_images=10,  collections=None)</span>

<h3>Args</h3>


 - **tensor**: a tensor to add summary
 - **name**: name of the tensor
 - **max_images**: num of images to add summary
 - **collections**: training or validation collections

 ---------- 

