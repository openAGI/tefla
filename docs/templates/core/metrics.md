# Computes accuracy metric

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/metrics.py#L406 target="_blank"><b>tefla.core.metrics.accuracy_op</b></a></span>  (predictions,  targets,  num_classes=5)</span>

<h3>Args</h3>


 - **predictions**: 2D tensor/array, predictions of the network
 - **targets**: 2D tensor/array, ground truth labels of the network
 - **num_classes**: int, num_classes of the network

<h3>Returns</h3>


accuracy

 ---------- 

# Retruns one hot vector

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/metrics.py#L427 target="_blank"><b>tefla.core.metrics.one_hot</b></a></span>  (vec,  m=None)</span>

<h3>Args</h3>


 - **vec**: a vector
 - **m**: num_classes

 ---------- 

# Compute dice coef

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/metrics.py#L452 target="_blank"><b>tefla.core.metrics.dice_coef</b></a></span>  (y_true,  y_pred)</span>

<h3>Args</h3>


 - **y_true**: a 2-D `array`, ground truth label
 - **y_pred**: q 2-D `array`, prediction

<h3>Returns</h3>


a `float`, dice value

 ---------- 

# Computes character level accuracy

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/metrics.py#L468 target="_blank"><b>tefla.core.metrics.char_accuracy</b></a></span>  (predictions,  targets,  rej_char,  streaming=False)</span>
Both predictions and targets should have the same shape
[batch_size x seq_length].

<h3>Args</h3>


 - **predictions**: predicted characters ids.
 - **targets**: ground truth character ids.
 - **rej_char**: the character id used to mark an empty element (end of sequence).
 - **streaming**: if True, uses the streaming mean from the slim.metric module.

<h3>Returns</h3>


a update_ops for execution and value tensor whose value on evaluation
returns the total character accuracy.

 ---------- 

# Computes sequence level accuracy

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/metrics.py#L498 target="_blank"><b>tefla.core.metrics.sequence_accuracy</b></a></span>  (predictions,  targets,  rej_char,  streaming=False)</span>
Both input tensors should have the same shape: [batch_size x seq_length].

<h3>Args</h3>


 - **predictions**: predicted character classes.
 - **targets**: ground truth character classes.
 - **rej_char**: the character id used to mark empty element (end of sequence).
 - **streaming**: if True, uses the streaming mean from the slim.metric module.

<h3>Returns</h3>


a update_ops for execution and value tensor whose value on evaluation
returns the total sequence accuracy.

 ---------- 

