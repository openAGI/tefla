# Dataflow handling class

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/dataflow.py#L12 target="_blank"><b>tefla.dataset.dataflow.Dataflow</b></a></span>  (dataset,  num_readers=1,  shuffle=True,  num_epochs=None,  min_queue_examples=1024,  capacity=2048)</span>

<h3>Args</h3>


 - **dataset**: an instance of the dataset class
 - **num_readers**: num of readers to  read the dataset
 - **shuffle**: a bool, shuffle the dataset
 - **num_epochs**: total number of epoch for training or validation
 - **min_queue_examples**: minimum number of items after dequeue
 - **capacity**: total queue capacity

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/dataflow.py#L31 target="_blank"><b>get</b></a></span>  (items,  image_size,  resize_size=None)</span>

<h5>Args</h5>


 - **items**: a list, with items to get from the dataset
e.g.: ['image', 'label']
 - **image_size**: a list with original image size
e.g.: [width, height, channel]
 - **resize_size**: if image resize required, provide a list of width and height
e.g.: [width, height]

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/dataflow.py#L52 target="_blank"><b>get_batch</b></a></span>  (batch_size,  target_probs,  image_size,  resize_size=None,  init_probs=None,  enqueue_many=False,  queue_capacity=2048,  threads_per_queue=1,  name='balancing_op')</span>

Stochastically creates batches based on per-class probabilities.
This method discards examples. Internally, it creates one queue to
amortize the cost of disk reads, and one queue to hold the properly-proportioned batch.

<h5>Args</h5>


 - **batch_size**: a int, batch_size
 - **target_probs**: probabilities of class samples to be present in the batch
 - **image_size**: a list with original image size
e.g.: [width, height, channel]
 - **resize_size**: if image resize required, provide a list of width and height
e.g.: [width, height]
 - **init_probs**: initial probs of data sample in the first batch
 - **enqueue_many**: bool, if true, interpret input tensors as having a batch dimension.
 - **queue_capacity**: Capacity of the large queue that holds input examples.
 - **threads_per_queue**: Number of threads for the large queue that holds
input examples and for the final queue with the proper class proportions.
 - **name**: a optional scope/name of the op

 --------- 

