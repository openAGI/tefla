# TFrecords reader class

<span class="extra_h1"><span style="color:black;"><b>tefla.dataset.reader.Reader</b></span>  (dataset,  reader_kwargs=None,  shuffle=True,  num_readers=16,  capacity=1,  num_epochs=None)</span>

<h3>Args</h3>


 - **dataset**: an instance of the dataset class
 - **reader_kwargs**: extra arguments to be passed to the TFRecordReader
 - **shuffle**: whether to shuffle the dataset
 - **num_readers**:a int, num of readers to launch
 - **capacity**: a int, capacity of the queue used
 - **num_epochs**: a int, num of epochs for training or validation

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black"><b>parallel_reader</b></span>  (min_queue_examples=1024)</span>

Primarily used for Training ops

<h5>Args</h5>


 - **min_queue_examples**: min number of queue examples after dequeue

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black"><b>single_reader</b></span>  (num_epochs=1,  shuffle=False,  capacity=1)</span>

Data will be read using single TFRecordReader, primarily used for validation

<h5>Args</h5>


 - **num_epochs**: number of epoch
 - **shuffle**: shuffle the dataset. False for validation
 - **capacity**: queue capacity

<h5>Returns</h5>


a single item from the tfrecord files

 --------- 

