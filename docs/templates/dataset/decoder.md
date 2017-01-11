# A Decoder class to decode examples

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/decoder.py#L9 target="_blank"><b>tefla.dataset.decoder.Decoder</b></a></span>  (feature_keys)</span>

<h3>Args</h3>


 - **feature_keys**: a dict, with features name and data types
 - e.g.:
features_keys = {'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),'image/class/label': tf.FixedLenFeature([], tf.int64, - default_value=tf.zeros([], dtype=tf.int64)),
}

<h2>Methods</h2>

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/decoder.py#L32 target="_blank"><b>decode</b></a></span>  (example_serialized,  image_size,  resize_size=None)</span>
<h5>Args</h5>


 - **example_serialized**: scalar Tensor tf.string containing a serialized
Example protocol buffer.
 - **Returns**:
image_buffer: Tensor tf.string containing the contents of a JPEG file.
label: Tensor tf.int32 containing the label.
text: Tensor tf.string containing the human-readable label.

<h5>Returns</h5>


image_buffer: Tensor tf.string containing the contents of a JPEG file.
label: Tensor tf.int32 containing the label.
text: Tensor tf.string containing the human-readable label.

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/decoder.py#L85 target="_blank"><b>distort_image</b></a></span>  (image,  distort_op,  height,  width,  thread_id=0,  scope=None)</span>
<h5>Args</h5>


 - **image**: 3-D float Tensor of image
 - **height**: integer
 - **width**: integer
 - **thread_id**: integer indicating the preprocessing thread.
 - **scope**: Optional scope for name_scope.
<h5>Returns</h5>


 - 3-D float Tensor of distorted image used for training.

<h5>Returns</h5>


3-D float Tensor of distorted image used for training.

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/decoder.py#L104 target="_blank"><b>eval_image</b></a></span>  (image,  height,  width,  scope=None)</span>
<h5>Args</h5>


 - **image**: 3-D float Tensor
 - **height**: integer
 - **width**: integer
 - **scope**: Optional scope for name_scope.
<h5>Returns</h5>


 - 3-D float Tensor of prepared image.

<h5>Returns</h5>


3-D float Tensor of prepared image.

 <span class="hr_large"></span> 



<span class="extra_h2"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/dataset/decoder.py#L133 target="_blank"><b>parse_example_proto</b></a></span>  (example_serialized,  is_bbox=False)</span>
The output of the build_image_data.py image preprocessing script is a dataset
containing serialized Example protocol buffers. Each Example proto contains
the following fields:
image/height: 462
image/width: 581
image/colorspace: 'RGB'
image/channels: 3
image/class/label: 615
image/class/synset: 'n03623198'
image/class/text: 'knee pad'
image/object/bbox/xmin: 0.1
image/object/bbox/xma<h5>x</h5>

 0.9
image/object/bbox/ymin: 0.2
image/object/bbox/yma<h5>x</h5>

 0.6
image/object/bbox/label: 615
image/format: 'JPEG'
image/filename: 'ILSVRC2012_val_00041207.JPEG'
image/encoded: <JPEG encoded string>

<h5>Args</h5>


 - **example_serialized**: scalar Tensor tf.string containing a serialized
 - Example protocol buffer.

<h5>Returns</h5>


image_buffer: Tensor tf.string containing the contents of a JPEG file.
label: Tensor tf.int32 containing the label.
bbo<h5>x</h5>

 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
where each coordinate is [0, 1) and the coordinates are arranged as
[ymin, xmin, ymax, xmax].
text: Tensor tf.string containing the human-readable label.

 --------- 

