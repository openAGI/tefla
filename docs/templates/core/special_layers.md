# Spatial Transformer Layer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/special_layers.py#L4 target="_blank"><b>tefla.core.special_layers.spatialtransformer</b></a></span>  (U,  theta,  batch_size=64,  downsample_factor=1.0,  num_transform=1,  name='SpatialTransformer',  **kwargs)</span>

Implements a spatial transformer layer as described in [1]_.
It's based on lasagne implementation in [2]_, modified by Mrinal Haloi

<h3>Args</h3>


 - **U**: float
The output of a convolutional net should have the
shape [batch_size, height, width, num_channels].
 - **theta**: float
The output of the localisation network should be [batch_size, num_transform, 6] or [batch_size, 6] if num_transform=1
```python`theta`` to : - identity = np.array([[1., 0., 0.], -  [0., 1., 0.]]) - identity = identity.flatten() - theta = tf.Variable(initial_value=identity)
```
 - **downsample_factor**: a float, determines output shape, downsample input shape by downsample_factor

<h3>Returns</h3>


spatial transformed output of the network

 ---------- 

