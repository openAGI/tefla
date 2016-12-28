# He Normal initializer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/initializers.py#L10 target="_blank"><b>tefla.core.initializers.he_normal</b></a></span>  (seed=None,  scale=1.0,  dtype=tf.float32)</span>
Kaiming He et al. (2015): Delving deep into rectifiers: Surpassing human-level 
performance on imagenet classification. arXiv preprint arXiv:1502.01852.

<h3>Args</h3>


 - **scale**: float
   Scaling factor for the weights. Set this to ``1.0`` for linear and
   sigmoid units, to ``sqrt(2)`` for rectified linear units, and
   to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
   leakiness ``alpha``. Other transfer functions may need different factors.

 ---------- 

# He Uniform initializer

<span class="extra_h1"><span style="color:black;"><a href=https://github.com/n3011/tefla/blob/master/tefla/core/initializers.py#L27 target="_blank"><b>tefla.core.initializers.he_uniform</b></a></span>  (seed=None,  scale=1.0,  dtype=tf.float32)</span>

<h3>Args</h3>


 - **scale**: float
   Scaling factor for the weights. Set this to ``1.0`` for linear and
   sigmoid units, to ``sqrt(2)`` for rectified linear units, and
   to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
   leakiness ``alpha``. Other transfer functions may need different factors.

 ---------- 

