# Clips the gradients by the given value

<span class="extra_h1"><span style="color:black;"><b>tefla.core.training.clip_grad_global_norms</b></span>  (tvars,  loss,  opt,  global_norm=1,  gate_gradients=1,  gradient_noise_scale=4.0,  GATE_GRAPH=2,  grad_loss=None,  agre_method=None,  col_grad_ops=False)</span>

<h3>Args</h3>


 - **tvars**: trainable variables used for gradint updates
 - **loss**: total loss of the network
 - **opt**: optimizer
 - **global_norm**: the maximum global norm

<h3>Returns</h3>


A list of clipped gradient to variable pairs.
 

 ---------- 

# Multiply specified gradients

<span class="extra_h1"><span style="color:black;"><b>tefla.core.training.multiply_gradients</b></span>  (grads_and_vars,  gradient_multipliers)</span>

<h3>Args</h3>


 - **grads_and_vars**: A list of gradient to variable pairs (tuples).
 - **gradient_multipliers**: A map from either `Variables` or `Variable` op names
 -   to the coefficient by which the associated gradient should be scaled.

<h3>Returns</h3>


The updated list of gradient to variable pairs.

 ---------- 

# Adds scaled noise from a 0-mean normal distribution to gradients

<span class="extra_h1"><span style="color:black;"><b>tefla.core.training.add_scaled_noise_to_gradients</b></span>  (grads_and_vars,  gradient_noise_scale=10.0)</span>

<h3>Args</h3>


 - **grads_and_vars**: list of gradient and variables
 - **gardient_noise_scale**: value of noise factor

<h3>Returns</h3>


noise added gradients

 ---------- 

