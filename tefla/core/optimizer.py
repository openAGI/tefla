import numpy as np
import numbers
import tensorflow as tf
from tensorflow import random_normal, shape
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class RMSPropunroll(optimizer.Optimizer):

    def __init__(self, learning_rate=1e-4, use_locking=False, name='SMORMS3'):
        super(RMSPropunroll, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._eps_t = None
        self._g2 = None
        self._g = None
        self._xi = None

    def _prepare(self):
        if isinstance(self._learning_rate, numbers.Number):
            self._eps_placeholder = None
            self._eps_op = None
            self._eps_t = tf.convert_to_tensor(
                self._learning_rate, name="epsilon")
        else:
            self._eps_placeholder = tf.placeholder(
                self._learning_rate.dtype, [1], name='eps_placeholder')
            self._eps_op = tf.assign(
                self._learning_rate, self._eps_placeholder)
            self._eps_t = self._learning_rate

    def _create_slots(self, var_list):
        if self._g2 is None:
            with tf.Graph.colocate_with(var_list[0]):
                for v in var_list:
                    self._xi = self._zeros_slot(v, "xi", self._name)
                    self._g = self._zeros_slot(v, "g", self._name)
                    self._g2 = self._zeros_slot(v, "g2", self._name)

    def _optimizer_step(self, grad, var, xi, g, g2):
        eps = 1e-16
        r_t = 1. / (xi + 1.)
        g_t = (1. - r_t) * g + r_t * grad
        g2_t = (1. - r_t) * g2 + r_t * grad**2
        var_t = var - grad * tf.minimum(g_t * g_t / (g2_t + eps), self._eps_t) / \
            (tf.sqrt(g2_t + eps) + eps)
        xi_t = 1 + xi * (1 - g_t * g_t / (g2_t + eps))
        return var_t, xi_t, g_t, g2_t

    def _apply_dense(self, grad, var):
        xi = self.get_slot(var, "xi")
        g = self.get_slot(var, "g")
        g2 = self.get_slot(var, "g2")
        var_t, xi_t, g_t, g2_t = self._optimizer_step(grad, var, xi, g, g2)

        # update helper variables
        xi_update = tf.assign(xi, xi_t, use_locking=self._use_locking)
        g_update = tf.assign(g, g_t, use_locking=self._use_locking)
        g2_update = tf.assign(g2, g2_t, use_locking=self._use_locking)
        var_update = tf.assign(
            var, var_t, use_locking=self._use_locking)

        all_updates = [xi_update, g_update, g2_update, var_update]
        return tf.group(*all_updates)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("apply_sparse not yet implemented")

    @property
    def eps_op(self):
        return self._eps_op, self._eps_placeholder

    def unroll(self, grads, var_list, opt_vars=None):
        next_opt_vars = []
        next_vars = []
        for i, (grad, var) in enumerate(zip(grads, var_list)):
            if opt_vars is None:
                xi = self.get_slot(var, "xi")
                g = self.get_slot(var, "g")
                g2 = self.get_slot(var, "g2")
            else:
                xi, g, g2 = opt_vars[i]
            var_t, xi_t, g_t, g2_t = self._optimizer_step(grad, var, xi, g, g2)
            next_opt_vars.append([xi_t, g_t, g2_t])
            next_vars.append(var_t)
        return next_vars, next_opt_vars


class AdagradDAOptimizer(optimizer.Optimizer):
    """Adagrad Dual Averaging algorithm for sparse linear models.
    See this [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
    This optimizer takes care of regularization of unseen features in a mini batch
    by updating them when they are seen with a closed form update rule that is
    equivalent to having updated them on every mini-batch.
    AdagradDA is typically used when there is a need for large sparsity in the
    trained model. This optimizer only guarantees sparsity for linear models. Be
    careful when using AdagradDA for deep networks as it will require careful
    initialization of the gradient accumulators for it to train.
    """

    def __init__(self,
                 learning_rate,
                 global_step,
                 initial_gradient_squared_accumulator_value=0.1,
                 l1_regularization_strength=0.0,
                 l2_regularization_strength=0.0,
                 use_locking=False,
                 name="AdagradDA"):
        """Construct a new AdagradDA optimizer.
        Args:
          learning_rate: A `Tensor` or a floating point value.  The learning rate.
          global_step: A `Tensor` containing the current training step number.
          initial_gradient_squared_accumulator_value: A floating point value.
            Starting value for the accumulators, must be positive.
          l1_regularization_strength: A float value, must be greater than or
            equal to zero.
          l2_regularization_strength: A float value, must be greater than or
            equal to zero.
          use_locking: If `True` use locks for update operations.
          name: Optional name prefix for the operations created when applying
            gradients.  Defaults to "AdagradDA".
        Raises:
          ValueError: If the `initial_gradient_squared_accumulator_value` is
          invalid.
        """
        if initial_gradient_squared_accumulator_value <= 0.0:
            raise ValueError("initial_gradient_squared_accumulator_value must be"
                             "positive: %s" %
                             initial_gradient_squared_accumulator_value)
        super(AdagradDAOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._initial_gradient_squared_accumulator_value = (
            initial_gradient_squared_accumulator_value)
        # Created in Initialize.
        self._learning_rate_tensor = None
        self._l1_regularization_strength = l1_regularization_strength
        self._l2_regularization_strength = l2_regularization_strength
        self._global_step = global_step

    def _create_slots(self, var_list):
        for v in var_list:
            with tf.Graph.colocate_with(v):
                g_val = tf.constant(
                    0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)
                gg_val = tf.constant(
                    self._initial_gradient_squared_accumulator_value,
                    shape=v.get_shape(),
                    dtype=v.dtype.base_dtype)
            self._get_or_make_slot(
                v, g_val, "gradient_accumulator", self._name)
            self._get_or_make_slot(v, gg_val, "gradient_squared_accumulator",
                                   self._name)

    def _prepare(self):
        self._learning_rate_tensor = tf.convert_to_tensor(
            self._learning_rate, name="learning_rate")

    def _apply_dense(self, grad, var):
        g_acc = self.get_slot(var, "gradient_accumulator")
        gg_acc = self.get_slot(var, "gradient_squared_accumulator")
        with tf.device(grad[0].device):
            global_step = tf.identity(self._global_step) + 1
        return training_ops.apply_adagrad_da(
            var,
            g_acc,
            gg_acc,
            grad,
            tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            tf.cast(self._l1_regularization_strength,
                    var.dtype.base_dtype),
            tf.cast(self._l2_regularization_strength,
                    var.dtype.base_dtype),
            global_step,
            use_locking=self._use_locking)

    def _resource_apply_dense(self, grad, var):
        g_acc = self.get_slot(var, "gradient_accumulator")
        gg_acc = self.get_slot(var, "gradient_squared_accumulator")
        with tf.device(grad[0].device):
            global_step = tf.identity(self._global_step) + 1
        return training_ops.resource_apply_adagrad_da(
            var.handle,
            g_acc.handle,
            gg_acc.handle,
            grad,
            tf.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
            tf.cast(self._l1_regularization_strength,
                    grad.dtype.base_dtype),
            tf.cast(self._l2_regularization_strength,
                    grad.dtype.base_dtype),
            global_step,
            use_locking=self._use_locking)

    def _apply_sparse(self, grad, var):
        g_acc = self.get_slot(var, "gradient_accumulator")
        gg_acc = self.get_slot(var, "gradient_squared_accumulator")
        with tf.device(grad[0].device):
            global_step = tf.identity(self._global_step) + 1
        return training_ops.sparse_apply_adagrad_da(
            var,
            g_acc,
            gg_acc,
            grad.values,
            grad.indices,
            tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            tf.cast(self._l1_regularization_strength,
                    var.dtype.base_dtype),
            tf.cast(self._l2_regularization_strength,
                    var.dtype.base_dtype),
            global_step,
            use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices):
        g_acc = self.get_slot(var, "gradient_accumulator")
        gg_acc = self.get_slot(var, "gradient_squared_accumulator")
        with tf.device(grad[0].device):
            global_step = tf.identity(self._global_step) + 1
        return training_ops.resource_sparse_apply_adagrad_da(
            var.handle,
            g_acc.handle,
            gg_acc.handle,
            grad,
            indices,
            tf.cast(self._learning_rate_tensor, grad.dtype),
            tf.cast(self._l1_regularization_strength, grad.dtype),
            tf.cast(self._l2_regularization_strength, grad.dtype),
            global_step,
            use_locking=self._use_locking)
