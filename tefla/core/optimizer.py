import numpy as np
import numbers
import tensorflow as tf
from tensorflow import random_normal, shape
from tensorflow.python.training import optimizer
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops


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
            self._eps_t = ops.convert_to_tensor(
                self._learning_rate, name="epsilon")
        else:
            self._eps_placeholder = tf.placeholder(
                self._learning_rate.dtype, [1], name='eps_placeholder')
            self._eps_op = state_ops.assign(
                self._learning_rate, self._eps_placeholder)
            self._eps_t = self._learning_rate

    def _create_slots(self, var_list):
        if self._g2 is None:
            with ops.colocate_with(var_list[0]):
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
        xi_update = state_ops.assign(xi, xi_t, use_locking=self._use_locking)
        g_update = state_ops.assign(g, g_t, use_locking=self._use_locking)
        g2_update = state_ops.assign(g2, g2_t, use_locking=self._use_locking)
        var_update = state_ops.assign(
            var, var_t, use_locking=self._use_locking)

        all_updates = [xi_update, g_update, g2_update, var_update]
        return control_flow_ops.group(*all_updates)

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
