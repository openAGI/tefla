# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import pprint
import numpy as np

from tefla.da.iterator import BatchIterator
from tefla.core.lr_policy import NoDecayPolicy
import tefla.core.summary as summary
import tefla.core.logger as log


TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class Base(object):

    def __init__(self, model, cnf, training_iterator=BatchIterator(32, False),
                 validation_iterator=BatchIterator(128, False), start_epoch=1, resume_lr=0.01, classification=True, clip_norm=True, norm_threshold=5, n_iters_per_epoch=1094, gpu_memory_fraction=0.94, is_summary=False, log_file_name='/tmp/deepcnn.log', verbosity=0, loss_type='softmax_cross_entropy', weights_dir='weights'):
        self.model = model
        self.cnf = cnf
        self.training_iterator = training_iterator
        self.validation_iterator = validation_iterator
        self.classification = classification
        self.lr_policy = cnf.get('lr_policy', NoDecayPolicy(0.01))
        self.lr_policy.start_epoch = start_epoch
        self.lr_policy.base_lr = resume_lr
        self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
        self.validation_metrics_def = self.cnf.get('validation_scores', [])
        self.clip_norm = clip_norm
        self.norm_threshold = norm_threshold
        self.gradient_multipliers = None
        self.gpu_memory_fraction = gpu_memory_fraction
        self.is_summary = is_summary
        self.loss_type = loss_type
        self.weights_dir = weights_dir
        log.setFileHandler(log_file_name)
        log.setVerbosity(str(verbosity))

    def _setup_summaries(self, d_grads_and_var, g_grads_and_var=None):
        with tf.name_scope('summaries'):
            self.epoch_loss = tf.placeholder(
                tf.float32, shape=[], name="epoch_loss")

            # Training summaries
            tf.scalar_summary('learning rate', self.learning_rate,
                              collections=[TRAINING_EPOCH_SUMMARIES])
            tf.scalar_summary('training (cross entropy) loss', self.epoch_loss,
                              collections=[TRAINING_EPOCH_SUMMARIES])
            if g_grads_and_var is not None:
                self.epoch_loss_g = tf.placeholder(
                    tf.float32, shape=[], name="epoch_loss_g")
                tf.scalar_summary('training (cross entropy) loss', self.epoch_loss_g,
                                  collections=[TRAINING_EPOCH_SUMMARIES])
            if len(self.inputs.get_shape()) == 4:
                summary.summary_image(self.inputs, 'inputs', max_images=10, collections=[
                                      TRAINING_BATCH_SUMMARIES])
            for key, val in self.training_end_points.iteritems():
                summary.summary_activation(val, name=key, collections=[
                                           TRAINING_BATCH_SUMMARIES])
            summary.summary_trainable_params(['scalar', 'histogram', 'norm'], collections=[
                                             TRAINING_BATCH_SUMMARIES])
            summary.summary_gradients(d_grads_and_var, [
                                      'scalar', 'histogram', 'norm'], collections=[TRAINING_BATCH_SUMMARIES])
            if g_grads_and_var is not None:
                summary.summary_gradients(g_grads_and_var, [
                                          'scalar', 'histogram', 'norm'], collections=[TRAINING_BATCH_SUMMARIES])

            # Validation summaries
            for key, val in self.validation_end_points.iteritems():
                summary.summary_activation(val, name=key, collections=[
                                           VALIDATION_BATCH_SUMMARIES])

            tf.scalar_summary('validation loss', self.epoch_loss,
                              collections=[VALIDATION_EPOCH_SUMMARIES])
            self.validation_metric_placeholders = []
            for metric_name, _ in self.validation_metrics_def:
                validation_metric = tf.placeholder(
                    tf.float32, shape=[], name=metric_name.replace(' ', '_'))
                self.validation_metric_placeholders.append(validation_metric)
                tf.scalar_summary(metric_name, validation_metric,
                                  collections=[VALIDATION_EPOCH_SUMMARIES])
            self.validation_metric_placeholders = tuple(
                self.validation_metric_placeholders)

    def _optimizer(self, lr, optname='momentum', decay=0.9, momentum=0.9, epsilon=1e-08, beta1=0.9, beta2=0.999):
        """ definew the optimizer to use.

        Args:
            lr: learning rate, a scalar or a policy
            optname: optimizer name
            decay: variable decay value, scalar
            momentum: momentum value, scalar

        Returns:
            optimizer to use
         """
        if optname == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(
                learning_rate=lr, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
        if optname == 'adagrad':
            opt = tf.train.AdagradOptimizer(
                lr, initial_accumulator_value=0.1, use_locking=False, name='Adadelta')
        if optname == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(
                lr, decay=0.9, momentum=0.0, epsilon=epsilon)
        if optname == 'momentum':
            opt = tf.train.MomentumOptimizer(
                lr, momentum, use_locking=False, name='momentum', use_nesterov=True)
        if optname == 'adam':
            opt = tf.train.AdamOptimizer(
                learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, use_locking=False, name='Adam')
        return opt

    def _sigmoid_kl_with_logits(self, logits, targets):
        """ Sigmoid cross entropy with smooth labels

        Args:
            logits: logits
            targets: smooth targets

        Returns:
            cross entropy loss
        """

        assert isinstance(targets, float)
        if targets in [0., 1.]:
            entropy = 0.
        else:
            entropy = - targets * \
                np.log(targets) - (1. - targets) * np.log(1. - targets)
        return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy

    def _moving_averages_op(self):
        variable_averages = tf.train.ExponentialMovingAverage(
            self.cnf.get('MOVING_AVERAGE_DECAY', 0.999))
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        return variables_averages_op

    def _tensors_in_checkpoint_file(self, file_name, tensor_name=None, all_tensors=True):
        list_variables = []
        try:
            reader = tf.train.NewCheckpointReader(file_name)
            if all_tensors:
                var_to_shape_map = reader.get_variable_to_shape_map()
                for key in var_to_shape_map:
                    list_variables.append(key)
            else:
                list_variables.append(tensor_name)
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print(
                    "It's likely that your checkpoint file has been compressed with SNAPPY.")
        return list_variables

    def _load_weights(self, sess, weights_from):
        log.info("Loading session/weights from %s..." % weights_from)
        if weights_from:
            try:
                names_to_restore = self._tensors_in_checkpoint_file(
                    weights_from)
                variables_to_restore = []
                for v_name in names_to_restore:
                    try:
                        temp = [v for v in tf.all_variables() if v.name.strip(':0') == str(v_name)][
                            0]
                        variables_to_restore.append(temp)
                    except Exception, e:
                        log.info(
                            "Unable to get corect variables Error: %s." % e.message)
                        continue
                new_saver = tf.train.Saver(variables_to_restore)
                new_saver.restore(sess, weights_from)
                print("Loaded weights from %s" % weights_from)
            except ValueError:
                log.debug(
                    "Couldn't load weights from %s; starting from scratch" % weights_from)
                sess.run(tf.initialize_all_variables())
        else:
            sess.run(tf.initialize_all_variables())

    def _print_layer_shapes(self, end_points, log):
        log.info("\nModel layer output shapes:")
        for k, v in end_points.iteritems():
            log.info("%s - %s" % (k, v.get_shape()))

    def _clip_grad_norms(self, gradients_to_variables, max_norm=5):
        """Clips the gradients by the given value.

        Args:
            gradients_to_variables: A list of gradient to variable pairs (tuples).
            max_norm: the maximum norm value.

        Returns:
            A list of clipped gradient to variable pairs.
         """
        grads_and_vars = []
        for grad, var in gradients_to_variables:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    tmp = tf.clip_by_norm(grad.values, max_norm)
                    grad = tf.IndexedSlices(
                        tmp, grad.indices, grad.dense_shape)
                else:
                    grad = tf.clip_by_norm(grad, max_norm)
            grads_and_vars.append((grad, var))
        return grads_and_vars

    def _clip_grad_global_norms(self, tvars, loss, opt, global_norm=8, gate_gradients=1, gradient_noise_scale=None, GATE_GRAPH=2, grad_loss=None, agre_method=None, col_grad_ops=False):
        """Clips the gradients by the given value.

        Args:
            tvars: trainable variables used for gradint updates
            loss: total loss of the network
            opt: optimizer
            global_norm: the maximum global norm

        Returns:
            A list of clipped gradient to variable pairs.
         """
        var_refs = [v.ref() for v in tvars]
        grads = tf.gradients(loss, var_refs, grad_ys=grad_loss, gate_gradients=(
            gate_gradients == 1), aggregation_method=agre_method, colocate_gradients_with_ops=col_grad_ops)
        if gradient_noise_scale is not None:
            grads = self._add_scaled_noise_to_gradients(
                list(zip(grads, tvars)), gradient_noise_scale=gradient_noise_scale)
        if gate_gradients == GATE_GRAPH:
            grads = tf.tuple(grads)
        grads, _ = tf.clip_by_global_norm(grads, global_norm)
        grads_and_vars = list(zip(grads, tvars))
        return grads_and_vars

    def _multiply_gradients(self, grads_and_vars, gradient_multipliers):
        """Multiply specified gradients.

        Args:
            grads_and_vars: A list of gradient to variable pairs (tuples).
            gradient_multipliers: A map from either `Variables` or `Variable` op names
              to the coefficient by which the associated gradient should be scaled.

        Returns:
            The updated list of gradient to variable pairs.

        Raises:
            ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers`
            is empty or None or if `gradient_multipliers` is not a dictionary.
        """
        if not isinstance(grads_and_vars, list):
            raise ValueError('`grads_and_vars` must be a list.')
        if not gradient_multipliers:
            raise ValueError('`gradient_multipliers` is empty.')
        if not isinstance(gradient_multipliers, dict):
            raise ValueError('`gradient_multipliers` must be a dict.')

        multiplied_grads_and_vars = []
        for grad, var in grads_and_vars:
            if var in gradient_multipliers or var.op.name in gradient_multipliers:
                key = var if var in gradient_multipliers else var.op.name
                if grad is None:
                    raise ValueError('Requested multiple of `None` gradient.')

                if isinstance(grad, tf.IndexedSlices):
                    tmp = grad.values * \
                        tf.constant(gradient_multipliers[
                                    key], dtype=grad.dtype)
                    grad = tf.IndexedSlices(
                        tmp, grad.indices, grad.dense_shape)
                else:
                    grad *= tf.constant(gradient_multipliers[
                                        key], dtype=grad.dtype)
            multiplied_grads_and_vars.append((grad, var))
        return multiplied_grads_and_vars

    def _add_scaled_noise_to_gradients(self, grads_and_vars, gradient_noise_scale=10.0):
        """Adds scaled noise from a 0-mean normal distribution to gradients

        Args:
            grads_and_vars: list of gradient and variables
            gardient_noise_scale: value of noise factor

        Returns:
            noise added gradients

        Raises:
        ValueError: If `grads_and_vars` is not a list
        """
        if not isinstance(grads_and_vars, list):
            raise ValueError('`grads_and_vars` must be a list.')

        gradients, variables = zip(*grads_and_vars)
        noisy_gradients = []
        for gradient in gradients:
            if gradient is None:
                noisy_gradients.append(None)
                continue
            if isinstance(gradient, tf.IndexedSlices):
                gradient_shape = gradient.dense_shape
            else:
                gradient_shape = gradient.get_shape()
            noise = tf.truncated_normal(gradient_shape) * gradient_noise_scale
            noisy_gradients.append(gradient + noise)
        # return list(zip(noisy_gradients, variables))
        return noisy_gradients

    def _verbosity(self, verbosity, log):
        return{
            '0': log.DEBUG,
            '1': log.INFO,
            '2': log.WARN,
            '3': log.ERROR,
            '4': log.FATAL,
        }[verbosity]

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None and len(self.update_ops) == 0:
            self.update_ops = None
            # if update_ops is not None:
            #     regularized_training_loss = control_flow_ops.with_dependencies(update_ops, regularized_training_loss)

    def _print_info(self, data_set):
        log.info('Config:')
        log.info(pprint.pformat(self.cnf))
        data_set.print_info()
        log.info('Max epochs: %d' % self.num_epochs)
        all_vars = set(tf.all_variables())
        trainable_vars = set(tf.trainable_variables())
        non_trainable_vars = all_vars.difference(trainable_vars)

        log.info("\n---Trainable vars in model:")
        name_shapes = map(lambda v: (v.name, v.get_shape()), trainable_vars)
        for n, s in sorted(name_shapes, key=lambda ns: ns[0]):
            log.info('%s %s' % (n, s))

        log.info("\n---Non Trainable vars in model:")
        name_shapes = map(lambda v: (v.name, v.get_shape()),
                          non_trainable_vars)
        for n, s in sorted(name_shapes, key=lambda ns: ns[0]):
            log.info('%s %s' % (n, s))

        all_ops = tf.get_default_graph().get_operations()
        log.debug("\n---All ops in graph")
        names = map(lambda v: v.name, all_ops)
        for n in sorted(names):
            log.debug(n)

        self._print_layer_shapes(self.training_end_points, log)
