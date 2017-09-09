# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import

import pprint
import numpy as np
import os

from ..da.iterator import BatchIterator
from .lr_policy import NoDecayPolicy
from .losses import kappa_log_loss_clipped, segment_loss, dice_loss
from . import summary
from . import logger as log
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.training import moving_averages


TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class Base(object):

    def __init__(self, model, cnf, training_iterator=BatchIterator(32, False),
                 validation_iterator=BatchIterator(32, False), num_classes=5, start_epoch=1, resume_lr=0.01, classification=True, clip_norm=True, norm_threshold=5, n_iters_per_epoch=1094, gpu_memory_fraction=0.94, is_summary=False, log_file_name='/tmp/deepcnn.log', verbosity=1, loss_type='softmax_cross_entropy', label_smoothing=0.009, model_name='graph.pbtxt'):
        self.model = model
        self.model_name = model_name
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
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        log.setFileHandler(log_file_name)
        log.setVerbosity(str(verbosity))
        super(Base, self).__init__()

    def _setup_summaries(self, d_grads_and_var=None, input_summary=False, g_grads_and_var=None, activation_summary=False, params_summary=False, epoch_loss_g=False):
        with tf.name_scope('summaries'):
            self.epoch_loss = tf.placeholder(
                tf.float32, shape=[], name="epoch_loss")

            # Training summaries
            tf.summary.scalar('learning rate', self.learning_rate,
                              collections=[TRAINING_EPOCH_SUMMARIES])
            tf.summary.scalar('training (cross entropy) loss', self.epoch_loss,
                              collections=[TRAINING_EPOCH_SUMMARIES])
            if epoch_loss_g:
                self.epoch_loss_g = tf.placeholder(
                    tf.float32, shape=[], name="epoch_loss_g")
                tf.summary.scalar('training generator (cross entropy) loss', self.epoch_loss_g,
                                  collections=[TRAINING_EPOCH_SUMMARIES])
            if g_grads_and_var is not None:
                self.epoch_loss_g = tf.placeholder(
                    tf.float32, shape=[], name="epoch_loss_g")
                tf.summary.scalar('training (cross entropy) loss', self.epoch_loss_g,
                                  collections=[TRAINING_EPOCH_SUMMARIES])
            if input_summary:
                if len(self.inputs.get_shape()) == 4:
                    summary.summary_image(self.inputs, 'inputs', max_images=10, collections=[
                        TRAINING_BATCH_SUMMARIES])
            if activation_summary:
                for key, val in self.training_end_points.iteritems():
                    summary.summary_activation(val, name=key, collections=[
                                               TRAINING_BATCH_SUMMARIES])
            if params_summary:
                summary.summary_trainable_params(['scalar', 'histogram', 'norm'], collections=[
                                                 TRAINING_BATCH_SUMMARIES])
            if d_grads_and_var is not None:
                summary.summary_gradients(d_grads_and_var, [
                                          'scalar', 'histogram', 'norm'], collections=[TRAINING_BATCH_SUMMARIES])
            if g_grads_and_var is not None:
                summary.summary_gradients(g_grads_and_var, [
                                          'scalar', 'histogram', 'norm'], collections=[TRAINING_BATCH_SUMMARIES])

            # Validation summaries
            if activation_summary:
                for key, val in self.validation_end_points.iteritems():
                    summary.summary_activation(val, name=key, collections=[
                                               VALIDATION_BATCH_SUMMARIES])

            tf.summary.scalar('validation loss', self.epoch_loss,
                              collections=[VALIDATION_EPOCH_SUMMARIES])
            self.validation_metric_placeholders = []
            for metric_name, _ in self.validation_metrics_def:
                validation_metric = tf.placeholder(
                    tf.float32, shape=[], name=metric_name.replace(' ', '_'))
                self.validation_metric_placeholders.append(validation_metric)
                tf.summary.scalar(metric_name, validation_metric,
                                  collections=[VALIDATION_EPOCH_SUMMARIES])
            self.validation_metric_placeholders = tuple(
                self.validation_metric_placeholders)

    def _optimizer(self, lr, optname='momentum', decay=0.9, momentum=0.9, epsilon=1e-08, beta1=0.5, beta2=0.999, l1_reg=0.0, l2_reg=0.0, accum_val=0.1, lr_power=-0.5):
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
        if optname == 'proximalgd':
            opt = tf.train.ProximalGradientDescentOptimizer(
                lr, l1_regularization_strength=l1_reg, l2_regularization_strength=l2_reg, use_locking=False, name='ProximalGradientDescent')
        if optname == 'proximaladagrad':
            opt = tf.train.ProximalAdagradOptimizer(lr, initial_accumulator_value=accum_val, l1_regularization_strength=l1_reg,
                                                    l2_regularization_strength=l2_reg, use_locking=False, name='ProximalGradientDescent')
        if optname == 'ftrl':
            opt = tf.train.FtrlOptimizer(lr, learning_rate_power=lr_power, initial_accumulator_value=accum_val,
                                         l1_regularization_strength=l1_reg, l2_regularization_strength=l2_reg, use_locking=False, name='Ftrl')
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
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * targets) - entropy

    def _moving_averages_op(self):
        variable_averages = tf.train.ExponentialMovingAverage(
            self.cnf.get('MOVING_AVERAGE_DECAY', 0.999))
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        return variables_averages_op

    def _tensors_in_checkpoint_file(self, file_name, tensor_name=None, all_tensors=True):
        try:
            reader = tf.train.NewCheckpointReader(file_name)
            if all_tensors:
                var_to_shape_map = reader.get_variable_to_shape_map()
                list_variables = var_to_shape_map.keys()
            else:
                list_variables = [tensor_name]
        except Exception as e:  # pylint: disable=broad-except
            if "corrupted compressed block contents" in str(e):
                log.debug(
                    "It's likely that your checkpoint file has been compressed with SNAPPY.")
        return list_variables

    def _load_weights(self, sess, saver, weights_from):
        log.info("Loading session/weights from %s..." % weights_from)
        try:
            saver.restore(sess, weights_from)
        except Exception as e:
            log.info("Partial restoring session.")
            try:
                names_to_restore = self._tensors_in_checkpoint_file(
                    weights_from)
                variables_to_restore = []
                for v_name in names_to_restore:
                    try:
                        temp = [v for v in tf.global_variables() if v.name.strip(':0') == str(v_name)][
                            0]
                        variables_to_restore.append(temp)
                    except Exception:
                        log.info(
                            'Variable %s doesnt exist in new model' % v_name)
                for var in variables_to_restore:
                    log.info("Loading: %s %s)" %
                             (var.name, var.get_shape()))
                    restorer = tf.train.Saver([var])
                    try:
                        restorer.restore(sess, weights_from)
                    except Exception as e:
                        log.info("Problem loading: %s -- %s" %
                                 (var.name, e.message))
                        continue
                log.info("Loaded weights from %s" % weights_from)
            except ValueError:
                log.debug(
                    "Couldn't load weights from %s; starting from scratch" % weights_from)
                sess.run(tf.global_variables_initializer())

    def _print_layer_shapes(self, end_points, log):
        log.info("\nModel layer output shapes:")
        for k, v in end_points.iteritems():
            log.info("%s - %s" % (k, v.get_shape()))

    def _adjust_ground_truth(self, y):
        if self.loss_type == 'kappa_log':
            return np.eye(self.num_classes)[y]
        else:
            return y if self.classification else y.reshape(-1, 1).astype(np.float32)

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            try:
                grad = tf.concat(grads, 0)
            except Exception:
                grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

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
        var_refs = [v.read_value() for v in tvars]
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
            if (grad is not None and (var in gradient_multipliers or var.name in gradient_multipliers)):
                key = var if var in gradient_multipliers else var.name
                multiplier = tf.constant(
                    gradient_multipliers[key], dtype=tf.float32)
                if isinstance(grad, tf.IndexedSlices):
                    grad_values = grad.values * multiplier
                    grad = tf.IndexedSlices(
                        grad_values, grad.indices, grad.dense_shape)
                else:
                    grad *= multiplier
            multiplied_grads_and_vars.append((grad, var))
        return multiplied_grads_and_vars

    def _scale_gradient(self, layer_grad, scale, name="scale_gradient"):
        """Scales gradients for the backwards pass.
        This might be used to, for example, allow one part of a model to learn at a
        lower rate than the rest.
        WARNING: Think carefully about how your optimizer works. If, for example, you
        use rmsprop, the gradient is always rescaled (with some additional epsilon)
        towards unity. This means `scale_gradient` won't have the effect of
        lowering the learning rate.
        If `scale` is `0.0`, this op reduces to `tf.stop_gradient`. If `scale`
        is `1.0`, this op reduces to `tf.identity`.

        Args:
          layer_grad: A `tf.Tensor`.
          scale: The scale factor for the gradient on the backwards pass.
          name: A name for the operation (optional).

        Returns:
          A `tf.Tensor` with the same type as the input tensor.
        """
        if scale == 0.0:
            return tf.stop_gradient(layer_grad, name=name)
        elif scale == 1.0:
            return tf.identity(layer_grad, name=name)
        else:
            scale_tensor = tf.convert_to_tensor(scale)

            @function.Defun(tf.float32, tf.float32,
                            python_grad_func=lambda op, g: (
                                g * op.inputs[1], None),
                            func_name="ScaleGradient")
            def gradient_scaler(x, unused_scale):
                return x

            output = gradient_scaler(
                layer_grad, scale_tensor, name=name)
            output.set_shape(net.get_shape())

        return output

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

    def _adaptive_max_norm(self, norm, std_factor, decay, global_step, epsilon, name):
        """Find max_norm given norm and previous average."""
        with tf.variable_scope(name, "AdaptiveMaxNorm", [norm]):
            log_norm = tf.log(norm + epsilon)

            def moving_average(name, value, decay):
                moving_average_variable = tf.get_variable(name,
                                                          shape=value.get_shape(),
                                                          dtype=value.dtype,
                                                          initializer=tf.zeros_initializer(),
                                                          trainable=False)
                return moving_averages.assign_moving_average(moving_average_variable, value, decay, zero_debias=False)

            # quicker adaptation at the beginning
            if global_step is not None:
                n = tf.to_float(global_step)
                decay = tf.minimum(decay, n / (n + 1.))

            # update averages
            mean = moving_average("mean", log_norm, decay)
            sq_mean = moving_average(
                "sq_mean", tf.square(log_norm), decay)

            variance = sq_mean - tf.square(mean)
            std = tf.sqrt(tf.maximum(epsilon, variance))
            max_norms = tf.exp(mean + std_factor * std)
            return max_norms, mean

    def _adaptive_gradient_clipping(self, grads_and_vars, std_factor=2., decay=0.95, static_max_norm=None, global_step=None, epsilon=1e-8, name=None):
        """function for adaptive gradient clipping."""
        grads, variables = zip(*grads_and_vars)
        norm = tf.global_norm(grads)
        max_norm, log_mean = self._adaptive_max_norm(norm, std_factor, decay,
                                                     global_step, epsilon, name)

        # factor will be 1. if norm is smaller than max_norm
        factor = tf.where(norm < max_norm,
                          tf.ones_like(norm),
                          tf.exp(log_mean) / norm)

        if static_max_norm is not None:
            factor = tf.minimum(static_max_norm / norm, factor)

        # apply factor
        clipped_grads = []
        for grad in grads:
            if grad is None:
                clipped_grads.append(None)
            elif isinstance(grad, tf.IndexedSlices):
                clipped_grads.append(tf.IndexedSlices(grad.values * factor, grad.indices,
                                                      grad.dense_shape))
            else:
                clipped_grads.append(grad * factor)

        return list(zip(clipped_grads, variables))

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None and len(self.update_ops) == 0:
            self.update_ops = None
            # if update_ops is not None:
            #     regularized_training_loss = control_flow_ops.with_dependencies(update_ops, regularized_training_loss)

    def _print_info(self, data_set=None):
        log.info('Config:')
        log.info(pprint.pformat(self.cnf))
        try:
            data_set.print_info()
        except Exception:
            log.info('No Dataset info found')
        log.info('Max epochs: %d' % self.num_epochs)
        all_vars = set(tf.global_variables())
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
        try:
            self._print_layer_shapes(self.training_end_points, log)
        except Exception:
            log.info('Multi GPU setup')

    def total_network_params(self):
        def variable_params(v):
            return reduce(
                lambda x, y: x * y, v.get_shape().as_list())
        n = sum(variable_params(v) for v in tf.trainable_variables())
        print("Number of trainable network params: %dK" % (n / 1000,))

    def write_params(self):
        opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
        opts['dump_to_file'] = os.path.abspath(
            self.cnf.get('model_params_file', '/tmp/graph_params.log'))
        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(), tfprof_options=opts)

        with tf.gfile.GFile(self.cnf.get('model_params_file', '/tmp/graph_params.log')) as file:
            tf.logging.info(file.read())

    def write_graph(self, graph_def, output_dir='/tmp'):
        log.info('Writing model graph .pbtxt to %s' % output_dir)
        tf.train.write_graph(graph_def, output_dir, self.model_name)


class BaseMixin(object):

    def __init__(self, label_smoothing=0.009):
        self.label_smoothing = label_smoothing
        super(BaseMixin, self).__init__()

    def _loss_regression(self, logits, labels, is_training):
        labels = tf.cast(labels, tf.int64)
        sq_loss = tf.square(tf.sub(logits, labels), name='regression loss')
        sq_loss_mean = tf.reduce_mean(sq_loss, name='regression')
        if is_training:
            tf.add_to_collection('losses', sq_loss_mean)

            l2_loss = tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))
            l2_loss = l2_loss * self.cnf.get('l2_reg', 0.0)
            tf.add_to_collection('losses', l2_loss)

            return tf.add_n(tf.get_collection('losses'), name='total_loss')
        else:
            return sq_loss_mean

    def _loss_softmax(self, logits, labels, is_training):
        log.info('Using softmax loss')
        labels = tf.cast(labels, tf.int64)
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_loss')
        ce_loss_mean = tf.reduce_mean(ce_loss, name='cross_entropy')
        if is_training:
            tf.add_to_collection('losses', ce_loss_mean)

            l2_loss = tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))
            l2_loss = l2_loss * self.cnf.get('l2_reg', 0.0)
            tf.add_to_collection('losses', l2_loss)

            return tf.add_n(tf.get_collection('losses'), name='total_loss')
        else:
            return ce_loss_mean

    def _loss_dice(self, predictions, labels, is_training):
        log.info('Using DICE loss')
        labels = tf.cast(labels, tf.int64)
        num_classes = predictions.get_shape().as_list()[-1]
        labels = tf.one_hot(labels, num_classes)
        dc_loss = dice_loss(predictions, labels)
        dc_loss_mean = tf.reduce_mean(dc_loss, name='dice_loss_')
        if is_training:
            tf.add_to_collection('losses', dc_loss_mean)

            l2_loss = tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))
            l2_loss = l2_loss * self.cnf.get('l2_reg', 0.0)
            tf.add_to_collection('losses', l2_loss)

            return tf.add_n(tf.get_collection('losses'), name='total_loss')
        else:
            return dc_loss_mean

    def _loss_kappa(self, predictions, labels, is_training, y_pow=2):
        log.info('Using KAPPA loss')
        labels = tf.cast(labels, tf.int64)
        if is_training:
            kappa_loss = kappa_log_loss_clipped(
                predictions, labels, y_pow=y_pow, label_smoothing=self.label_smoothing, batch_size=self.cnf['batch_size_train'])
            tf.add_to_collection('losses', kappa_loss)
            l2_loss = tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))
            l2_loss = l2_loss * self.cnf.get('l2_reg', 0.0)
            tf.add_to_collection('losses', l2_loss)

            return tf.add_n(tf.get_collection('losses'), name='total_loss')
        else:
            kappa_loss = kappa_log_loss_clipped(
                predictions, labels, batch_size=self.cnf['batch_size_test'])
            return kappa_loss

    def _tower_loss(self, scope, model, images, labels, is_training, reuse, loss_type='kappa_log', y_pow=2, is_classification=True, gpu_id=0):
        if is_training:
            self.training_end_points = model(
                images, is_training=is_training, reuse=reuse, num_classes=self.num_classes)
            if is_classification:
                if loss_type == 'kappa_log':
                    loss_temp = self._loss_kappa(
                        self.training_end_points['predictions'], labels, is_training, y_pow=y_pow)
                elif loss_type == 'dice_loss':
                    loss_temp = self._loss_dice(
                        self.training_end_points['predictions'], labels, is_training)
                else:
                    loss_temp = self._loss_softmax(self.training_end_points[
                        'logits'], labels, is_training)
            else:
                loss_temp = self._loss_regression(self.training_end_points[
                    'logits'], labels, is_training)
            losses = tf.get_collection('losses', scope)
            total_loss = tf.add_n(losses, name='total_loss')
            if gpu_id == 0:
                self._print_layer_shapes(self.training_end_points, log)
        else:
            self.validation_end_points = model(
                images, is_training=is_training, reuse=reuse, num_classes=self.num_classes)
            if is_classification:
                if loss_type == 'kappa_log':
                    loss = self._loss_kappa(self.validation_end_points[
                                            'predictions'], labels, is_training)
                else:
                    loss = self._loss_softmax(self.validation_end_points[
                        'logits'], labels, is_training)
            else:
                loss = self._loss_regression(self.validation_end_points[
                    'logits'], labels, is_training)
            validation_predictions = self.validation_end_points['predictions']
            total_loss = {'loss': loss, 'predictions': validation_predictions}

        return total_loss

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
