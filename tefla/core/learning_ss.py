# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import

import os
import time
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from . import logger as log
from . import summary as summary
from .base import Base
from ..utils import util


TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class SemiSupervisedTrainer(Base):
    """
    Semi Supervised Trainer


    Args:
        model: model definition
        cnf: dict, training configs
        training_iterator: iterator to use for training data access, processing and augmentations
        validation_iterator: iterator to use for validation data access, processing and augmentations
        start_epoch: int, training start epoch; for resuming training provide the last
        epoch number to resume training from, its a required parameter for training data balancing
        resume_lr: float, learning rate to use for new training
        classification: bool, classificattion or regression
        clip_norm: bool, to clip gradient using gradient norm, stabilizes the training
        n_iters_per_epoch: int,  number of iteratiosn for each epoch;
            e.g: total_training_samples/batch_size
        gpu_memory_fraction: amount of gpu memory to use
        is_summary: bool, to write summary or not
    """

    def __init__(self, model, cnf, clip_by_global_norm=False, **kwargs):
        self.clip_by_global_norm = clip_by_global_norm
        super(SemiSupervisedTrainer, self).__init__(model, cnf, **kwargs)

    def fit(self, data_set, num_classes=6, weights_from=None, start_epoch=1, summary_every=199, model_name='multiclass_ss', weights_dir='weights'):
        """
        Train the model on the specified dataset

        Args:
            data_set: dataset instance to use to access data for training/validation
            weights_from: str, if not None, initializes model from exisiting weights
            start_epoch: int,  epoch number to start training from
                e.g. for retarining set the epoch number you want to resume training from
            summary_every: int, epoch interval to write summary; higher value means lower frequency
                of summary writing
        """
        with tf.Graph().as_default(), tf.device('/gpu:0'):
            self._setup_model_loss(num_classes=num_classes)
            if self.is_summary:
                self._setup_summaries(self.capped_d_grads, self.capped_g_grads)
            self._setup_misc()
            self._print_info(data_set)
            self._train_semi_supervised(
                data_set, start_epoch, weights_from, summary_every, model_name, weights_dir)

    def _train_semi_supervised(self, dataset, start_epoch, weights_from, summary_every, model_name, weights_dir):
        training_X, training_y, validation_X, validation_y = \
            dataset.training_X, dataset.training_y, dataset.validation_X, dataset.validation_y
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
        if not os.path.exists(weights_dir + '/best_models'):
            os.mkdir(weights_dir + '/best_models')

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=None)
        if self.is_summary:
            training_batch_summary_op = tf.merge_all_summaries(
                key=TRAINING_BATCH_SUMMARIES)
            training_epoch_summary_op = tf.merge_all_summaries(
                key=TRAINING_EPOCH_SUMMARIES)
            validation_batch_summary_op = tf.merge_all_summaries(
                key=VALIDATION_BATCH_SUMMARIES)
            validation_epoch_summary_op = tf.merge_all_summaries(
                key=VALIDATION_EPOCH_SUMMARIES)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.cnf.get('gpu_memory_fraction', 0.9))
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, gpu_options=gpu_options))
        sess.run(init)
        if start_epoch > 1:
            weights_from = "weights/model-epoch-%d.ckpt" % (
                start_epoch - 1)

        if weights_from:
            self._load_weights(sess, saver, weights_from)

        learning_rate_value = self.lr_policy.initial_lr
        log.info("Initial learning rate: %f " % learning_rate_value)
        if self.is_summary:
            train_writer, validation_writer = summary.create_summary_writer(
                self.cnf.get('summary_dir', '/tmp/tefla-summary'), sess)
        # keep track of maximum accuracy and auroc and save corresponding
        # weights
        training_history = []
        seed_delta = 100
        batch_iter_idx = 1
        n_iters_per_epoch = len(
            dataset.training_X) // self.training_iterator.batch_size
        self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
        for epoch in xrange(start_epoch, self.cnf.get('mum_epochs', 550) + 1):
            np.random.seed(epoch + seed_delta)
            tf.set_random_seed(epoch + seed_delta)
            tic = time.time()
            d_train_losses = []
            g_train_losses = []
            batch_train_sizes = []
            for batch_num, (Xb, yb) in enumerate(self.training_iterator(training_X, training_y)):
                if Xb.shape[0] < self.cnf['batch_size_train']:
                    continue
                feed_dict_train = {self.inputs: Xb,
                                   self.labels: yb, self.learning_rate_d: learning_rate_value, self.learning_rate_g: learning_rate_value}
                log.debug('1. Loading batch %d data done.' % batch_num)
                if epoch % summary_every == 0 and self.is_summary:
                    log.debug('2. Running training steps with summary...')
                    _, _d_loss_real, _d_loss_fake, _d_loss_class, summary_str_train = sess.run(
                        [self.train_op_d, self.d_loss_real, self.d_loss_fake, self.d_loss_class, training_batch_summary_op], feed_dict=feed_dict_train)
                    _, _g_loss = sess.run([self.train_op_g, self.g_losses[
                                          0]], feed_dict=feed_dict_train)
                    train_writer.add_summary(summary_str_train, epoch)
                    train_writer.flush()
                    log.debug(
                        '2. Running training steps with summary done.')
                    log.info("Epoch %d, Batch %d D_loss_real: %s, D_loss_fake: %s,D_loss_class: %s, G_loss: %s" % (
                        epoch, batch_num, _d_loss_real, _d_loss_fake, _d_loss_class, _g_loss))
                else:
                    log.debug(
                        '2. Running training steps without summary...')
                    _, _d_loss_real, _d_loss_fake, _d_loss_class = sess.run(
                        [self.train_op_d, self.d_loss_real, self.d_loss_fake, self.d_loss_class], feed_dict=feed_dict_train)
                    _, _g_loss = sess.run([self.train_op_g, self.g_losses[
                                          0]], feed_dict=feed_dict_train)
                    log.debug(
                        '2. Running training steps without summary done.')

                d_train_losses.append(
                    _d_loss_real + _d_loss_fake + _d_loss_class)
                g_train_losses.append(_g_loss)
                batch_train_sizes.append(len(Xb))
                learning_rate_value = self.lr_policy.batch_update(
                    learning_rate_value, batch_iter_idx)
                batch_iter_idx += 1
                log.debug('4. Training batch %d done.' % batch_num)
            d_avg_loss = np.average(
                d_train_losses, weights=batch_train_sizes)
            g_avg_loss = np.average(
                g_train_losses, weights=batch_train_sizes)
            log.info("Epoch %d, D_avg_loss: %s, G_avg_loss %s" %
                     (epoch, d_avg_loss, g_avg_loss))
            # Plot training loss every epoch
            log.debug('5. Writing epoch summary...')
            if self.is_summary:
                summary_str_train = sess.run(training_epoch_summary_op, feed_dict={
                                             self.epoch_loss: d_avg_loss, self.epoch_loss_g: g_avg_loss, self.learning_rate_d: learning_rate_value, self.learning_rate_g: learning_rate_value})
                train_writer.add_summary(summary_str_train, epoch)
                train_writer.flush()
            log.debug('5. Writing epoch summary done.')
            # Validation prediction and metrics
            validation_losses = []
            batch_validation_metrics = [[]
                                        for _, _ in self.validation_metrics_def]
            epoch_validation_metrics = []
            batch_validation_sizes = []
            for batch_num, (validation_Xb, validation_y_true) in enumerate(self.validation_iterator(validation_X, validation_y)):
                feed_dict_val = {self.inputs: validation_Xb,
                                 self.labels: validation_y_true}
                log.debug(
                    '6. Loading batch %d validation data done.' % batch_num)
                if (epoch - 1) % summary_every == 0 and self.is_summary:
                    log.debug(
                        '7. Running validation steps with summary...')
                    validation_y_pred, _val_loss, summary_str_validation = sess.run(
                        [self.predictions, self.test_loss, validation_batch_summary_op], feed_dict=feed_dict_val)

                    validation_writer.add_summary(
                        summary_str_validation, epoch)
                    validation_writer.flush()
                    log.debug(
                        '7. Running validation steps with summary done.')
                    log.debug(
                        "Epoch %d, Batch %d validation loss: %s" % (epoch, batch_num, _val_loss))
                    log.debug("Epoch %d, Batch %d validation predictions: %s" % (
                        epoch, batch_num, validation_y_pred))
                else:
                    log.debug(
                        '7. Running validation steps without summary...')
                    validation_y_pred, _val_loss = sess.run(
                        [self.predictions, self.test_loss], feed_dict=feed_dict_val)

                    log.debug(
                        '7. Running validation steps without summary done.')
                validation_losses.append(_val_loss)
                batch_validation_sizes.append(len(validation_Xb))
                for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                    metric_score = metric_function(
                        validation_y_true, validation_y_pred)
                    batch_validation_metrics[i].append(metric_score)
                log.debug('8. Validation batch %d done' % batch_num)

            epoch_validation_loss = np.average(
                validation_losses, weights=batch_validation_sizes)
            for i, (_, _) in enumerate(self.validation_metrics_def):
                epoch_validation_metrics.append(
                    np.average(batch_validation_metrics[i], weights=batch_validation_sizes))
            log.debug('9. Writing epoch validation summary...')
            if self.is_summary:
                summary_str_validate = sess.run(validation_epoch_summary_op, feed_dict={
                                                self.epoch_loss: epoch_validation_loss, self.validation_metric_placeholders: epoch_validation_metrics})
                validation_writer.add_summary(summary_str_validate, epoch)
                validation_writer.flush()
            log.debug('9. Writing epoch validation summary done.')

            custom_metrics_string = [', %s: %.3f' % (name, epoch_validation_metrics[i]) for i, (name, _) in
                                     enumerate(self.validation_metrics_def)]
            custom_metrics_string = ''.join(custom_metrics_string)

            log.info(
                "Epoch %d [(%s, %s) images, %6.1fs]: t-loss: %.3f, v-loss: %.3f%s" %
                (epoch, np.sum(batch_train_sizes), np.sum(batch_validation_sizes), time.time() - tic,
                 d_avg_loss,
                 epoch_validation_loss,
                 custom_metrics_string)
            )
            epoch_info = dict(
                epoch=epoch,
                training_loss=d_avg_loss,
                validation_loss=epoch_validation_loss
            )

            training_history.append(epoch_info)
            saver.save(sess, "%s/model-epoch-%d.ckpt" % (weights_dir, epoch))

            learning_rate_value = self.lr_policy.epoch_update(
                learning_rate_value, training_history)
            log.info("Current learning rate: %f " % learning_rate_value)
            end_points_G_val = self.model.generator(
                [self.cnf['batch_size_test'], 100], False, True, batch_size=self.cnf['batch_size_test'])

            util.save_images('generated_images.jpg',
                             sess.run(end_points_G_val['softmax']), width=128, height=128)

            G = sess.run(end_points_G_val['softmax'])
            cv2.imwrite('generated_image.jpg', G[0, :, :, :] * 50 + 128)

        if self.is_summary:
            train_writer.close()
            validation_writer.close()

    def _feature_matching_loss(self, real_data_features, fake_data_features):
        real_data_mean = tf.reduce_mean(
            real_data_features, axis=0)
        fake_data_mean = tf.reduce_mean(
            fake_data_features, axis=0)
        feature_loss = tf.reduce_mean(
            tf.abs(tf.subtract(real_data_mean, fake_data_mean)))

        return feature_loss

    def _tower_loss_semi_supervised(self, inputs, targets, gpu_idx=0, num_classes=11, is_fm_loss=False):
        with tf.variable_scope("train_specific"):
            avg_error_rate = tf.get_variable(
                'avg_error_rate', [], initializer=tf.constant_initializer(0.), trainable=False)
            num_error_rate = tf.get_variable(
                'num_error_rate', [], initializer=tf.constant_initializer(0.), trainable=False)

        batch_size_train = self.cnf['batch_size_train']
        batch_size_val = self.cnf['batch_size_test']
        self.end_points_G = self.model.generator(
            [batch_size_train, 100], True, None, batch_size_val)

        if gpu_idx == 0:
            G_means = tf.reduce_mean(
                self.end_points_G['softmax'], 0, keep_dims=True)
            G_vars = tf.reduce_mean(
                tf.square(self.end_points_G['softmax'] - G_means), 0, keep_dims=True)
            G = tf.Print(self.end_points_G['softmax'], [tf.reduce_mean(G_means), tf.reduce_mean(
                G_vars)], "generator mean and average var", first_n=1)
            inputs_means = tf.reduce_mean(inputs, 0, keep_dims=True)
            inputs_vars = tf.reduce_mean(
                tf.square(inputs - inputs_means), 0, keep_dims=True)
            inputs = tf.Print(inputs, [tf.reduce_mean(inputs_means), tf.reduce_mean(
                inputs_vars)], "image mean and average var", first_n=1)

        joint = tf.concat([inputs, G], 0)
        log.info('Input size of unlabelled and generated %s' %
                 (joint.get_shape()))
        self.end_points_D = self.model.discriminator(
            joint, True, None, num_classes=num_classes, batch_size=batch_size_train)
        self.end_points_D_val = self.model.discriminator(
            inputs, False, True, num_classes=num_classes, batch_size=batch_size_val)

        # For printing layers shape
        self.training_end_points = self.end_points_D
        self.training_end_points.update(self.end_points_G)

        tf.summary.histogram("d", self.end_points_D['D_on_data'])
        tf.summary.histogram("d_", self.end_points_D['D_on_G'])
        tf.summary.image("G", G)

        d_label_smooth = self.cnf['d_label_smooth']  # 0.25
        self.d_loss_real = self._sigmoid_kl_with_logits(
            self.end_points_D['D_on_data_logits'], 1. - d_label_smooth)
        class_loss_weight = 1.
        self.d_loss_class = class_loss_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.end_points_D['class_logits'], labels=tf.to_int64(targets))
        self.test_loss = 1. - \
            tf.reduce_mean(tf.to_float(tf.nn.in_top_k(
                self.end_points_D_val['logits'], targets, 1)))
        self.error_rate = 1. - \
            tf.reduce_mean(tf.to_float(tf.nn.in_top_k(
                self.end_points_D['class_logits'], targets, 1)))
        if gpu_idx == 0:
            update = tf.assign(num_error_rate, num_error_rate + 1.)
            with tf.control_dependencies([update]):
                tc = tf.maximum(.01, 1. / num_error_rate)
            update = tf.assign(avg_error_rate, (1. - tc) *
                               avg_error_rate + tc * self.error_rate)
            with tf.control_dependencies([update]):
                # d_loss_class = tf.Print(d_loss_class, [avg_error_rate], "running top-1 error rate")
                self.d_loss_class = tf.identity(self.d_loss_class)
        self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.end_points_D['D_on_G_logits'], labels=tf.zeros_like(self.end_points_D['D_on_G_logits']))
        self.d_loss_class = tf.reduce_mean(self.d_loss_class)
        self.d_loss_real = tf.reduce_mean(self.d_loss_real)
        self.d_loss_fake = tf.reduce_mean(self.d_loss_fake)

        if is_fm_loss:
            global_pool_head = self.end_points_D['global_pool']
            real_data_features = tf.slice(global_pool_head, [0, 0], [
                                          batch_size_train, num_classes])
            fake_data_features = tf.slice(global_pool_head, [batch_size_train, 0], [
                                          batch_size_train, num_classes])
            self.g_loss = self._feature_matching_loss(
                real_data_features, fake_data_features)
        else:
            generator_target_prob = self.cnf[
                'generator_target_prob']  # 0.75 / 2.0
            self.g_loss = self._sigmoid_kl_with_logits(
                self.end_points_D['D_on_G_logits'], generator_target_prob)
            self.g_loss = tf.reduce_mean(self.g_loss)

        if gpu_idx == 0:
            self.g_losses = []
        self.g_losses.append(self.g_loss)

        self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_class
        if gpu_idx == 0:
            self.d_loss_reals = []
            self.d_loss_fakes = []
            self.d_loss_classes = []
            self.d_losses = []
        self.d_loss_reals.append(self.d_loss_real)
        self.d_loss_fakes.append(self.d_loss_fake)
        self.d_loss_classes.append(self.d_loss_class)
        self.d_losses.append(self.d_loss)
        self.predictions = self.end_points_D_val['predictions']

    def _get_vars_semi_supervised(self):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('d_')]
        g_vars = [var for var in t_vars if var.name.startswith('g_')]
        for x in d_vars:
            assert x not in g_vars
        for x in g_vars:
            assert x not in d_vars
        for x in t_vars:
            assert x in g_vars or x in d_vars

        return {'d_vars': d_vars, 'g_vars': g_vars}

    def sigmoid_kl_with_logits(logits, targets):
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
            entropy = - targets * np.log(targets) - \
                (1. - targets) * np.log(1. - targets)
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits) * targets, logits=logits) - entropy

    def _setup_model_loss(self, update_ops=None, num_classes=6):
        self.learning_rate_d = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder")
        self.learning_rate_g = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder")

        d_optimizer = self._optimizer(self.learning_rate_d, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        g_optimizer = self._optimizer(self.learning_rate_g, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        # Get images and labels for ImageNet and split the batch across GPUs.
        assert self.cnf['batch_size_train'] % self.cnf.get(
            'num_gpus', 1) == 0, ('Batch size must be divisible by number of GPUs')

        self.inputs = tf.placeholder(tf.float32, shape=(
            None, self.model.image_size[0], self.model.image_size[0], 3), name="input")
        self.labels = tf.placeholder(tf.int32, shape=(None,))

        self._tower_loss_semi_supervised(
            self.inputs, self.labels, num_classes=num_classes, is_fm_loss=True)

        # global_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        global_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops is None:
            update_ops = global_update_ops
        else:
            update_ops = set(update_ops)
        # Make sure update_ops are computed before total_loss.
        if update_ops:
            with tf.control_dependencies(update_ops):
                barrier = tf.no_op(name='update_barrier')
                self.d_losses[-1] = control_flow_ops.with_dependencies(
                    [barrier], self.d_losses[-1])
                self.g_losses[-1] = control_flow_ops.with_dependencies(
                    [barrier], self.g_losses[-1])
                self.d_loss_real = control_flow_ops.with_dependencies(
                    [barrier], self.d_loss_real)
                self.d_loss_fake = control_flow_ops.with_dependencies(
                    [barrier], self.d_loss_fake)
                self.d_loss_class = control_flow_ops.with_dependencies(
                    [barrier], self.d_loss_class)
        t_vars = self._get_vars_semi_supervised()
        if self.clip_by_global_norm:
            self.capped_d_grads = self._clip_grad_global_norms(
                t_vars['d_vars'], self.d_losses[-1], d_optimizer, gradient_noise_scale=0.0)
            self.capped_g_grads = self._clip_grad_global_norms(
                t_vars['g_vars'], self.g_losses[-1], g_optimizer, gradient_noise_scale=0.0)
        else:
            self.capped_d_grads = self._clip_grad_norms(d_optimizer.compute_gradients(
                self.d_losses[-1], t_vars['d_vars']))
            self.capped_g_grads = self._clip_grad_norms(g_optimizer.compute_gradients(
                self.g_losses[-1], t_vars['g_vars']))
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        if self.gradient_multipliers is not None:
            with tf.name_scope('multiply_grads'):
                self.capped_d_grads = self._multiply_gradients(
                    self.capped_d_grads, self.gradient_multipliers)
        apply_d_gradient_op = d_optimizer.apply_gradients(
            self.capped_d_grads, global_step=global_step)
        apply_g_gradient_op = g_optimizer.apply_gradients(
            self.capped_g_grads, global_step=global_step)
        self.train_op_d = control_flow_ops.with_dependencies(
            [apply_d_gradient_op], self.d_losses[-1])
        self.train_op_g = control_flow_ops.with_dependencies(
            [apply_g_gradient_op], self.g_losses[-1])
