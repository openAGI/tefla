# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from tefla.core import logger as log
from tefla.core import summary as summary
from tefla.core.base import Base
from tefla.utils import util


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
        super(SemiSupervisedTrainer, self).__init__(self, util.load_module(model), cnf, **kwargs)

    def fit(self, data_set, weights_from=None, start_epoch=1, summary_every=199, model_name='multiclass_ss', weights_dir='weights'):
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
        self._setup_model_loss()
        if self.is_summary:
            self._setup_summaries()
        self._setup_misc()
        self._print_info(data_set)
        self._train_semi_supervised(data_set, start_epoch, weights_from, summary_every, model_name, weights_dir)

    def _train_semi_supervised(self, dataset, start_epoch, weights_from, summary_every, model_name, weights_dir):
        with tf.Graph().as_default(), tf.device('/gpu:0'):
            training_X, training_y, validation_X, validation_y = \
                dataset.training_X, dataset.training_y, dataset.validation_X, dataset.validation_y
            if not os.path.exists(weights_dir):
                os.mkdir(weights_dir)
            if not os.path.exists(weights_dir + '/best_models'):
                os.mkdir(weights_dir + '/best_models')

            # Create a saver.
            saver = tf.train.Saver(max_to_keep=None)
            if not os.path.exists(weights_dir):
                os.mkdir(weights_dir)
            if self.is_summary:
                training_batch_summary_op = tf.merge_all_summaries(key=TRAINING_BATCH_SUMMARIES)
                training_epoch_summary_op = tf.merge_all_summaries(key=TRAINING_EPOCH_SUMMARIES)
                validation_batch_summary_op = tf.merge_all_summaries(key=VALIDATION_BATCH_SUMMARIES)
                validation_epoch_summary_op = tf.merge_all_summaries(key=VALIDATION_EPOCH_SUMMARIES)

            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.cnf['memory_assign'])
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
            sess.run(init)
            if start_epoch > 1:
                weights_from = "weights/model-epoch-%d.ckpt" % (start_epoch - 1)

            if weights_from:
                self._load_weights(sess, saver, weights_from)

            learning_rate_value = self.lr_policy.initial_lr
            log.info("Initial learning rate: %f " % learning_rate_value)
            if self.is_summary:
                train_writer, validation_writer = summary.create_summary_writer(self.cnf.get('summary_dir', '/tmp/tefla-summary'), sess)
            # keep track of maximum accuracy and auroc and save corresponding weights
            seed_delta = 100
            batch_iter_idx = 1
            n_iters_per_epoch = len(dataset.training_X) // self.training_iterator.batch_size
            self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
            for epoch in xrange(start_epoch, self.mum_epochs + 1):
                np.random.seed(epoch + seed_delta)
                tf.set_random_seed(epoch + seed_delta)
                tic = time.time()
                d_train_losses = []
                g_train_losses = []
                batch_train_sizes = []
                for batch_num, (Xb, yb) in enumerate(self.training_iterator(training_X, training_y)):
                    feed_dict_train = {self.inputs: Xb.transpose(0, 2, 3, 1), self.labels: yb, self.learning_rate: learning_rate_value}
                    log.debug('1. Loading batch %d data done.' % batch_num)
                    if epoch % summary_every == 0 and self.is_summary:
                        log.debug('2. Running training steps with summary...')
                        _, _d_loss_real, _d_loss_fake, _d_loss_class, summary_str_train = sess.run([self.train_op_d, self.d_loss_real, self.d_loss_fake, self.d_loss_class, training_batch_summary_op], feed_dict=feed_dict_train)
                        _, _g_loss = sess.run([self.train_op_g, self.g_losses[0]], feed_dict=feed_dict_train)
                        train_writer.add_summary(summary_str_train, epoch)
                        train_writer.flush()
                        log.debug('2. Running training steps with summary done.')
                        log.debug("Epoch %d, Batch %d D_loss_real: %s, D_loss_fake: %s,D_loss_class: %s, G_loss: %s" % (epoch, batch_num, _d_loss_real, _d_loss_fake, _d_loss_class, _g_loss))
                    else:
                        log.debug('2. Running training steps without summary...')
                        _, _d_loss_real, _d_loss_fake, _d_loss_class = sess.run([self.train_op_d, self.d_loss_real, self.d_loss_fake, self.d_loss_class], feed_dict=feed_dict_train)
                        _, _g_loss = sess.run([self.train_op_g, self.g_losses[0]], feed_dict=feed_dict_train)
                        log.debug('2. Running training steps without summary done.')

                    d_train_losses.append(_d_loss_real + _d_loss_fake + _d_loss_class)
                    g_train_losses.append(_g_loss)
                    batch_train_sizes.append(len(Xb))
                    learning_rate_value = self.lr_policy.batch_update(learning_rate_value, batch_iter_idx)
                    batch_iter_idx += 1
                    log.debug('4. Training batch %d done.' % batch_num)
                d_avg_loss = np.average(d_train_losses, weights=batch_train_sizes)
                g_avg_loss = np.average(g_train_losses, weights=batch_train_sizes)
                log.debug("Epoch %d, D_avg_loss: %s, G_avg_loss" % (epoch, d_avg_loss, g_avg_loss))
                # Plot training loss every epoch
                log.debug('5. Writing epoch summary...')
                if self.is_summary:
                    summary_str_train = sess.run(training_epoch_summary_op, feed_dict={self.epoch_loss: d_avg_loss, self.epoch_loss_g: g_avg_loss, self.learning_rate: learning_rate_value})
                    train_writer.add_summary(summary_str_train, epoch)
                    train_writer.flush()
                log.debug('5. Writing epoch summary done.')
                # Validation prediction and metrics
                validation_losses = []
                batch_validation_metrics = [[] for _, _ in self.validation_metrics_def]
                epoch_validation_metrics = []
                batch_validation_sizes = []
                for batch_num, (validation_Xb, validation_y_true) in enumerate(self.validation_iterator(validation_X, validation_y)):
                    feed_dict_val = {self.inputs: validation_Xb.transpose(0, 2, 3, 1), self.labels: validation_y_true}
                    log.debug('6. Loading batch %d validation data done.' % batch_num)
                    if (epoch - 1) % summary_every == 0 and self.is_summary:
                        log.debug('7. Running validation steps with summary...')
                        validation_y_pred, _val_loss, summary_str_validation = sess.run([self.predictions, self.test_loss, validation_batch_summary_op], feed_dict=feed_dict_val)

                        validation_writer.add_summary(summary_str_validation, epoch)
                        validation_writer.flush()
                        log.debug('7. Running validation steps with summary done.')
                        log.debug(
                            "Epoch %d, Batch %d validation loss: %s" % (epoch, batch_num, _val_loss))
                        log.debug("Epoch %d, Batch %d validation predictions: %s" % (
                            epoch, batch_num, validation_y_pred))
                    else:
                        log.debug('7. Running validation steps without summary...')
                        validation_y_pred, _val_loss = sess.run([self.predictions, self.test_loss], feed_dict=feed_dict_val)

                        log.debug('7. Running validation steps without summary done.')
                    validation_losses.append(_val_loss)
                    batch_validation_sizes.append(len(validation_Xb))
                    for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                        metric_score = metric_function(validation_y_true, validation_y_pred)
                        batch_validation_metrics[i].append(metric_score)
                    log.debug('8. Validation batch %d done' % batch_num)

                epoch_validation_loss = np.average(validation_losses, weights=batch_validation_sizes)
                for i, (_, _) in enumerate(self.validation_metrics_def):
                    epoch_validation_metrics.append(
                        np.average(batch_validation_metrics[i], weights=batch_validation_sizes))
                log.debug('9. Writing epoch validation summary...')
                if self.is_summary:
                    summary_str_validate = sess.run(validation_epoch_summary_op, feed_dict={self.epoch_loss: epoch_validation_loss, self.validation_metric_placeholders: epoch_validation_metrics})
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

                saver.save(sess, "%s/model-epoch" % (weights_dir), global_step=epoch)
                learning_rate_value = self.lr_policy.epoch_update(learning_rate_value)

                # Learning rate step decay
            if self.is_summary:
                train_writer.close()
                validation_writer.close()

    def feature_matching_loss(self, real_data_features, fake_data_features):
        real_data_mean = tf.reduce_mean(real_data_features, reduction_indices=0)
        fake_data_mean = tf.reduce_mean(fake_data_features, reduction_indices=0)
        feature_loss = tf.reduce_mean(tf.abs(tf.sub(real_data_mean, fake_data_mean)))

        return feature_loss

    def tower_loss_semi_supervised(self, model_graph, inputs, targets, gpu_idx=0, num_classes=11, is_fm_loss=True):
        with tf.variable_scope("train_specific"):
            avg_error_rate = tf.get_variable('avg_error_rate', [], initializer=tf.constant_initializer(0.), trainable=False)
            num_error_rate = tf.get_variable('num_error_rate', [], initializer=tf.constant_initializer(0.), trainable=False)

        batch_size_train = self.cnf['batch_size_train']
        batch_size_val = self.cnf['batch_size_test']
        end_points_G = model_graph.generator([batch_size_train, 100])

        if gpu_idx == 0:
            G_means = tf.reduce_mean(end_points_G['softmax'], 0, keep_dims=True)
            G_vars = tf.reduce_mean(tf.square(end_points_G['softmax'] - G_means), 0, keep_dims=True)
            G = tf.Print(end_points_G['softmax'], [tf.reduce_mean(G_means), tf.reduce_mean(G_vars)], "generator mean and average var", first_n=1)
            inputs_means = tf.reduce_mean(inputs, 0, keep_dims=True)
            inputs_vars = tf.reduce_mean(tf.square(inputs - inputs_means), 0, keep_dims=True)
            inputs = tf.Print(inputs, [tf.reduce_mean(inputs_means), tf.reduce_mean(inputs_vars)], "image mean and average var", first_n=1)

        joint = tf.concat(0, [inputs, G])
        print(joint.get_shape())
        end_points_D = model_graph.discriminator(joint, batch_size=batch_size_train)
        end_points_D_val = model_graph.discriminator(inputs, phase=1, batch_size=batch_size_val)

        tf.histogram_summary("d", end_points_D['D_on_data'])
        tf.histogram_summary("d_", end_points_D['D_on_G'])
        tf.image_summary("G", G)

        d_label_smooth = self.cnf['d_label_smooth']  # 0.25
        d_loss_real = self._sigmoid_kl_with_logits(end_points_D['D_on_data_logits'], 1. - d_label_smooth)
        class_loss_weight = 1.
        d_loss_class = class_loss_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(end_points_D['class_logits'], tf.to_int64(targets))
        test_loss = 1. - tf.reduce_mean(tf.to_float(tf.nn.in_top_k(end_points_D_val['logits'], targets, 1)))
        error_rate = 1. - tf.reduce_mean(tf.to_float(tf.nn.in_top_k(end_points_D['class_logits'], targets, 1)))
        if gpu_idx == 0:
            update = tf.assign(num_error_rate, num_error_rate + 1.)
            with tf.control_dependencies([update]):
                tc = tf.maximum(.01, 1. / num_error_rate)
            update = tf.assign(avg_error_rate, (1. - tc) * avg_error_rate + tc * error_rate)
            with tf.control_dependencies([update]):
                # d_loss_class = tf.Print(d_loss_class, [avg_error_rate], "running top-1 error rate")
                d_loss_class = tf.identity(d_loss_class)
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(end_points_D['D_on_G_logits'], tf.zeros_like(end_points_D['D_on_G_logits']))
        d_loss_class = tf.reduce_mean(d_loss_class)
        d_loss_real = tf.reduce_mean(d_loss_real)
        d_loss_fake = tf.reduce_mean(d_loss_fake)

        if is_fm_loss:
            global_pool_head = end_points_D['global_pool']
            real_data_features = tf.slice(global_pool_head, [0, 0], [batch_size_train, num_classes])
            fake_data_features = tf.slice(global_pool_head, [batch_size_train, 0], [batch_size_train, num_classes])
            g_loss = self._feature_matching_loss(real_data_features, fake_data_features)
        else:
            generator_target_prob = self.cnf['generator_target_prob']  # 0.75 / 2.0
            g_loss = self._sigmoid_kl_with_logits(end_points_D['D_on_G_logits'], generator_target_prob)
            g_loss = tf.reduce_mean(g_loss)

        if gpu_idx == 0:
            g_losses = []
        g_losses.append(g_loss)

        d_loss = d_loss_real + d_loss_fake + d_loss_class
        if gpu_idx == 0:
            d_loss_reals = []
            d_loss_fakes = []
            d_loss_classes = []
            d_losses = []
        d_loss_reals.append(d_loss_real)
        d_loss_fakes.append(d_loss_fake)
        d_loss_classes.append(d_loss_class)
        d_losses.append(d_loss)

        return d_losses, g_losses, d_loss_real, d_loss_fake, d_loss_class, test_loss, end_points_D['D_on_data']

    def get_vars_semi_supervised(self):
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
            entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
        return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy

    def _setup_model_loss(self, update_ops=None):
        self.learning_rate_d = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")
        self.learning_rate_g = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")

        d_optimizer = self._optimizer(self.learning_rate_d, optname=self.cnf.get('optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        g_optimizer = self._optimizer(self.learning_rate_g, optname=self.cnf.get('optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        # Get images and labels for ImageNet and split the batch across GPUs.
        assert self.cnf['batch_size_train'] % self.cnf['num_gpus'] == 0, ('Batch size must be divisible by number of GPUs')

        self.inputs = tf.placeholder(tf.float32, shape=(None, self.cnf['w'], self.cnf['h'], 3), name="input")
        self.labels = tf.placeholder(tf.int32, shape=(None,))

        self.d_losses, self.g_losses, self.d_loss_real, self.d_loss_fake, self.d_loss_class, self.test_loss, self.predictions = self._tower_loss_semi_supervised(self.model, self.inputs, self.labels, num_classes=self.num_classes)

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
                self.d_losses[-1] = control_flow_ops.with_dependencies([barrier], self.d_losses[-1])
                self.g_losses[-1] = control_flow_ops.with_dependencies([barrier], self.g_losses[-1])
                self.d_loss_real = control_flow_ops.with_dependencies([barrier], self.d_loss_real)
                self.d_loss_fake = control_flow_ops.with_dependencies([barrier], self.d_loss_fake)
                self.d_loss_class = control_flow_ops.with_dependencies([barrier], self.d_loss_class)
        # grads = opt.compute_gradients(loss)
        # capped_grads = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads]
        # grads = add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale=10.0):
        t_vars = self._get_vars_semi_supervised()
        if self.clip_by_global_norm:
            capped_d_grads = self._clip_grad_global_norms(t_vars['d_vars'], self.d_losses[-1], d_optimizer, gradient_noise_scale=0.0)
            capped_g_grads = self._clip_grad_global_norms(t_vars['g_vars'], self.g_losses[-1], g_optimizer, gradient_noise_scale=0.0)
        else:
            capped_d_grads = d_optimizer.compute_gradients(self.d_losses[-1], t_vars['d_vars'])
            capped_g_grads = g_optimizer.compute_gradients(self.g_losses[-1], t_vars['g_vars'])
        # capped_grads = clip_grad_norms(grads)
        # Scale gradients.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        if self.gradient_multipliers is not None:
            with tf.name_scope('multiply_grads'):
                capped_d_grads = self._multiply_gradients(capped_d_grads, self.gradient_multipliers)
        apply_d_gradient_op = d_optimizer.apply_gradients(capped_d_grads, global_step=global_step)
        apply_g_gradient_op = g_optimizer.apply_gradients(capped_g_grads, global_step=global_step)
        self.train_op_d = control_flow_ops.with_dependencies([apply_d_gradient_op], self.d_losses[-1])
        self.train_op_g = control_flow_ops.with_dependencies([apply_g_gradient_op], self.g_losses[-1])
