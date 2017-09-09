# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import

import os
import re
import pprint
import time

import numpy as np
import tensorflow as tf

from .base import Base, BaseMixin
from . import summary as summary
from . import logger as log
from ..utils import util


TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class SupervisedLearner(Base, BaseMixin):
    """
    Supervised Trainer, support data parallelism, multi GPU


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
        super(SupervisedLearner, self).__init__(
            model, cnf, **kwargs)

    def fit(self, data_set, weights_from=None, weights_dir='weights', start_epoch=1, summary_every=10, keep_moving_averages=False, **kwargs):
        """
        Train the model on the specified dataset

        Args:
            data_set: dataset instance to use to access data for training/validation
            weights_from: str, if not None, initializes model from exisiting weights
            start_epoch: int,  epoch number to start training from
                e.g. for retarining set the epoch number you want to resume training from
            summary_every: int, epoch interval to write summary; higher value means lower frequency
                of summary writing
            keep_moving_averages: a bool, keep moving averages of trainable variables
        """
        with tf.Graph().as_default():
            self._setup_model_loss(
                keep_moving_averages=keep_moving_averages)
            if self.is_summary:
                self. _setup_summaries(**kwargs)
            self._setup_misc()
            self._print_info(data_set)
            self._train_loop(data_set, weights_from, weights_dir,
                             start_epoch, summary_every)

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None:
            with tf.control_dependencies([tf.group(*self.update_ops)]):
                self.training_loss = tf.identity(
                    self.training_loss, name='train_loss')

    def _train_loop(self, data_set, weights_from, weights_dir, start_epoch, summary_every):
        training_X, training_y, validation_X, validation_y = \
            data_set.training_X, data_set.training_y, data_set.validation_X, data_set.validation_y
        saver = tf.train.Saver(max_to_keep=None)
        if not os.path.exists(weights_dir):
            tf.gfile.MakeDirs(weights_dir)
        if self.is_summary:
            training_batch_summary_op = tf.summary.merge_all(
                key=TRAINING_BATCH_SUMMARIES)
            training_epoch_summary_op = tf.summary.merge_all(
                key=TRAINING_EPOCH_SUMMARIES)
            validation_batch_summary_op = tf.summary.merge_all(
                key=VALIDATION_BATCH_SUMMARIES)
            validation_epoch_summary_op = tf.summary.merge_all(
                key=VALIDATION_EPOCH_SUMMARIES)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)) as sess:
            if start_epoch > 1:
                weights_from = "weights/model-epoch-%d.ckpt" % (
                    start_epoch - 1)

            sess.run(tf.global_variables_initializer())
            if weights_from:
                self._load_weights(sess, saver, weights_from)

            learning_rate_value = self.lr_policy.initial_lr
            log.info("Initial learning rate: %f " % learning_rate_value)
            if self.is_summary:
                train_writer, validation_writer = summary.create_summary_writer(
                    self.cnf.get('summary_dir', '/tmp/tefla-summary'), sess)

            seed_delta = 100
            training_history = []
            batch_iter_idx = 1
            n_iters_per_epoch = len(
                data_set.training_X) // self.training_iterator.batch_size
            self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
            self.total_network_params()
            self.write_graph(sess.graph_def, weights_dir)
            for epoch in xrange(start_epoch, self.num_epochs + 1):
                np.random.seed(epoch + seed_delta)
                tf.set_random_seed(epoch + seed_delta)
                tic = time.time()
                training_losses = []
                batch_train_sizes = []

                for batch_num, (Xb, yb) in enumerate(self.training_iterator(training_X, training_y)):
                    if Xb.shape[0] < self.cnf['batch_size_train']:
                        continue
                    feed_dict_train = {self.inputs: Xb, self.labels: self._adjust_ground_truth(yb),
                                       self.learning_rate: learning_rate_value}

                    log.debug('1. Loading batch %d data done.' % batch_num)
                    if epoch % summary_every == 0 and self.is_summary and training_batch_summary_op is not None:
                        log.debug('2. Running training steps with summary...')
                        training_loss_e, summary_str_train, _ = sess.run(
                            [self.training_loss, training_batch_summary_op,
                             self.train_op],
                            feed_dict=feed_dict_train)
                        train_writer.add_summary(summary_str_train, epoch)
                        train_writer.flush()
                        log.debug(
                            '2. Running training steps with summary done.')
                        log.debug("Epoch %d, Batch %d training loss: %s" %
                                  (epoch, batch_num, training_loss_e))
                        log.debug("Epoch %d, Batch %d training predictions: %s" % (
                            epoch, batch_num, training_predictions_e))
                    else:
                        log.debug(
                            '2. Running training steps without summary...')
                        training_loss_e, _ = sess.run([self.training_loss, self.train_op],
                                                      feed_dict=feed_dict_train)
                        log.debug(
                            '2. Running training steps without summary done.')

                    training_losses.append(training_loss_e)
                    batch_train_sizes.append(len(Xb))

                    if self.update_ops is not None:
                        log.debug('3. Running update ops...')
                        sess.run(self.update_ops, feed_dict=feed_dict_train)
                        log.debug('3. Running update ops done.')

                    learning_rate_value = self.lr_policy.batch_update(
                        learning_rate_value, batch_iter_idx)
                    batch_iter_idx += 1
                    log.debug('4. Training batch %d done.' % batch_num)

                epoch_training_loss = np.average(
                    training_losses, weights=batch_train_sizes)

                # Plot training loss every epoch
                log.debug('5. Writing epoch summary...')
                if self.is_summary:
                    summary_str_train = sess.run(training_epoch_summary_op, feed_dict={
                                                 self.epoch_loss: epoch_training_loss, self.learning_rate: learning_rate_value})
                    train_writer.add_summary(summary_str_train, epoch)
                    train_writer.flush()
                log.debug('5. Writing epoch summary done.')

                # Validation prediction and metrics
                validation_losses = []
                batch_validation_metrics = [[]
                                            for _, _ in self.validation_metrics_def]
                epoch_validation_metrics = []
                batch_validation_sizes = []
                for batch_num, (validation_Xb, validation_yb) in enumerate(
                        self.validation_iterator(validation_X, validation_y)):
                    if validation_Xb.shape[0] < self.cnf['batch_size_test']:
                        continue
                    feed_dict_validation = {self.validation_inputs: validation_Xb,
                                            self.validation_labels: self._adjust_ground_truth(validation_yb)}
                    log.debug(
                        '6. Loading batch %d validation data done.' % batch_num)

                    if (epoch - 1) % summary_every == 0 and self.is_summary and validation_batch_summary_op is not None:
                        log.debug(
                            '7. Running validation steps with summary...')
                        _validation_metric, summary_str_validate = sess.run(
                            [self.validation_metric, validation_batch_summary_op], feed_dict=feed_dict_validation)
                        validation_writer.add_summary(
                            summary_str_validate, epoch)
                        validation_writer.flush()
                        log.debug(
                            '7. Running validation steps with summary done.')
                        log.debug(
                            "Epoch %d, Batch %d validation loss: %s" % (epoch, batch_num, _validation_metric[-1]))
                        log.debug("Epoch %d, Batch %d validation predictions: %s" % (
                            epoch, batch_num, _validation_metric[0]))
                    else:
                        log.debug(
                            '7. Running validation steps without summary...')
                        _validation_metric = sess.run(
                            self.validation_metric, feed_dict=feed_dict_validation)
                        log.debug(
                            '7. Running validation steps without summary done.')
                    validation_losses.append(_validation_metric[-1])
                    batch_validation_sizes.append(
                        self.cnf.get('batch_size_test', 32))

                    for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                        batch_validation_metrics[i].append(
                            _validation_metric[i])
                    log.debug('8. Validation batch %d done' % batch_num)

                epoch_validation_loss = np.average(
                    validation_losses, weights=batch_validation_sizes)
                for i, (_, _) in enumerate(self.validation_metrics_def):
                    epoch_validation_metrics.append(
                        np.average(batch_validation_metrics[i], weights=batch_validation_sizes))

                # Write validation epoch summary every epoch
                log.debug('9. Writing epoch validation summary...')
                if self.is_summary:
                    summary_str_validate = sess.run(validation_epoch_summary_op, feed_dict={
                        self.epoch_loss: epoch_validation_loss, self.validation_metric_placeholders: epoch_validation_metrics})
                    validation_writer.add_summary(
                        summary_str_validate, epoch)
                    validation_writer.flush()
                log.debug('9. Writing epoch validation summary done.')

                custom_metrics_string = [', %s: %.3f' % (name, epoch_validation_metrics[i]) for i, (name, _) in
                                         enumerate(self.validation_metrics_def)]
                custom_metrics_string = ''.join(custom_metrics_string)

                log.info(
                    "Epoch %d [(%s, %s) images, %6.1fs]: t-loss: %.3f, v-loss: %.3f%s" %
                    (epoch, np.sum(batch_train_sizes), np.sum(batch_validation_sizes), time.time() - tic,
                     epoch_training_loss,
                     epoch_validation_loss,
                     custom_metrics_string)
                )

                saver.save(sess, "%s/model-epoch-%d.ckpt" %
                           (weights_dir, epoch))

                epoch_info = dict(
                    epoch=epoch,
                    training_loss=epoch_training_loss,
                    validation_loss=epoch_validation_loss
                )

                training_history.append(epoch_info)

                log.debug('10. Epoch done. [%d]' % epoch)
                learning_rate_value = self.lr_policy.epoch_update(
                    learning_rate_value, training_history)
                log.info("Learning rate: %f " % learning_rate_value)
            if self.is_summary:
                train_writer.close()
                validation_writer.close()

    def _process_towers_grads(self, opt, model, is_training=True, reuse=None, is_classification=True):
        tower_grads = []
        tower_loss = []
        if self.cnf.get('num_gpus', 1) > 1:
            images_gpus = tf.split(
                self.inputs, self.cnf.get('num_gpus', 1), axis=0)
            labels_gpus = tf.split(
                self.labels, self.cnf.get('num_gpus', 1), axis=0)
        else:
            images_gpus = [self.inputs]
            labels_gpus = [self.labels]
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(self.cnf.get('num_gpus', 1)):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (self.cnf.get('TOWER_NAME', 'tower'), i)) as scope:
                        loss = self._tower_loss(scope, model, images_gpus[i], labels_gpus[
                            i], is_training, reuse=i > 0, loss_type=self.loss_type, is_classification=is_classification, gpu_id=i)

                        tf.get_variable_scope().reuse_variables()
                        if self.clip_by_global_norm:
                            grads_and_vars = self._clip_grad_global_norms(tf.trainable_variables(
                            ), loss, opt, global_norm=self.norm_threshold, gradient_noise_scale=0.0)
                        else:
                            grads_and_vars = opt.compute_gradients(loss)
                        tower_grads.append(grads_and_vars)
                        tower_loss.append(loss)

        grads_and_vars = self._average_gradients(tower_grads)

        return grads_and_vars, sum(tower_loss)

    def _process_towers_loss(self, opt, model, is_training=False, reuse=True, is_classification=True):
        tower_loss = []
        predictions = []
        validation_metric = []
        validation_metric_tmp = [[] for _, _ in self.validation_metrics_def]
        if self.cnf.get('num_gpus', 1) > 1:
            images_gpus = tf.split(
                self.validation_inputs, self.cnf.get('num_gpus', 1), axis=0)
            labels_gpus = tf.split(
                self.validation_labels, self.cnf.get('num_gpus', 1), axis=0)
        else:
            images_gpus = [self.validation_inputs]
            labels_gpus = [self.validation_labels]
        for i in xrange(self.cnf.get('num_gpus', 1)):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (self.cnf.get('TOWER_NAME', 'tower'), i)) as scope:
                    loss_pred = self._tower_loss(scope, model, images_gpus[i], labels_gpus[
                        i], is_training, reuse, loss_type=self.loss_type, is_classification=is_classification)
                    tower_loss.append(loss_pred['loss'])
                    predictions.append(loss_pred['predictions'])
                    if self.loss_type == 'kappa_log':
                        labels_gpus[i] = tf.argmax(labels_gpus[i], axis=1)
                    for j, (_, metric_function) in enumerate(self.validation_metrics_def):
                        metric_score = metric_function(
                            labels_gpus[i], tf.argmax(loss_pred['predictions'], 1))
                        validation_metric_tmp[j].append(metric_score)
        predictions = tf.convert_to_tensor(predictions)
        predictions = tf.reshape(predictions, [-1, self.num_classes])
        for i, (_, _) in enumerate(self.validation_metrics_def):
            validation_metric.append(sum(validation_metric_tmp[i]))
        return sum(tower_loss), predictions, validation_metric

    def _setup_model_loss(self, val=True, keep_moving_averages=False):
        self.learning_rate = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder")
        # Keep old variable around to load old params, till we need this
        self.obsolete_learning_rate = tf.Variable(
            1.0, trainable=False, name="learning_rate")
        optimizer = self._optimizer(self.learning_rate, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        self.inputs = tf.placeholder(tf.float32, shape=(
            self.cnf['batch_size_train'],) + self.cnf['input_size'], name="input")
        if self.loss_type == 'kappa_log':
            self.labels = tf.placeholder(tf.int64, shape=(
                self.cnf['batch_size_train'], self.num_classes))
            self.validation_labels = tf.placeholder(
                tf.int64, shape=(self.cnf['batch_size_test'], self.num_classes))
        else:
            self.labels = tf.placeholder(
                tf.int64, shape=(self.cnf['batch_size_train'],))
            self.validation_labels = tf.placeholder(
                tf.int64, shape=(self.cnf['batch_size_test'],))
        self.validation_inputs = tf.placeholder(tf.float32, shape=(
            self.cnf['batch_size_test'],) + self.cnf['input_size'], name="validation_input")
        self.grads_and_vars, self.training_loss = self._process_towers_grads(
            optimizer, self.model, is_classification=self.classification)
        self.validation_loss, self.validation_predictions, self.validation_metric = self._process_towers_loss(
            optimizer, self.model, is_classification=self.classification)
        self.validation_metric.append(self.validation_loss)

        if self.clip_norm and not self.clip_by_global_norm:
            self.grads_and_vars = self._clip_grad_norms(
                self.grads_and_vars, max_norm=self.norm_threshold)
        apply_gradients_op = optimizer.apply_gradients(self.grads_and_vars)
        if keep_moving_averages:
            variables_averages_op = self._moving_averages_op()
            with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
                self.train_op = tf.no_op(name='train')
        else:
            self.train_op = apply_gradients_op
