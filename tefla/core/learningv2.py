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
from ..da.data_augmentation import inputs, distorted_inputs
from ..dataset.base import Dataset
from ..dataset.decoder import Decoder
from ..dataset.dataflow import Dataflow
from ..da.preprocessor import InceptionPreprocessor


TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class SupervisedLearner(Base, BaseMixin):
    """
    Supervised Learner, support data parallelism, multi GPU, accept TFRecords data as input


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

    def __init__(self, model, cnf, clip_by_global_norm=False, data_balancing=1, **kwargs):
        self.clip_by_global_norm = clip_by_global_norm
        self.data_balancing = data_balancing
        super(SupervisedLearner, self).__init__(
            model, cnf, **kwargs)

    def fit(self, data_dir, data_dir_val=None, features_keys=None, weights_from=None, weights_dir='weights', max_to_keep=None, start_epoch=1, summary_every=10, training_set_size=None, val_set_size=None, dataset_name='cifar10', keep_moving_averages=False):
        """
        Train the model on the specified dataset

        Args:
            data_dir: str, training dataset directory (where tfrecords are staored for training)
            data_dir_val: str optional, validation dataset directory (where tfrecords are stored for validation)
            features_keys: a dict, tfrecords keys to datum features
            e.g.:
            features_keys = {
                'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
            weights_from: str, if not None, initializes model from exisiting weights
            training_set_size: int, number of training examples
            val_set_size: int, set if data_dir_val not None, number of validation examples
            dataset_name: a optional, Name of the dataset
            start_epoch: int,  epoch number to start training from
                e.g. for retarining set the epoch number you want to resume training from
            summary_every: int, epoch interval to write summary; higher value means lower frequency
                of summary writing
            keep_moving_averages: a bool, keep moving averages of trainable variables
        """
        tf.reset_default_graph()
        with tf.Graph().as_default():
            dataflow_train, dataflow_val = self._data_ops(
                data_dir, data_dir_val, features_keys=features_keys, training_set_size=training_set_size, val_set_size=val_set_size, dataset_name=dataset_name)
            self._setup_model_loss(
                dataflow_train, dataflow_val=dataflow_val, keep_moving_averages=keep_moving_averages, loss_type=self.loss_type)
            if self.is_summary:
                # self._setup_summaries(self.grads_and_vars)
                self._setup_summaries(
                    activation_summary=True)
            self._setup_misc()
            self._print_info(data_dir)
            if max_to_keep is not None:
                max_to_keep = int(max_to_keep)
            self._train_loop(dataflow_train, dataflow_val, weights_from, weights_dir,
                             start_epoch, summary_every, max_to_keep=max_to_keep)

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None:
            with tf.control_dependencies([tf.group(*self.update_ops)]):
                self.training_loss = tf.identity(
                    self.training_loss, name='train_loss')

    def _data_ops(self, data_dir, data_dir_val, features_keys=None, training_set_size=50000, val_set_size=10000, dataset_name='datarandom'):
        num_readers = self.cnf.get('num_readers', 8)
        self.preprocessor = InceptionPreprocessor()
        if features_keys is None:
            features_keys = {
                'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }

        decoder = Decoder(features_keys)

        dataset = Dataset(dataset_name, decoder, data_dir,
                          num_examples_per_epoch=training_set_size, batch_size=self.cnf['batch_size_train'])

        dataflow_train = Dataflow(dataset, num_readers=num_readers,
                                  shuffle=True, min_queue_examples=self.cnf.get('min_queue_examples', 1000), capacity=self.cnf.get('capacity', 2000))
        if data_dir_val is not None:
            dataset_val = Dataset(dataset_name, decoder, data_dir_val,
                                  num_examples_per_epoch=val_set_size, batch_size=self.cnf['batch_size_train'])

            dataflow_val = Dataflow(dataset_val, num_readers=num_readers,
                                    shuffle=False, min_queue_examples=self.cnf.get('min_queue_examples', 1000), capacity=self.cnf.get('capacity', 2000))
            return dataflow_train, dataflow_val
        else:
            return dataflow_train, None

    def _train_loop(self, dataset, dataset_val, weights_from, weights_dir, start_epoch, summary_every, max_to_keep=None):
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        weights_dir = "weights"
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
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False))
        if start_epoch > 1:
            weights_from = "weights/model-epoch-%d.ckpt" % (
                start_epoch - 1)

        sess.run(tf.global_variables_initializer())
        if weights_from:
            self._load_weights(sess, saver, weights_from)

        learning_rate_value = self.lr_policy.initial_lr
        log.info("Initial learning rate: %f " % learning_rate_value)
        if self.is_summary:
            summary_dir = self.cnf.get('summary_dir', '/tmp/tefla-summary')
            if not os.path.exists(summary_dir):
                tf.gfile.MakeDirs(summary_dir)
            train_writer, validation_writer = summary.create_summary_writer(
                summary_dir, sess)

        seed_delta = 100
        training_history = []
        current_probs = np.array([1 / float(self.num_classes)
                                  for _ in range(0, self.num_classes)])
        print(current_probs.shape)
        diff_probs = self._get_diff_prob(current_probs)
        batch_iter_idx = 1
        n_iters_per_epoch = dataset.n_iters_per_epoch
        self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
        self.total_network_params()
        self.write_params()
        self.write_graph(sess.graph_def, weights_dir)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in xrange(start_epoch, self.num_epochs + 1):
            np.random.seed(epoch + seed_delta)
            tf.set_random_seed(epoch + seed_delta)
            tic = time.time()
            training_losses = []
            batch_train_sizes = []

            for batch_num in xrange(1, n_iters_per_epoch + 1):
                feed_dict_train = {
                    self.learning_rate: learning_rate_value, self.target_probs: list(current_probs)}

                log.debug('1. Loading batch %d data done.' % batch_num)
                if epoch % summary_every == 0 and self.is_summary:
                    log.debug('2. Running training steps with summary...')
                    training_predictions_e, training_loss_e, summary_str_train, _ = sess.run(
                        [self.training_predictions, self.training_loss, training_batch_summary_op,
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
                batch_train_sizes.append(self.cnf['batch_size_train'])

                if self.update_ops is not None:
                    log.debug('3. Running update ops...')
                    sess.run(self.update_ops, feed_dict=feed_dict_train)
                    log.debug('3. Running update ops done.')

                learning_rate_value = self.lr_policy.batch_update(
                    learning_rate_value, batch_iter_idx)
                batch_iter_idx += 1
                log.debug('4. Training batch %d done.' % batch_num)

            current_probs += diff_probs
            log.debug('The value of current_probs {}'.format(current_probs))
            epoch_training_loss = np.average(
                training_losses, weights=batch_train_sizes)
            log.info("Epoch %d [(%s) images, %6.1fs]: t-loss: %.3f" %
                     (epoch, np.sum(batch_train_sizes), time.time() - tic, epoch_training_loss))

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
            epoch_validation_loss = 0
            batch_validation_sizes = []
            if dataset_val is not None:
                for batch_num in xrange(dataset_val.n_iters_per_epoch):
                    log.debug(
                        '6. Loading batch %d validation data done.' % batch_num)

                    if (epoch - 1) % summary_every == 0 and self.is_summary:
                        log.debug(
                            '7. Running validation steps with summary...')
                        _validation_metric, summary_str_validate = sess.run(
                            [self.validation_metric, validation_batch_summary_op])
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
                        _validation_metric = sess.run(self.validation_metric)
                        log.debug(
                            '7. Running validation steps without summary done.')
                    validation_losses.append(_validation_metric[-1])
                    batch_validation_sizes.append(
                        self.cnf['batch_size_test'])

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
            learning_rate_value = self.lr_policy.epoch_update(
                learning_rate_value, training_history)
            log.info("Learning rate: %f " % learning_rate_value)

        if self.is_summary:
            train_writer.close()
            validation_writer.close()
        coord.request_stop()
        coord.join(stop_grace_period_secs=0.05)

    def _get_diff_prob(self, target_probs):
        init_probs = np.array(self.cnf['init_probs'])
        diff_probs = (init_probs - target_probs) / float(self.cnf['num_epochs'])
        return diff_probs

    def _adjust_ground_truth(self, labels):
        if self.loss_type == 'kappa_log':
            return tf.to_float(tf.one_hot(labels, self.num_classes))
        else:
            return labels if self.classification else tf.reshape(labels, shape=(-1, 1))

    def _process_towers_grads(self, dataset, opt, model, is_training=True, reuse=None, loss_type='cross_entropy', is_classification=True):
        tower_grads = []
        tower_loss = []
        self.target_probs = tf.placeholder_with_default(tf.convert_to_tensor([1 / float(self.num_classes) for _ in range(0, self.num_classes)]),
                                                        shape=[self.num_classes, ], name="target_probs")
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(self.cnf.get('num_gpus', 1)):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (self.cnf.get('TOWER_NAME', 'tower'), i)) as scope:
                        images, labels = distorted_inputs(dataset, self.cnf['tfrecords_im_size'], self.cnf.get(
                            'crop_size'), batch_size=self.cnf['batch_size_train'], num_preprocess_threads=32, num_readers=8, target_probs=self.target_probs, init_probs=tf.convert_to_tensor(self.cnf['init_probs']), image_preprocessing=self.preprocessor.preprocess_image, data_balancing=self.data_balancing)
                        labels = self._adjust_ground_truth(labels)
                        loss = self._tower_loss(scope, model, images, labels, is_training=is_training,
                                                reuse=i > 0, is_classification=is_classification, gpu_id=i, loss_type=loss_type)

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

    def _process_towers_loss(self, dataset, opt, model, is_training=False, reuse=True, is_classification=True, loss_type='cross_entropy'):
        tower_loss = []
        predictions = []
        validation_metric = []
        validation_metric_tmp = [[] for _, _ in self.validation_metrics_def]
        for i in xrange(self.cnf.get('num_gpus', 1)):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (self.cnf.get('TOWER_NAME', 'tower'), i)) as scope:
                    images, labels = inputs(dataset, self.cnf['tfrecords_im_size'], self.cnf.get(
                        'crop_size'), batch_size=self.cnf['batch_size_test'], num_preprocess_threads=32, num_readers=8, image_preprocessing=self.preprocessor.preprocess_image)
                    labels = self._adjust_ground_truth(labels)
                    loss_pred = self._tower_loss(
                        scope, model, images, labels, is_training=is_training, reuse=reuse, is_classification=is_classification, loss_type=loss_type)
                    tower_loss.append(loss_pred['loss'])
                    predictions.append(loss_pred['predictions'])
                    if self.loss_type == 'kappa_log':
                        labels = tf.argmax(labels, axis=1)
                    for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                        metric_score = metric_function(
                            labels, tf.argmax(loss_pred['predictions'], 1))
                        validation_metric_tmp[i].append(metric_score)
        predictions = tf.convert_to_tensor(predictions)
        predictions = tf.reshape(predictions, [-1, self.num_classes])
        for i, (_, _) in enumerate(self.validation_metrics_def):
            validation_metric.append(
                tf.divide(sum(validation_metric_tmp[i]), self.cnf.get('num_gpus')))
        return sum(tower_loss), predictions, validation_metric

    def _setup_model_loss(self, dataflow, dataflow_val=None, keep_moving_averages=False, loss_type='cross_entropy'):
        self.learning_rate = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder")
        # Keep old variable around to load old params, till we need this
        self.obsolete_learning_rate = tf.Variable(
            1.0, trainable=False, name="learning_rate")
        optimizer = self._optimizer(self.learning_rate, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        self.grads_and_vars, self.training_loss = self._process_towers_grads(
            dataflow, optimizer, self.model, is_classification=self.classification, loss_type=loss_type)
        if dataflow_val is not None:
            self.validation_loss, self.validation_predictions, self.validation_metric = self._process_towers_loss(
                dataflow_val, optimizer, self.model, is_classification=self.classification, loss_type=loss_type)
            self.validation_metric.append(self.validation_loss)

        if self.clip_norm and not self.clip_by_global_norm:
            self.grads_and_vars = self._clip_grad_norms(
                self.grads_and_vars, max_norm=self.norm_threshold)
        apply_gradients_op = optimizer.apply_gradients(self.grads_and_vars)
        if keep_moving_averages:
            variables_averages_op = self._moving_averages_op()
            with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
                self.train_op = tf.no_op(name='train_op')
            # self.train_op = tf.group(apply_gradients_op,
            # variables_averages_op):
        else:
            self.train_op = apply_gradients_op
