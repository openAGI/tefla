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
from datetime import datetime

import numpy as np
import tensorflow as tf

from .base import Base
from . import logger as log
from ..utils import util
from ..da.data_augmentation import inputs, distorted_inputs
from ..dataset.base import Dataset
from ..dataset.decoder import Decoder
from ..dataset.dataflow import Dataflow


TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class DistSupervisedLearner(Base):
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
        summarize_gradients: Whether or not add summaries for each gradient.
        gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
        aggregation_method: Specifies the method used to combine gradient terms.
            Valid values are defined in the class `AggregationMethod`.
        colocate_gradients_with_ops: Whether or not to try colocating the gradients
            with the ops that generated the
    """

    def __init__(self, model, cnf, clip_by_global_norm=False, gradient_noise_scale=None, gradient_multipliers=None, aggregation_method=None, colocate_gradients_with_ops=False, **kwargs):
        self.clip_by_global_norm = clip_by_global_norm
        self.gradient_noise_scale = gradient_noise_scale
        self.gradient_multipliers = gradient_multipliers
        self.aggregation_method = aggregation_method
        self.colocate_gradients_with_ops = colocate_gradients_with_ops
        super(DistSupervisedLearner, self).__init__(
            model, cnf, **kwargs)

    def fit(self, task_id, server, cluster_spec, datadir, datadir_val, features_keys=None, weights_from=None, is_training=True, start_epoch=1, reuse=None, num_replicas_to_aggregate=1, variables_to_train=None, training_set_size=None, val_set_size=None, dataset_name='imagenet', summary_every=None, keep_moving_averages=None):
        """
        Train the model on the specified dataset

        Args:
            task_id: int, id of the task
            server: name of the TensorFlow server for dc to use. See the tf.Session constructor for
                how this is interpreted.
            datadir: datadir, training / val dataset
            cluster_spec: cluster specifications
            reuse: whether to resue variables
            weights_from: str, if not None, initializes model from exisiting weights
            start_epoch: int,  epoch number to start training from
                e.g. for retarining set the epoch number you want to resume training from
            summary_every: int, epoch interval to write summary; higher value means lower frequency
                of summary writing
            variables_to_train: an optional list of variables to train. If None, it will
                  default to all tf.trainable_variables()
            keep_moving_averages: a bool, keep moving averages of trainable variables
        """
        if self.is_summary:
            self._setup_summaries()
        dataflow_train, dataflow_val = self._setup_data_ops(
            datadir, datadir_val, features_keys=features_keys, training_set_size=training_set_size, val_set_size=val_set_size, dataset_name=dataset_name)
        self._setup_misc()
        self._print_info(datadir)
        self.train(task_id, server, dataflow_train, dataflow_val, cluster_spec, is_training, weights_from=weights_from,
                   start_epoch=1, reuse=None, num_replicas_to_aggregate=-1, variables_to_train=None)

    def _setup_data_ops(self, data_dir, data_dir_val, features_keys=None, training_set_size=50000, val_set_size=10000, dataset_name='datarandom'):
        num_readers = self.cnf.get('num_readers', 8)
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

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None and len(self.update_ops) == 0:
            self.update_ops = None
            # if update_ops is not None:
            #     self.regularized_training_loss = tf.with_dependencies(update_ops, self.regularized_training_loss)

    def train(self, task_id, server, dataflow, dataflow_val, cluster_spec, is_training, logdir=None, weights_from=None, start_epoch=1, reuse=None, num_replicas_to_aggregate=-1, variables_to_train=None):
        num_workers = len(cluster_spec.as_dict()['worker'])
        print(num_workers)
        num_parameter_servers = len(cluster_spec.as_dict()['ps'])
        if num_replicas_to_aggregate == -1:
            num_replicas_to_aggregate = num_workers
        else:
            num_replicas_to_aggregate = num_replicas_to_aggregate

        assert num_workers > 0 and num_parameter_servers > 0, (
            ' num_workers and num_parameter_servers must be > 0.')

        is_chief = (task_id == 0)
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/replica:0/task:%d/gpu:0" % (task_id), cluster=cluster_spec)) as scope:
            global_step = tf.get_variable('global_step', shape=[
            ], dtype=tf.int64, initializer=tf.zeros_initializer(), trainable=False)
            learning_rate = self.lr_policy.initial_lr
            n_iters_per_epoch = dataflow.n_iters_per_epoch
            n_val_iters_per_epoch = dataflow_val.n_iters_per_epoch
            self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
            images, labels = distorted_inputs(dataflow, self.cnf['tfrecords_im_size'], self.cnf.get(
                'crop_size'), batch_size=self.cnf['batch_size_train'], num_preprocess_threads=32, num_readers=8)
            labels = self._adjust_ground_truth(labels)
            val_images, val_labels = inputs(dataflow_val, self.cnf['tfrecords_im_size'], self.cnf.get(
                'crop_size'), batch_size=self.cnf['batch_size_test'], num_preprocess_threads=32, num_readers=8)
            val_labels = self._adjust_ground_truth(val_labels)

            total_loss, opt, validation_metric = self._setup_model_loss(images, labels, val_images, val_labels,
                                                                        is_chief, task_id, num_workers, is_training, scope, initial_lr=learning_rate, reuse=None, global_step=global_step, num_replicas_to_aggregate=num_replicas_to_aggregate)
            train_op = self.create_train_op(total_loss, opt, global_step=global_step, update_ops=None, variables_to_train=None, clip_by_global_norm=self.clip_by_global_norm, gradient_noise_scale=self.gradient_noise_scale,
                                            gradient_multipliers=self.gradient_multipliers, gate_gradients=tf.train.Optimizer.GATE_OP, aggregation_method=self.aggregation_method, colocate_gradients_with_ops=self.colocate_gradients_with_ops)

            chief_queue_runners = [opt.get_chief_queue_runner()]
            init_tokens_op = opt.get_init_tokens_op()

            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()

            local_init_op = tf.group(
                tf.local_variables_initializer(), tf.tables_initializer())

            with tf.control_dependencies([local_init_op] if local_init_op is not None else []):
                if is_chief:
                    local_init_op = opt.chief_init_op
                else:
                    local_init_op = opt.local_step_init_op
            ready_for_local_init_op = opt.ready_for_local_init_op

            sv = tf.train.Supervisor(is_chief=is_chief, logdir=self.cnf.get('train_dir', '/tmp'), init_op=init_op, local_init_op=local_init_op, ready_for_local_init_op=ready_for_local_init_op, summary_op=None,
                                     global_step=global_step, saver=saver, save_model_secs=self.cnf.get('save_interval_secs', 600))

            log.info('%s Supervisor' % datetime.now())

            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess_config = tf.ConfigProto(
                gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=self.cnf.get('log_device_placement', False))

            if start_epoch > 1:
                weights_from = "weights/model-epoch-%d.ckpt" % (
                    start_epoch - 1)

            next_summary_time = time.time() + self.cnf.get('save_summaries_secs', 180000000)
            batch_iter_idx = 1
            epoch = 1
            training_history = []
            self.total_network_params()
            self.write_params()
            coord = tf.train.Coordinator()
            with sv.managed_session(server.target, config=sess_config) as sess:
                log.info('Starting Session.')
                if is_chief:
                    if logdir:
                        sv.start_standard_services(sess)
                log.info('Starting Queues.')
                if is_chief:
                    sv.start_queue_runners(sess, chief_queue_runners)
                    chief_queue_runners[0].create_threads(
                        sess, coord=coord, daemon=True, start=True)
                    sess.run(init_tokens_op)

                    if weights_from:
                        self._load_weights(sess, saver, weights_from)

                # threads = sv.start_queue_runners(sess)
                tf.train.start_queue_runners(sess, coord=coord)
                try:
                    while not sv.should_stop():
                        training_losses = []
                        batch_train_sizes = []
                        epoch_start_time = time.time()
                        for iteration in range(n_iters_per_epoch):
                            print(str(iteration))
                            feed_dict_train = {
                                self.learning_rate: learning_rate}
                            start_time = time.time()
                            loss_value, step = sess.run(
                                [train_op, global_step], feed_dict=feed_dict_train)
                            assert not np.isnan(
                                loss_value), 'Model diverged with loss = NaN'
                            if step > self.cnf.get('max_steps', 10000000):
                                log.info(
                                    'Completed max iterations; Stopping Training.')
                                sv.request_stop()
                                break

                            duration = time.time() - start_time

                            if step % 30 == 0:
                                examples_per_sec = self.cnf.get(
                                    'batch_size', 32) / float(duration)
                                format_str = (
                                    'Worker %d: %s: step %d, loss = %.2f (%.1f examples/sec; %.3f  sec/batch)')
                                log.info(format_str % (task_id, datetime.now(
                                ), step, loss_value, examples_per_sec, duration))

                            if is_chief and next_summary_time < time.time():
                                log.info(
                                    'Running Summary operation on the chief.')
                                # summary_str = sess.run(summary_op)
                                # sv.summary_computed(sess, summary_str)
                                log.info('Finished running Summary operation.')

                                # Determine the next time for running the
                                # summary.
                                next_summary_time += self.cnf.get(
                                    'save_summaries_secs', 180)

                            training_losses.append(loss_value)
                            batch_train_sizes.append(
                                self.cnf.get('batch_size_train'))
                            learning_rate = self.lr_policy.batch_update(
                                learning_rate, batch_iter_idx)
                            batch_iter_idx += 1
                            log.debug('4. Training batch %d done.' % iteration)
                        log.info('Learning rate value: %f.' % learning_rate)
                        epoch_training_loss = np.average(
                            training_losses, weights=batch_train_sizes)
                        # epoch_duration = time.time() - epoch_start_time
                        # Validation prediction and metrics
                        validation_losses = []
                        batch_validation_metrics = [
                            [] for _, _ in self.validation_metrics_def]
                        epoch_validation_metrics = []
                        batch_validation_sizes = []
                        for iteration in range(n_val_iters_per_epoch):
                            log.debug(
                                '6. Loading batch %d validation data done.' % iteration)
                            log.debug(
                                '7. Running validation steps without summary...')
                            _validation_metric = sess.run(validation_metric)
                            log.debug(
                                '7. Running validation steps without summary done.')
                            validation_losses.append(_validation_metric[-1])
                            batch_validation_sizes.append(
                                self.cnf.get('batch_size_test'))
                            for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                                batch_validation_metrics[i].append(
                                    _validation_metric[i])

                            log.debug('8. Validation batch %d done' % iteration)

                        epoch_validation_loss = np.average(
                            validation_losses, weights=batch_validation_sizes)
                        for i, (_, _) in enumerate(self.validation_metrics_def):
                            epoch_validation_metrics.append(np.average(batch_validation_metrics[
                                                            i], weights=batch_validation_sizes))

                        custom_metrics_string = [', %s: %.3f' % (name, epoch_validation_metrics[
                                                                 i]) for i, (name, _) in enumerate(self.validation_metrics_def)]
                        custom_metrics_string = ''.join(custom_metrics_string)

                        log.info("Epoch %d [(%s, %s) images, %6.1fs]: t-loss: %.3f, v-loss: %.3f%s" %
                                 (epoch, np.sum(batch_train_sizes), np.sum(batch_validation_sizes), time.time() - epoch_start_time,
                                  epoch_training_loss, epoch_validation_loss, custom_metrics_string))

                        epoch_info = dict(
                            epoch=epoch,
                            training_loss=epoch_training_loss,
                            validation_loss=epoch_validation_loss
                        )

                        training_history.append(epoch_info)

                        log.debug('10. Epoch done. [%d]' % epoch)
                        epoch += 1
                        if is_chief and logdir:
                            if not os.path.exists(logdir):
                                tf.gfile.MakeDirs(logdir)
                            sv.saver.save(
                                sess, logdir, global_step=sv.global_step)
                    sv.stop(threads)
                    sv.coord.request_stop()
                    sv.coord.join(stop_grace_period_secs=0.05)
                except Exception as e:
                    print(e.message)
                    if is_chief:
                        log.info('About to execute sync_clean_up_op!')
                        raise

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

    def _tower_loss(self, scope, model, images, labels, is_training, reuse, is_classification=True):
        if is_training:
            self.training_end_points = model(
                images, is_training=is_training, reuse=reuse)
            if is_classification:
                loss_temp = self._loss_softmax(self.training_end_points[
                    'logits'], labels, is_training)
            else:
                loss_temp = self._loss_regression(self.training_end_points[
                    'logits'], labels, is_training)
            losses = tf.get_collection('losses', scope)
            total_loss = tf.add_n(losses, name='total_loss')
            for l in losses + [total_loss]:
                loss_name = re.sub('%s_[0-9]*/' %
                                   self.cnf['TOWER_NAME'], '', l.op.name)
            return losses, total_loss
        else:
            self.validation_end_points = model(
                images, is_training=is_training, reuse=reuse)
            if is_classification:
                total_loss = self._loss_softmax(self.validation_end_points[
                                                'logits'], labels, is_training)
            else:
                total_loss = self._loss_regression(self.validation_end_points[
                                                   'logits'], labels, is_training)
            self.validation_predictions = self.validation_end_points[
                'predictions']

            return total_loss

    def _setup_model_loss(self, inputs, labels, validation_inputs, validation_labels, is_chief, task_id, num_workers, is_training, scope, initial_lr=0.1, reuse=None, global_step=None, num_replicas_to_aggregate=-1):
        validation_metric = []
        validation_metric_tmp = [[] for _, _ in self.validation_metrics_def]
        self.learning_rate = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder")

        losses, total_loss = self._tower_loss(
            scope, self.model, inputs, labels, is_training, reuse, is_classification=True)
        val_total_loss = self._tower_loss(
            scope, self.model, validation_inputs, validation_labels, False, True, is_classification=True)
        for i, (_, metric_function) in enumerate(self.validation_metrics_def):
            metric_score = metric_function(
                validation_labels, tf.argmax(self.validation_predictions, 1))
            validation_metric_tmp[i].append(metric_score)
        for i, (_, _) in enumerate(self.validation_metrics_def):
            validation_metric.append(sum(validation_metric_tmp[i]))
        validation_metric.append(val_total_loss)

        if is_chief:
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            loss_averages_op = loss_averages.apply(losses + [total_loss])

            with tf.control_dependencies([loss_averages_op]):
                total_loss = tf.identity(total_loss)

        exp_moving_averager = tf.train.ExponentialMovingAverage(
            self.cnf.get('mv_decay', 0.9), global_step)

        variables_to_average = (
            tf.trainable_variables() + tf.moving_average_variables())

        # Create synchronous replica optimizer.
        learning_rate = self.lr_policy.batch_update(initial_lr, 0)
        opt = self._optimizer(learning_rate, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_replicas_to_aggregate,
                                             total_num_replicas=num_workers, variable_averages=exp_moving_averager, variables_to_average=variables_to_average)
        return total_loss, opt, validation_metric

    def create_train_op(self, total_loss, optimizer, global_step=None, update_ops=None, variables_to_train=None, clip_by_global_norm=False, gradient_noise_scale=None, gradient_multipliers=None, gate_gradients=tf.train.Optimizer.GATE_OP, aggregation_method=None, colocate_gradients_with_ops=False):
        """Creates an `Operation` that evaluates the gradients and returns the loss.
        Args:
            total_loss: A `Tensor` representing the total loss.
            optimizer: A tf.Optimizer to use for computing the gradients.
            global_step: A `Tensor` representing the global step variable. If left as
                `_USE_GLOBAL_STEP`, then tf.contrib.framework.global_step() is used.
            update_ops: An optional list of updates to execute. If `update_ops` is
                `None`, then the update ops are set to the contents of the
                `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
                it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
                a warning will be displayed.
            variables_to_train: an optional list of variables to train. If None, it will
                default to all tf.trainable_variables().
            clip_grad_global_norm: A bool, performs gradient clipping using global norm if True
                else performs gradient clipping using local norm.
            gradient_noise_scale: if not None, add noises to the gradients
            gradient_multipliers: if not None, a dict, multiples gradient with given args
            gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
            aggregation_method: Specifies the method used to combine gradient terms.
                Valid values are defined in the class `AggregationMethod`.
            colocate_gradients_with_ops: Whether or not to try colocating the gradients
                with the ops that generated them.
        Returns:
            A `Tensor` that when evaluated, computes the gradients and returns the total
                loss value.
        """
        if global_step is None:
            global_step = tf.get_variable('global_step', shape=[
            ], dtype=tf.int64, initializer=tf.zeros_initializer, trainable=False)

        # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
        global_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        if update_ops is None:
            update_ops = global_update_ops
        else:
            update_ops = set(update_ops)
        if not global_update_ops.issubset(update_ops):
            log.warn(
                'update_ops in create_train_op does not contain all the update_ops in GraphKeys.UPDATE_OPS')

        # Make sure update_ops are computed before total_loss.
        if update_ops:
            with tf.control_dependencies(update_ops):
                barrier = tf.no_op(name='update_barrier')
                total_loss = tf.with_dependencies([barrier], total_loss)

        if variables_to_train is None:
            variables_to_train = tf.trainable_variables()
        else:
            for v in variables_to_train:
                assert v in tf.trainable_variables()

        assert variables_to_train
        if clip_by_global_norm:
            grads_and_vars = self. _clip_grad_global_norms(variables_to_train, total_loss, optimizer, global_norm=8, gate_gradients=gate_gradients,
                                                           gradient_noise_scale=gradient_noise_scale, GATE_GRAPH=2, grad_loss=None, agre_method=aggregation_method, col_grad_ops=colocate_gradients_with_ops)
        else:
            grads_and_vars = optimizer.compute_gradients(total_loss, variables_to_train, gate_gradients=gate_gradients,
                                                         aggregation_method=aggregation_method, colocate_gradients_with_ops=colocate_gradients_with_ops)
            grads_and_vars = self._clip_grad_norms(grads_and_vars, max_norm=8)

        if gradient_multipliers is not None:
            grads_and_vars = self._multiply_gradients(
                grads_and_vars, gradient_multipliers)

        grad_updates = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        with tf.name_scope('train_op'):
            # Make sure total_loss is valid.
            total_loss = tf.check_numerics(
                total_loss, 'LossTensor is inf or nan')

        # Ensure the train_tensor computes grad_updates.
        with tf.control_dependencies([grad_updates]):
            total_loss = tf.identity(total_loss)
        return total_loss
