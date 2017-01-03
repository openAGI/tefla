from __future__ import division, print_function, absolute_import

import os
import re
import pprint
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from tefla.core.base import Base
import tefla.core.summary as summary
import tefla.core.logger as log
from tefla.utils import util


TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class DistSupervisedTrainer(Base):
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
        super(DistSupervisedTrainer, self).__init__(self, util.load_module(model), cnf, **kwargs)

    def train(self, train_op, task_id, target, dataset, cluster_spec, is_training, start_epoch=1, reuse=None, num_replicas_to_aggregate=-1):
        num_workers = len(cluster_spec.as_dict()['worker'])
        num_parameter_servers = len(cluster_spec.as_dict()['ps'])
        if num_replicas_to_aggregate == -1:
            num_replicas_to_aggregate = num_workers
        else:
            num_replicas_to_aggregate = num_replicas_to_aggregate

        assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and num_parameter_servers must be > 0.')

        is_chief = (task_id == 0)

        # Ops are assigned to worker by default.
        with tf.device('/job:worker/task:%d' % task_id) as scope:
            with tf.device(tf.replica_device_setter(cluster=cluster_spec)):
                # with tf.device(tf.replica_device_setter(ps_device='/job:ps/task:%d' % task_id)):
                global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64, initializer=tf.zeros_initializer, trainable=False)
                initial_lr = self.lr_policy.initial_lr
                total_loss, opt = self._setup_model_loss(self, dataset, is_chief, task_id, num_workers, is_training, scope, initial_lr=initial_lr, reuse=None, global_step=None, num_replicas_to_aggregate=-1)

                chief_queue_runners = [opt.get_chief_queue_runner()]
                init_tokens_op = opt.get_init_tokens_op()
                clean_up_op = opt.get_clean_up_op()

                saver = tf.train.Saver()
                init_op = tf.initialize_all_variables()

                sv = tf.train.Supervisor(is_chief=is_chief, logdir=self.cnf.get('train_dir', '/tmp'), init_op=init_op, summary_op=None, global_step=global_step, saver=saver, save_model_secs=self.cnf.get('save_interval_secs', 600))

                log.info('%s Supervisor' % datetime.now())

                sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=self.cnf('log_device_placement', False))

                sess = sv.prepare_or_wait_for_session(target, config=sess_config)

                queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
                sv.start_queue_runners(sess, queue_runners)
                tf.logging.info('Started %d queues for processing input data.', len(queue_runners))

                if is_chief:
                    sv.start_queue_runners(sess, chief_queue_runners)
                    sess.run(init_tokens_op)
                    if start_epoch > 1:
                        weights_from = "weights/model-epoch-%d.ckpt" % (start_epoch - 1)

                    if weights_from:
                        self._load_weights(sess, saver, weights_from)

                next_summary_time = time.time() + self.cnf.get('save_summaries_secs', 180000000)
                while not sv.should_stop():
                    try:
                        start_time = time.time()
                        loss_value, step = sess.run([train_op, global_step])
                        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                        if step > self.cnf.get('max_steps', 10000000):
                            break
                        duration = time.time() - start_time

                        if step % 30 == 0:
                            examples_per_sec = self.cnf.get('batch_size', 32) / float(duration)
                            format_str = ('Worker %d: %s: step %d, loss = %.2f (%.1f examples/sec; %.3f  sec/batch)')
                            log.info(format_str % (task_id, datetime.now(), step, loss_value, examples_per_sec, duration))

                        if is_chief and next_summary_time < time.time():
                            log.info('Running Summary operation on the chief.')
                            # summary_str = sess.run(summary_op)
                            # sv.summary_computed(sess, summary_str)
                            log.info('Finished running Summary operation.')

                        # Determine the next time for running the summary.
                        next_summary_time += self.cnf.get('save_summaries_secs', 180)
                    except Exception as e:
                        print(e.message)
                        if is_chief:
                            log.info('About to execute sync_clean_up_op!')
                        sess.run(clean_up_op)
                        raise

                sv.stop()

                if is_chief:
                    if not os.path.exists(self.weights_dir):
                        os.mkdir(self.weights_dir)
                    saver.save(sess, os.path.join(self.weights_dir, 'model.ckpt'), global_step=global_step)

    def _loss_regression(self, logits, labels, is_training):
        labels = tf.cast(labels, tf.int64)
        sq_loss = tf.square(tf.sub(logits, labels), name='regression loss')
        sq_loss_mean = tf.reduce_mean(sq_loss, name='regression')
        if is_training:
            tf.add_to_collection('losses', sq_loss_mean)

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            l2_loss = l2_loss * self.cnf.get('l2_reg', 0.0)
            tf.add_to_collection('losses', l2_loss)

            return tf.add_n(tf.get_collection('losses'), name='total_loss')
        else:
            return sq_loss_mean

    def _loss_softmax(self, logits, labels, is_training):
        labels = tf.cast(labels, tf.int64)
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_loss')
        ce_loss_mean = tf.reduce_mean(ce_loss, name='cross_entropy')
        if is_training:
            tf.add_to_collection('losses', ce_loss_mean)

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            l2_loss = l2_loss * self.cnf.get('l2_reg', 0.0)
            tf.add_to_collection('losses', l2_loss)

            return tf.add_n(tf.get_collection('losses'), name='total_loss')
        else:
            return ce_loss_mean

    def _tower_loss(self, scope, model, images, labels, is_training, resue, is_classification=True):
        if is_training:
            self.training_end_points = model(images, is_training=is_training, reuse=resue)
            if is_classification:
                _, = self._loss_softmax(self.training_end_points['logits'], labels, is_training)
            else:
                _, = self._loss_regression(self.training_end_points['logits'], labels, is_training)
            losses = tf.get_collection('losses', scope)
            total_loss = tf.add_n(losses, name='total_loss')
            for l in losses + [total_loss]:
                loss_name = re.sub('%s_[0-9]*/' % self.cnf['TOWER_NAME'], '', l.op.name)
                tf.scalar_summary(loss_name, l)
        else:
            self.validation_end_points = model(images, is_training=is_training, reuse=resue)
            if is_classification:
                total_loss = self._loss_softmax(self.validation_end_points['logits'], labels, is_training)
            else:
                total_loss = self._loss_regression(self.validation_end_points['logits'], labels, is_training)

        return losses, total_loss

    def _setup_model_loss(self, dataset, is_chief, task_id, num_workers, is_training, scope, initial_lr=0.1, reuse=None, global_step=None, num_replicas_to_aggregate=-1):
        # TODO define image_processing
        from tefla.da.future import image_processing
        images, labels = image_processing.distorted_inputs(dataset, batch_size=self.cnf.get('batch_size', 32), num_preprocess_threads=self.cnf.get('num_preprocess_threads', 8))

        losses, total_loss = self._tower_loss(scope, self.model, images, labels, is_training, reuse, is_classification=True)

        if is_chief:
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            loss_averages_op = loss_averages.apply(losses + [total_loss])

            with tf.control_dependencies([loss_averages_op]):
                total_loss = tf.identity(total_loss)

        exp_moving_averager = tf.train.ExponentialMovingAverage(self.cnf.get('mv_decay', 0.9), global_step)

        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())

        # Create synchronous replica optimizer.
        learning_rate = self.lr_policy.batch_update(initial_lr, global_step)
        opt = self._optimizer(learning_rate, optname=self.cnf.get('optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_replicas_to_aggregate, replica_id=task_id, total_num_replicas=num_workers, variable_averages=exp_moving_averager, variables_to_average=variables_to_average)
        return total_loss, opt

    def create_train_op(self, total_loss, optimizer, global_step=None, update_ops=None, variables_to_train=None, clip_grad_global_norm=False, gradient_noise_scale=None, gradient_multipliers=None, gate_gradients=tf.Optimizer.GATE_OP, aggregation_method=None, colocate_gradients_with_ops=False):
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
            global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64, initializer=tf.zeros_initializer, trainable=False)

        # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
        global_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        if update_ops is None:
            update_ops = global_update_ops
        else:
            update_ops = set(update_ops)
        if not global_update_ops.issubset(update_ops):
            log.warn('update_ops in create_train_op does not contain all the update_ops in GraphKeys.UPDATE_OPS')

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
        if clip_grad_global_norm:
            grads_and_vars = self. _clip_grad_global_norms(variables_to_train, total_loss, optimizer, global_norm=8, gate_gradients=gate_gradients, gradient_noise_scale=gradient_noise_scale, GATE_GRAPH=2, grad_loss=None, agre_method=aggregation_method, col_grad_ops=colocate_gradients_with_ops)
        else:
            grads_and_vars = optimizer.compute_gradients(total_loss, variables_to_train, gate_gradients=gate_gradients, aggregation_method=aggregation_method, colocate_gradients_with_ops=colocate_gradients_with_ops)
            grads_and_vars = self._clip_grad_norms(grads_and_vars, max_norm=8)

        if gradient_multipliers is not None:
            grads_and_vars = self._multiply_gradients(grads_and_vars, gradient_multipliers)

        grad_updates = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.name_scope('train_op'):
            # Make sure total_loss is valid.
            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan')

        # Ensure the train_tensor computes grad_updates.
        return tf.with_dependencies([grad_updates], total_loss)
