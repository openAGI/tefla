from __future__ import division, print_function, absolute_import

import os
import re
import pprint
import time
import traceback


import numpy as np
import matplotlib.pyplot as plt

from .base import Base, BaseMixin
from . import summary as summary
from . import logger as log
from ..utils import util
from ..dataset.pascal_voc import PascalVoc
from .losses import segment_loss
from .metrics import compute_hist
import tensorflow as tf

TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'


class SupervisedLearner(Base, BaseMixin):
    """
    Supervised Learner, support data parallelism, multi GPU, accept TFRecords data as input


    Args:
        model: model definition
        cnf: dict, training configs
        training_iterator: iterator to use for training data access, processing and augmentations
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

    def fit(self, data_dir, features_keys=None, weights_from=None, start_epoch=1, summary_every=10, num_classes=15, num_checkpoint_to_keep=None, weights_dir=None, training_set_size=None, dataset_name='cifar10', keep_moving_averages=False):
        """
        Train the model on the specified dataset

        Args:
            data_dir: str, training dataset directory (where tfrecords are staored for training)
            features_keys: a dict, tfrecords keys to datum features
            e.g.:
            features_keys = {
                'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
            weights_from: str, if not None, initializes model from exisiting weights
            training_set_size: int, number of training examples
            dataset_name: a optional, Name of the dataset
            start_epoch: int,  epoch number to start training from
                e.g. for retarining set the epoch number you want to resume training from
            summary_every: int, epoch interval to write summary; higher value means lower frequency
                of summary writing
            keep_moving_averages: a bool, keep moving averages of trainable variables
        """
        self._data_ops(data_dir, standardizer=self.cnf.get('standardizer'))
        self._setup_model_loss(
            keep_moving_averages=keep_moving_averages, num_classes=num_classes)
        if self.is_summary:
            self._setup_summaries(self.grads_and_vars)
        self._setup_misc()
        self._print_info()
        self._train_loop(weights_from, start_epoch, summary_every,
                         num_checkpoint_to_keep=num_checkpoint_to_keep, weights_dir=weights_dir)

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None:
            with tf.control_dependencies([tf.group(*self.update_ops)]):
                self.training_loss = tf.identity(
                    self.training_loss, name='train_loss')

    def _data_ops(self, data_dir, data_dir_val=None, standardizer=None, dataset_name='datarandom'):
        self.data_voc = PascalVoc(name='pascal_voc', data_dir=data_dir, standardizer=standardizer, is_label_filename=True, is_train=True, batch_size=1,
                                  extension='.jpg', capacity=2048, min_queue_examples=512, num_preprocess_threads=8)
        if data_dir_val is None:
            self.data_voc_val = None

    def _train_loop(self, weights_from, start_epoch, summary_every, num_checkpoint_to_keep=None, weights_dir=None):
        saver = tf.train.Saver(max_to_keep=num_checkpoint_to_keep)
        if weights_dir is None:
            weights_dir = "/home/artelus_server/data/tefla/skipunet_v3"
        if not os.path.exists(weights_dir):
            tf.gfile.MakeDirs(weights_dir)
        if self.is_summary:
            training_batch_summary_op = tf.merge_all_summaries(
                key=TRAINING_BATCH_SUMMARIES)
            training_epoch_summary_op = tf.merge_all_summaries(
                key=TRAINING_EPOCH_SUMMARIES)

        if start_epoch > 1:
            weights_from = "weights/model-epoch-%d.ckpt" % (
                start_epoch - 1)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False))
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
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
        n_iters_per_epoch = self.data_voc.n_iters_per_epoch
        self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
        self.total_network_params()
        self.write_graph(sess.graph_def, weights_dir)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for epoch in xrange(start_epoch, self.num_epochs + 1):
                np.random.seed(epoch + seed_delta)
                tf.set_random_seed(epoch + seed_delta)
                tic = time.time()
                training_losses = []
                batch_train_sizes = []

                for batch_num in xrange(1, n_iters_per_epoch + 1):
                    feed_dict_train = {self.learning_rate: learning_rate_value}

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
                    log.info("Batch Num %d [Time: %6.1fs]: t-loss: %.3f" %
                             (batch_num, time.time() - tic, training_loss_e))

                    if self.update_ops is not None:
                        log.debug('3. Running update ops...')
                        sess.run(self.update_ops, feed_dict=feed_dict_train)
                        log.debug('3. Running update ops done.')

                    learning_rate_value = self.lr_policy.batch_update(
                        learning_rate_value, batch_iter_idx)
                    batch_iter_idx += 1
                    log.info("Learning rate: %f " % learning_rate_value)
                    log.debug('4. Training batch %d done.' % batch_num)

                epoch_training_loss = np.average(
                    training_losses, weights=batch_train_sizes)
                log.info("Epoch %d [(%s) images, %6.1fs]: t-loss: %.3f" %
                         (epoch, np.sum(batch_train_sizes), time.time() - tic, epoch_training_loss))
                if self.data_voc_val is None:
                    epoch_validation_loss = 0.0

                # Plot training loss every epoch
                log.debug('5. Writing epoch summary...')
                if self.is_summary:
                    summary_str_train = sess.run(training_epoch_summary_op, feed_dict={
                        self.epoch_loss: epoch_training_loss, self.learning_rate: learning_rate_value})
                    train_writer.add_summary(summary_str_train, epoch)
                    train_writer.flush()
                log.debug('5. Writing epoch summary done.')
                if epoch > 0:
                    saver.save(sess, "%s/model-epoch-%d.ckpt" %
                               (weights_dir, epoch))
                log.info(
                    "Epoch %d [%s training images, %6.1fs]: t-loss: %.3f" %
                    (epoch, np.sum(batch_train_sizes), time.time() - tic,
                     epoch_training_loss)
                )
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

        except Exception:
            traceback.print_exc()
        coord.request_stop()
        coord.join(stop_grace_period_secs=0.05)

    def _process_towers_grads(self, opt, model, is_training=True, reuse=None, loss_type='cross', is_classification=True):
        tower_grads = []
        tower_loss = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(self.cnf.get('num_gpus', 1)):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (self.cnf.get('TOWER_NAME', 'tower'), i)) as scope:
                        images, labels = self.data_voc.get_batch(batch_size=self.cnf['batch_size_train'], height=self.cnf.get(
                            'im_height', 512), width=self.cnf.get('im_width', 512), output_height=self.cnf.get('output_height', 224), output_width=self.cnf.get('output_width', 224))
                        labels = tf.reshape(labels, shape=(-1,))
                        loss = self._tower_loss(scope, model, images, labels, is_training=is_training,
                                                reuse=i > 0, loss_type=loss_type, is_classification=is_classification, gpu_id=i)

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

    def _setup_model_loss(self, keep_moving_averages=False, num_classes=10):
        self.learning_rate = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder")
        optimizer = self._optimizer(self.learning_rate, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        self.grads_and_vars, self.training_loss = self._process_towers_grads(
            optimizer, self.model, is_classification=self.classification, loss_type=self.loss_type)

        if self.clip_norm and not self.clip_by_global_norm:
            self.grads_and_vars = self._clip_grad_norms(
                self.grads_and_vars, max_norm=self.norm_threshold)
        apply_gradients_op = optimizer.apply_gradients(self.grads_and_vars)
        if keep_moving_averages:
            variables_averages_op = self._moving_averages_op()
            with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
                self.train_op = tf.no_op(name='train_op')
        else:
            self.train_op = apply_gradients_op
