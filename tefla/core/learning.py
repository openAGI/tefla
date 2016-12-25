from __future__ import division, print_function, absolute_import

import os
import re
import pprint
import time

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


class SupervisedTrainer(Base):
    def __init__(self, model, cnf, clip_by_global_norm=False, **kwargs):
        self.clip_by_global_norm = clip_by_global_norm
        super(SupervisedTrainer, self).__init__(self, util.load_module(model), cnf, **kwargs)

    def fit(self, data_set, weights_from=None, start_epoch=1, summary_every=10, keep_moving_averages=False):
        self._setup_model_loss(keep_moving_averages)
        if self.is_summary:
            self._setup_summaries()
        self._setup_misc()
        self._print_info(data_set)
        self._train_loop(data_set, weights_from, start_epoch, summary_every)

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
        name_shapes = map(lambda v: (v.name, v.get_shape()), non_trainable_vars)
        for n, s in sorted(name_shapes, key=lambda ns: ns[0]):
            log.info('%s %s' % (n, s))

        all_ops = tf.get_default_graph().get_operations()
        log.debug("\n---All ops in graph")
        names = map(lambda v: v.name, all_ops)
        for n in sorted(names):
            log.debug(n)

        self._print_layer_shapes(self.training_end_points, log)

    def _train_loop(self, data_set, weights_from, start_epoch, summary_every):
        training_X, training_y, validation_X, validation_y = \
            data_set.training_X, data_set.training_y, data_set.validation_X, data_set.validation_y
        print(training_y)
        saver = tf.train.Saver(max_to_keep=None)
        weights_dir = "weights"
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
        if self.is_summary:
            training_batch_summary_op = tf.merge_all_summaries(key=TRAINING_BATCH_SUMMARIES)
            training_epoch_summary_op = tf.merge_all_summaries(key=TRAINING_EPOCH_SUMMARIES)
            validation_batch_summary_op = tf.merge_all_summaries(key=VALIDATION_BATCH_SUMMARIES)
            validation_epoch_summary_op = tf.merge_all_summaries(key=VALIDATION_EPOCH_SUMMARIES)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if start_epoch > 1:
                weights_from = "weights/model-epoch-%d.ckpt" % (start_epoch - 1)

            sess.run(tf.initialize_all_variables())
            if weights_from:
                self._load_weights(sess, saver, weights_from)

            learning_rate_value = self.lr_policy.initial_lr
            log.info("Initial learning rate: %f " % learning_rate_value)
            if self.is_summary:
                train_writer, validation_writer = summary.create_summary_writer(self.cnf.get('summary_dir', '/tmp/tefla-summary'), sess)

            seed_delta = 100
            training_history = []
            batch_iter_idx = 1
            n_iters_per_epoch = len(data_set.training_X) // self.training_iterator.batch_size
            self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
            for epoch in xrange(start_epoch, self.num_epochs + 1):
                np.random.seed(epoch + seed_delta)
                tf.set_random_seed(epoch + seed_delta)
                tic = time.time()
                training_losses = []
                batch_train_sizes = []

                for batch_num, (Xb, yb) in enumerate(self.training_iterator(training_X, training_y)):
                    feed_dict_train = {self.inputs: Xb, self.labels: self._adjust_ground_truth(yb),
                                       self.learning_rate: learning_rate_value}

                    log.debug('1. Loading batch %d data done.' % batch_num)
                    if epoch % summary_every == 0 and self.is_summary:
                        log.debug('2. Running training steps with summary...')
                        training_predictions_e, training_loss_e, summary_str_train, _ = sess.run(
                            [self.training_predictions, self.regularized_training_loss, training_batch_summary_op,
                             self.train_op],
                            feed_dict=feed_dict_train)
                        train_writer.add_summary(summary_str_train, epoch)
                        train_writer.flush()
                        log.debug('2. Running training steps with summary done.')
                        log.debug("Epoch %d, Batch %d training loss: %s" % (epoch, batch_num, training_loss_e))
                        log.debug("Epoch %d, Batch %d training predictions: %s" % (epoch, batch_num, training_predictions_e))
                    else:
                        log.debug('2. Running training steps without summary...')
                        training_loss_e, _ = sess.run([self.regularized_training_loss, self.train_op],
                                                      feed_dict=feed_dict_train)
                        log.debug('2. Running training steps without summary done.')

                    training_losses.append(training_loss_e)
                    batch_train_sizes.append(len(Xb))

                    if self.update_ops is not None:
                        log.debug('3. Running update ops...')
                        sess.run(self.update_ops, feed_dict=feed_dict_train)
                        log.debug('3. Running update ops done.')

                    learning_rate_value = self.lr_policy.batch_update(learning_rate_value, batch_iter_idx)
                    batch_iter_idx += 1
                    log.debug('4. Training batch %d done.' % batch_num)

                epoch_training_loss = np.average(training_losses, weights=batch_train_sizes)

                # Plot training loss every epoch
                log.debug('5. Writing epoch summary...')
                if self.is_summary:
                    summary_str_train = sess.run(training_epoch_summary_op, feed_dict={self.epoch_loss: epoch_training_loss, self.learning_rate: learning_rate_value})
                    train_writer.add_summary(summary_str_train, epoch)
                    train_writer.flush()
                log.debug('5. Writing epoch summary done.')

                # Validation prediction and metrics
                validation_losses = []
                batch_validation_metrics = [[] for _, _ in self.validation_metrics_def]
                epoch_validation_metrics = []
                batch_validation_sizes = []
                for batch_num, (validation_Xb, validation_yb) in enumerate(
                        self.validation_iterator(validation_X, validation_y)):
                    feed_dict_validation = {self.validation_inputs: validation_Xb,
                                            self.validation_labels: self._adjust_ground_truth(validation_yb)}
                    log.debug('6. Loading batch %d validation data done.' % batch_num)

                    if (epoch - 1) % summary_every == 0 and self.is_summary:
                        log.debug('7. Running validation steps with summary...')
                        validation_predictions_e, validation_loss_e, summary_str_validate = sess.run(
                            [self.validation_predictions, self.validation_loss, validation_batch_summary_op],
                            feed_dict=feed_dict_validation)
                        validation_writer.add_summary(summary_str_validate, epoch)
                        validation_writer.flush()
                        log.debug('7. Running validation steps with summary done.')
                        log.debug(
                            "Epoch %d, Batch %d validation loss: %s" % (epoch, batch_num, validation_loss_e))
                        log.debug("Epoch %d, Batch %d validation predictions: %s" % (
                            epoch, batch_num, validation_predictions_e))
                    else:
                        log.debug('7. Running validation steps without summary...')
                        validation_predictions_e, validation_loss_e = sess.run(
                            [self.validation_predictions, self.validation_loss],
                            feed_dict=feed_dict_validation)
                        log.debug('7. Running validation steps without summary done.')
                    validation_losses.append(validation_loss_e)
                    batch_validation_sizes.append(len(validation_Xb))

                    for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                        metric_score = metric_function(validation_yb, validation_predictions_e)
                        batch_validation_metrics[i].append(metric_score)
                    log.debug('8. Validation batch %d done' % batch_num)

                epoch_validation_loss = np.average(validation_losses, weights=batch_validation_sizes)
                for i, (_, _) in enumerate(self.validation_metrics_def):
                    epoch_validation_metrics.append(
                        np.average(batch_validation_metrics[i], weights=batch_validation_sizes))

                # Write validation epoch summary every epoch
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
                     epoch_training_loss,
                     epoch_validation_loss,
                     custom_metrics_string)
                )

                saver.save(sess, "%s/model-epoch-%d.ckpt" % (weights_dir, epoch))

                epoch_info = dict(
                    epoch=epoch,
                    training_loss=epoch_training_loss,
                    validation_loss=epoch_validation_loss
                )

                training_history.append(epoch_info)

                learning_rate_value = self.lr_policy.epoch_update(learning_rate_value, training_history)
                log.info("Learning rate: %f " % learning_rate_value)
                log.debug('10. Epoch done. [%d]' % epoch)
            if self.is_summary:
                train_writer.close()
                validation_writer.close()

    def _loss_softmax(self, logits, labels, is_training):
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

    def _loss_regression(self, logits, labels, is_training):
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

        return total_loss

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _process_towers_grads(self, opt, model, is_training=True, reuse=None, is_classification=True):
        tower_grads = []
        images_gpus = tf.split(0, self.cnf['num_gpus'], self.inputs)
        labels_gpus = tf.split(0, self.cnf['num_gpus'], self.labels)
        for i in xrange(self.cnf['num_gpus']):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (self.cnf['TOWER_NAME'], i)) as scope:
                    loss = self._tower_loss(scope, model, images_gpus[i], labels_gpus[i], is_training=is_training, reuse=reuse, is_classification=is_classification)

                    tf.get_variable_scope().reuse_variables()
                    reuse = True
                    if self.clip_by_global_norm:
                        grads_and_vars = self._clip_grad_global_norms(tf.trainable_variables(), loss, opt, global_norm=self.norm_threshold, gradient_noise_scale=0.0)
                    else:
                        grads_and_vars = opt.compute_gradients(loss)
                    tower_grads.append(grads_and_vars)

        grads_and_vars = self._average_gradients(tower_grads)

        return grads_and_vars, loss

    def _process_towers_loss(self, opt, model, is_training=False, reuse=True, is_classification=True):
        tower_loss = []
        images_gpus = tf.split(0, self.cnf['num_gpus'], self.validation_inputs)
        labels_gpus = tf.split(0, self.cnf['num_gpus'], self.validation_labels)
        for i in xrange(self.cnf['num_gpus']):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (self.cnf['TOWER_NAME'], i)) as scope:
                    loss = self._tower_loss(scope, model, images_gpus[i], labels_gpus[i], is_training=is_training, reuse=reuse, is_classification=is_classification)
                    tower_loss.append(loss)

        return sum(tower_loss)

    def _adjust_ground_truth(self, y):
        return y if self.classification else y.reshape(-1, 1).astype(np.float32)

    def _setup_model_loss(self, keep_moving_averages=False):
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")
        # Keep old variable around to load old params, till we need this
        self.obsolete_learning_rate = tf.Variable(1.0, trainable=False, name="learning_rate")
        optimizer = self._optimizer(self.learning_rate, optname=self.cnf.get('optname', 'momentum'), **self.cnf.get('opt_kwargs'))
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.model.crop_size[0], self.model.crop_size[1], 3), name="input")
        self.labels = tf.placeholder(tf.int32, shape=(None,))
        self.validation_inputs = tf.placeholder(tf.float32, shape=(None, self.model.crop_size[0], self.model.crop_size[1], 3), name="validation_input")
        self.validation_labels = tf.placeholder(tf.int32, shape=(None,))
        self.grads_and_vars, self.training_loss = self._process_towers_grads(optimizer, self.model, is_classification=self.classification)
        self.validation_loss = self._process_towers_loss(optimizer, self.model, is_classification=self.classification)

        if self.clip_norm and not self.clip_by_global_norm:
            self.grads_and_vars = self._clip_grad_norms(self.grads_and_vars, max_norm=self.norm_threshold)
        apply_gradients_op = optimizer.apply_gradients(self.grads_and_vars)
        if keep_moving_averages:
            variables_averages_op = self._moving_averages_op()
            with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
                self.train_op = tf.no_op(name='train')
        else:
            self.train_op = apply_gradients_op