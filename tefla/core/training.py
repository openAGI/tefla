from __future__ import division, print_function, absolute_import

import logging
import os
import pprint
import shutil
import time

import numpy as np
import tensorflow as tf

logger = logging.getLogger('tefla')

TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class StepDecayPolicy(object):
    def update(self, learning_rate, epoch, schedule, sess, verbose):
        if epoch in schedule.keys() and schedule[epoch] is not 'stop':
            sess.run(learning_rate.assign(schedule[epoch]))
            if verbose > -1:
                logger.info("Learning rate changed to: %f " % sess.run(learning_rate))


class SupervisedTrainer(object):
    def __init__(self, model, cnf, training_iterator, validation_iterator, classification=True,
                 lr_decay_policy=StepDecayPolicy()):
        self.model = model
        self.cnf = cnf
        self.training_iterator = training_iterator
        self.validation_iterator = validation_iterator
        self.classification = classification
        self.lr_decay_policy = lr_decay_policy
        self.schedule = self.cnf['schedule']
        self.validation_metrics_def = self.cnf['validation_scores']

    def fit(self, data_set, weights_from=None, start_epoch=1, summary_every=10, verbose=0):
        self._setup_predictions_and_loss()
        self._setup_optimizer()
        self._setup_summaries()
        self._setup_misc()
        self._print_info(data_set, verbose)
        self._train_loop(data_set, weights_from, start_epoch, summary_every,
                         verbose)

    def _setup_misc(self):
        self.num_epochs = dict((v, k) for k, v in self.schedule.iteritems()).get('stop', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None and len(self.update_ops) == 0:
            self.update_ops = None
            # if update_ops is not None:
            #     regularized_training_loss = control_flow_ops.with_dependencies(update_ops, regularized_training_loss)

    def _print_info(self, data_set, verbose):
        logger.info('Config:')
        logger.info(pprint.pformat(self.cnf))
        data_set.print_info()
        logger.info('Max epochs: %d' % self.num_epochs)
        if verbose > 0:
            all_vars = set(tf.all_variables())
            trainable_vars = set(tf.trainable_variables())
            non_trainable_vars = all_vars.difference(trainable_vars)

            logger.debug("\n---Trainable vars in model:")
            name_shapes = map(lambda v: (v.name, v.get_shape()), trainable_vars)
            for n, s in sorted(name_shapes, key=lambda ns: ns[0]):
                logger.debug('%s %s' % (n, s))

            logger.debug("\n---Non Trainable vars in model:")
            name_shapes = map(lambda v: (v.name, v.get_shape()), non_trainable_vars)
            for n, s in sorted(name_shapes, key=lambda ns: ns[0]):
                logger.debug('%s %s' % (n, s))

        logger.debug("\n---Number of Regularizable vars in model:")
        logger.debug(len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

        if verbose > 3:
            all_ops = tf.get_default_graph().get_operations()
            logger.debug("\n---All ops in graph")
            names = map(lambda v: v.name, all_ops)
            for n in sorted(names):
                logger.debug(n)

        print_layer_shapes(self.training_end_points)

    def _train_loop(self, data_set, weights_from, start_epoch, summary_every,
                    verbose):
        files, labels, validation_files, validation_labels = \
            data_set.training_files, data_set.training_labels, data_set.validation_files, data_set.validation_labels
        saver = tf.train.Saver(max_to_keep=None)
        weights_dir = "weights"
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        training_batch_summary_op = tf.merge_all_summaries(key=TRAINING_BATCH_SUMMARIES)
        training_epoch_summary_op = tf.merge_all_summaries(key=TRAINING_EPOCH_SUMMARIES)
        validation_batch_summary_op = tf.merge_all_summaries(key=VALIDATION_BATCH_SUMMARIES)
        validation_epoch_summary_op = tf.merge_all_summaries(key=VALIDATION_EPOCH_SUMMARIES)

        with tf.Session() as sess:
            if start_epoch > 1:
                weights_from = "weights/model-epoch-%d.ckpt" % (start_epoch - 1)

            sess.run(tf.initialize_all_variables())
            if weights_from:
                self._load_variables(sess, saver, weights_from)
                # sess.run(self.learning_rate.assign(0.003))

            logger.info("Initial learning rate: %f " % sess.run(self.learning_rate))
            train_writer, validation_writer = create_summary_writer(self.cnf['summary_dir'], sess)

            seed_delta = 100
            for epoch in xrange(start_epoch, self.num_epochs + 1):
                np.random.seed(epoch + seed_delta)
                tf.set_random_seed(epoch + seed_delta)
                tic = time.time()
                training_losses = []
                batch_train_sizes = []

                for batch_num, (X, y) in enumerate(self.training_iterator(files, labels)):
                    feed_dict_train = {self.inputs: X.transpose(0, 2, 3, 1),
                                       self.target: self._adjust_ground_truth(y)}

                    trace('1. Loading batch %d data done.' % batch_num, verbose)
                    if (epoch - 1) % summary_every == 0 and batch_num < 10:
                        trace('2. Running training steps with summary...', verbose)
                        training_predictions_e, training_loss_e, summary_str_train, _ = sess.run(
                            [self.training_predictions, self.regularized_training_loss, training_batch_summary_op,
                             self.optimizer_step],
                            feed_dict=feed_dict_train)
                        train_writer.add_summary(summary_str_train, epoch)
                        train_writer.flush()
                        trace('2. Running training steps with summary done.', verbose)
                        if verbose > 3:
                            logger.debug("Epoch %d, Batch %d training loss: %s" % (epoch, batch_num, training_loss_e))
                            logger.debug("Epoch %d, Batch %d training predictions: %s" %
                                         (epoch, batch_num, training_predictions_e))
                    else:
                        trace('2. Running training steps without summary...', verbose)
                        training_loss_e, _ = sess.run([self.regularized_training_loss, self.optimizer_step],
                                                      feed_dict=feed_dict_train)
                        trace('2. Running training steps without summary done.', verbose)

                    training_losses.append(training_loss_e)
                    batch_train_sizes.append(len(X))
                    if self.update_ops is not None:
                        trace('3. Running update ops...', verbose)
                        sess.run(self.update_ops, feed_dict=feed_dict_train)
                        trace('3. Running update ops done.', verbose)
                    trace('4. Training batch %d done.' % batch_num, verbose)

                epoch_training_loss = np.average(training_losses, weights=batch_train_sizes)

                # Plot training loss every epoch
                trace('5. Writing epoch summary...', verbose)
                summary_str_train = sess.run(training_epoch_summary_op,
                                             feed_dict={self.epoch_loss: epoch_training_loss})
                train_writer.add_summary(summary_str_train, epoch)
                train_writer.flush()
                trace('5. Writing epoch summary done.', verbose)

                # Validation prediction and metrics
                validation_losses = []
                batch_validation_metrics = [[] for _, _ in self.validation_metrics_def]
                epoch_validation_metrics = []
                batch_validation_sizes = []
                for batch_num, (validation_X, validation_y) in enumerate(
                        self.validation_iterator(validation_files, validation_labels)):
                    feed_dict_validation = {self.inputs: validation_X.transpose(0, 2, 3, 1),
                                            self.target: self._adjust_ground_truth(validation_y)}
                    trace('6. Loading batch %d validation data done.' % batch_num, verbose)

                    if (epoch - 1) % summary_every == 0 and batch_num < 10:
                        trace('7. Running validation steps with summary...', verbose)
                        validation_predictions_e, validation_loss_e, summary_str_validate = sess.run(
                            [self.validation_predictions, self.validation_loss, validation_batch_summary_op],
                            feed_dict=feed_dict_validation)
                        validation_writer.add_summary(summary_str_validate, epoch)
                        validation_writer.flush()
                        trace('7. Running validation steps with summary done.', verbose)
                        if verbose > 3:
                            logger.debug(
                                "Epoch %d, Batch %d validation loss: %s" % (epoch, batch_num, validation_loss_e))
                            logger.debug("Epoch %d, Batch %d validation predictions: %s" % (
                                epoch, batch_num, validation_predictions_e))
                    else:
                        trace('7. Running validation steps without summary...', verbose)
                        validation_predictions_e, validation_loss_e = sess.run(
                            [self.validation_predictions, self.validation_loss],
                            feed_dict=feed_dict_validation)
                        trace('7. Running validation steps without summary done.', verbose)
                    validation_losses.append(validation_loss_e)
                    batch_validation_sizes.append(len(validation_X))

                    for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                        metric_score = metric_function(validation_y, validation_predictions_e)
                        batch_validation_metrics[i].append(metric_score)
                    trace('8. Validation batch %d done' % batch_num, verbose)

                epoch_validation_loss = np.average(validation_losses, weights=batch_validation_sizes)
                for i, (_, _) in enumerate(self.validation_metrics_def):
                    epoch_validation_metrics.append(
                        np.average(batch_validation_metrics[i], weights=batch_validation_sizes))

                # Write validation epoch summary every epoch
                trace('9. Writing epoch validation summary...', verbose)
                summary_str_validate = sess.run(validation_epoch_summary_op,
                                                feed_dict={self.epoch_loss: epoch_validation_loss,
                                                           self.validation_metric_placeholders: epoch_validation_metrics})
                validation_writer.add_summary(summary_str_validate, epoch)
                validation_writer.flush()
                trace('9. Writing epoch validation summary done.', verbose)

                custom_metrics_string = [', %s: %.3f' % (name, epoch_validation_metrics[i]) for i, (name, _) in
                                         enumerate(self.validation_metrics_def)]
                custom_metrics_string = ''.join(custom_metrics_string)

                logger.info(
                    "Epoch %d [(%s, %s) images, %6.1fs]: t-loss: %.3f, v-loss: %.3f%s" %
                    (epoch, np.sum(batch_train_sizes), np.sum(batch_validation_sizes), time.time() - tic,
                     epoch_training_loss,
                     epoch_validation_loss,
                     custom_metrics_string)
                )

                if verbose > 0:
                    logger.info("Learning rate: %f " % sess.run(self.learning_rate))
                saver.save(sess, "%s/model-epoch-%d.ckpt" % (weights_dir, epoch))

                # Learning rate step decay
                self.lr_decay_policy.update(self.learning_rate, epoch, self.schedule, sess, verbose)
                trace('10. Epoch done. [%d]' % epoch, verbose)

            train_writer.close()
            validation_writer.close()

    def _setup_summaries(self):
        with tf.name_scope('summaries'):
            self.epoch_loss = tf.placeholder(tf.float32, shape=[], name="epoch_loss")

            # Training summaries
            tf.scalar_summary('learning rate', self.learning_rate, collections=[TRAINING_EPOCH_SUMMARIES])
            tf.scalar_summary('training (cross entropy) loss', self.epoch_loss,
                              collections=[TRAINING_EPOCH_SUMMARIES])

            tf.image_summary('input', self.inputs, 10, collections=[TRAINING_BATCH_SUMMARIES])
            for key, val in self.training_end_points.iteritems():
                variable_summaries(val, key, collections=[TRAINING_BATCH_SUMMARIES])
            for var in tf.trainable_variables():
                variable_summaries(var, var.op.name, collections=[TRAINING_BATCH_SUMMARIES])
            for grad, var in self.grads_and_vars:
                variable_summaries(var, var.op.name + '/grad', collections=[TRAINING_BATCH_SUMMARIES])

            # Validation summaries
            for key, val in self.validation_end_points.iteritems():
                variable_summaries(val, key, collections=[VALIDATION_BATCH_SUMMARIES])

            tf.scalar_summary('validation loss', self.epoch_loss, collections=[VALIDATION_EPOCH_SUMMARIES])
            self.validation_metric_placeholders = []
            for metric_name, _ in self.validation_metrics_def:
                validation_metric = tf.placeholder(tf.float32, shape=[], name=metric_name.replace(' ', '_'))
                self.validation_metric_placeholders.append(validation_metric)
                tf.scalar_summary(metric_name, validation_metric,
                                  collections=[VALIDATION_EPOCH_SUMMARIES])
            self.validation_metric_placeholders = tuple(self.validation_metric_placeholders)

    def _setup_optimizer(self):
        self.learning_rate = tf.Variable(self.schedule[0], trainable=False, name="learning_rate")
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate,
            momentum=0.9,
            use_nesterov=True)  # .minimize(regularized_training_loss)
        self.grads_and_vars = optimizer.compute_gradients(self.regularized_training_loss, tf.trainable_variables())
        self.optimizer_step = optimizer.apply_gradients(self.grads_and_vars)

    def _setup_predictions_and_loss(self):
        if self.classification:
            self._setup_classification_predictions_and_loss()
        else:
            self._setup_regression_predictions_and_loss()

    def _setup_classification_predictions_and_loss(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, shape=(None, self.cnf['w'], self.cnf['h'], 3), name="input")
        self.training_end_points = self.model(self.inputs, is_training=True, reuse=None)
        training_logits, self.training_predictions = self.training_end_points['logits'], self.training_end_points[
            'predictions']
        self.validation_end_points = self.model(self.inputs, is_training=False, reuse=True)
        validation_logits, self.validation_predictions = self.validation_end_points['logits'], \
                                                         self.validation_end_points[
                                                             'predictions']
        with tf.name_scope('predictions'):
            self.target = tf.placeholder(tf.int32, shape=(None,))
        with tf.name_scope('loss'):
            training_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    training_logits, self.target))

            self.validation_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    validation_logits, self.target))

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.regularized_training_loss = training_loss + l2_loss * self.cnf['l2_reg']

    def _setup_regression_predictions_and_loss(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, shape=(None, self.cnf['w'], self.cnf['h'], 3), name="input")
        self.training_end_points = self.model(self.inputs, is_training=True, reuse=None)
        self.training_predictions = self.training_end_points['predictions']
        self.validation_end_points = self.model(self.inputs, is_training=False, reuse=True)
        self.validation_predictions = self.validation_end_points['predictions']
        with tf.name_scope('predictions'):
            self.target = tf.placeholder(tf.float32, shape=(None, 1))
        with tf.name_scope('loss'):
            training_loss = tf.reduce_mean(
                tf.square(tf.sub(self.training_predictions, self.target)))

            self.validation_loss = tf.reduce_mean(
                tf.square(tf.sub(self.validation_predictions, self.target)))

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.regularized_training_loss = training_loss + l2_loss * self.cnf['l2_reg']

    def _adjust_ground_truth(self, y):
        return y if self.classification else y.reshape(-1, 1).astype(np.float32)

    def _load_variables(self, sess, saver, weights_from):
        logger.info("---Loading session/weights from %s..." % weights_from)
        try:
            saver.restore(sess, weights_from)
        except Exception as e:
            logger.info("Unable to restore entire session from checkpoint. Error: %s." % e.message)
            logger.info("Doing selective restore.")
            try:
                reader = tf.train.NewCheckpointReader(weights_from)
                names_to_restore = set(reader.get_variable_to_shape_map().keys())
                variables_to_restore = [v for v in tf.all_variables() if v.name[:-2] in names_to_restore]
                logger.info("Loading %d variables: " % len(variables_to_restore))
                for var in variables_to_restore:
                    logger.info("Loading: %s %s)" % (var.name, var.get_shape()))
                    restorer = tf.train.Saver([var])
                    try:
                        restorer.restore(sess, weights_from)
                    except Exception as e:
                        logger.info("Problem loading: %s -- %s" % (var.name, e.message))
                        continue
                logger.info("Loaded session/weights from %s" % weights_from)
            except Exception:
                logger.info("Couldn't load session/weights from %s; starting from scratch" % weights_from)
                sess.run(tf.initialize_all_variables())


def trace(msg, verbose):
    logger.debug(msg)


def create_summary_writer(summary_dir, sess):
    # if os.path.exists(summary_dir):
    #     shutil.rmtree(summary_dir)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
        os.mkdir(summary_dir + '/training')
        os.mkdir(summary_dir + '/validation')

    train_writer = tf.train.SummaryWriter(summary_dir + '/training', graph=sess.graph)
    val_writer = tf.train.SummaryWriter(summary_dir + '/validation', graph=sess.graph)
    return train_writer, val_writer


def variable_summaries(var, name, collections, extensive=True):
    if extensive:
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean, collections=collections, name='var_mean_summary')
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev, collections=collections, name='var_std_summary')
        tf.scalar_summary('max/' + name, tf.reduce_max(var), collections=collections, name='var_max_summary')
        tf.scalar_summary('min/' + name, tf.reduce_min(var), collections=collections, name='var_min_summary')
    return tf.histogram_summary(name, var, collections=collections, name='var_histogram_summary')


def print_layer_shapes(end_points):
    logger.info("Model layer output shapes:")
    for k, v in end_points.iteritems():
        logger.info("%s - %s" % (k, v.get_shape()))
