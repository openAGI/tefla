from __future__ import division, print_function, absolute_import

import logging
import os
import pprint
import time

import numpy as np
import tensorflow as tf

from ..da.iterator import BatchIterator
from .lr_policy import NoDecayPolicy
from .losses import kappa_log_loss_clipped
from . import summary

logger = logging.getLogger('tefla')

TRAINING_BATCH_SUMMARIES = 'training_batch_summaries'
TRAINING_EPOCH_SUMMARIES = 'training_epoch_summaries'
VALIDATION_BATCH_SUMMARIES = 'validation_batch_summaries'
VALIDATION_EPOCH_SUMMARIES = 'validation_epoch_summaries'


class SupervisedTrainer(object):
    """
    Supervised Trainer class

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

    def __init__(self, model, cnf, training_iterator=BatchIterator(32, False),
                 validation_iterator=BatchIterator(128, False), start_epoch=1, resume_lr=0.01, classification=True, clip_norm=True, n_iters_per_epoch=1094, num_classes=5,  gpu_memory_fraction=0.94, is_summary=False, loss_type='softmax_cross_entropy'):
        self.model = model
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
        self.gpu_memory_fraction = gpu_memory_fraction
        self.is_summary = is_summary
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.label_smoothing = 0.009

    def fit(self, data_set, weights_from=None, start_epoch=1, summary_every=10, weights_dir='weights', verbose=0):
        """
        Train the model on the specified dataset

        Args:
            data_set: dataset instance to use to access data for training/validation
            weights_from: str, if not None, initializes model from exisiting weights
            start_epoch: int,  epoch number to start training from
                e.g. for retarining set the epoch number you want to resume training from
            summary_every: int, epoch interval to write summary; higher value means lower frequency
                of summary writing
            verbose: log level
        """
        self._setup_predictions_and_loss(loss_type=self.loss_type)
        self._setup_optimizer()
        if self.is_summary:
            self._setup_summaries()
        self._setup_misc()
        self._print_info(data_set, verbose)
        self._train_loop(data_set, weights_from, start_epoch, summary_every,
                         verbose, weights_dir=weights_dir)

    def _setup_misc(self):
        self.num_epochs = self.cnf.get('num_epochs', 500)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops is not None:
            with tf.control_dependencies([self.update_ops]):
                self.regularized_training_loss = tf.identity(
                    self.regularized_training_loss)

    def _print_info(self, data_set, verbose):
        logger.info('Config:')
        logger.info(pprint.pformat(self.cnf))
        data_set.print_info()
        logger.info('Max epochs: %d' % self.num_epochs)
        if verbose > 0:
            all_vars = set(tf.global_variables())
            trainable_vars = set(tf.trainable_variables())
            non_trainable_vars = all_vars.difference(trainable_vars)

            logger.info("\n---Trainable vars in model:")
            name_shapes = map(lambda v: (v.name, v.get_shape()), trainable_vars)
            for n, s in sorted(name_shapes, key=lambda ns: ns[0]):
                logger.info('%s %s' % (n, s))

            logger.info("\n---Non Trainable vars in model:")
            name_shapes = map(lambda v: (
                v.name, v.get_shape()), non_trainable_vars)
            for n, s in sorted(name_shapes, key=lambda ns: ns[0]):
                logger.info('%s %s' % (n, s))

        # logger.debug("\n---Number of Regularizable vars in model:")
        # logger.debug(len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

        if verbose > 3:
            all_ops = tf.get_default_graph().get_operations()
            logger.debug("\n---All ops in graph")
            names = map(lambda v: v.name, all_ops)
            for n in sorted(names):
                logger.debug(n)

        _print_layer_shapes(self.training_end_points)

    def _train_loop(self, data_set, weights_from, start_epoch, summary_every, verbose, weights_dir='weights'):
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
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if start_epoch > 1:
                weights_from = "weights/model-epoch-%d.ckpt" % (start_epoch - 1)

            sess.run(tf.global_variables_initializer())
            if weights_from:
                _load_variables(sess, saver, weights_from)

            learning_rate_value = self.lr_policy.initial_lr
            logger.info("Initial learning rate: %f " % learning_rate_value)
            if self.is_summary:
                train_writer, validation_writer = _create_summary_writer(
                    self.cnf.get('summary_dir', '/tmp/tefla-summary'), sess)

            seed_delta = 100
            training_history = []
            batch_iter_idx = 1
            n_iters_per_epoch = len(
                data_set.training_X) // self.training_iterator.batch_size
            self.lr_policy.n_iters_per_epoch = n_iters_per_epoch
            for epoch in xrange(start_epoch, self.num_epochs + 1):
                np.random.seed(epoch + seed_delta)
                tf.set_random_seed(epoch + seed_delta)
                tic = time.time()
                training_losses = []
                batch_train_sizes = []

                for batch_num, (Xb, yb) in enumerate(self.training_iterator(training_X, training_y)):
                    feed_dict_train = {self.inputs: Xb, self.target: self._adjust_ground_truth(yb),
                                       self.learning_rate: learning_rate_value}

                    logger.debug('1. Loading batch %d data done.' % batch_num)
                    if epoch % summary_every == 0 and self.is_summary:
                        logger.debug(
                            '2. Running training steps with summary...')
                        training_predictions_e, training_loss_e, summary_str_train, _ = sess.run(
                            [self.training_predictions, self.regularized_training_loss, training_batch_summary_op,
                             self.optimizer_step],
                            feed_dict=feed_dict_train)
                        train_writer.add_summary(summary_str_train, epoch)
                        train_writer.flush()
                        logger.debug(
                            '2. Running training steps with summary done.')
                        if verbose > 3:
                            logger.debug("Epoch %d, Batch %d training loss: %s" % (
                                epoch, batch_num, training_loss_e))
                            logger.debug("Epoch %d, Batch %d training predictions: %s" %
                                         (epoch, batch_num, training_predictions_e))
                    else:
                        logger.debug(
                            '2. Running training steps without summary...')
                        training_loss_e, _ = sess.run([self.regularized_training_loss, self.optimizer_step],
                                                      feed_dict=feed_dict_train)
                        logger.debug(
                            '2. Running training steps without summary done.')

                    training_losses.append(training_loss_e)
                    batch_train_sizes.append(len(Xb))

                    if self.update_ops is not None:
                        logger.debug('3. Running update ops...')
                        sess.run(self.update_ops, feed_dict=feed_dict_train)
                        logger.debug('3. Running update ops done.')

                    learning_rate_value = self.lr_policy.batch_update(
                        learning_rate_value, batch_iter_idx)
                    batch_iter_idx += 1
                    logger.debug('4. Training batch %d done.' % batch_num)

                epoch_training_loss = np.average(
                    training_losses, weights=batch_train_sizes)

                # Plot training loss every epoch
                logger.debug('5. Writing epoch summary...')
                if self.is_summary:
                    summary_str_train = sess.run(training_epoch_summary_op, feed_dict={
                                                 self.epoch_loss: epoch_training_loss, self.learning_rate: learning_rate_value})
                    train_writer.add_summary(summary_str_train, epoch)
                    train_writer.flush()
                logger.debug('5. Writing epoch summary done.')

                # Validation prediction and metrics
                validation_losses = []
                batch_validation_metrics = [[]
                                            for _, _ in self.validation_metrics_def]
                epoch_validation_metrics = []
                batch_validation_sizes = []
                for batch_num, (validation_Xb, validation_yb) in enumerate(
                        self.validation_iterator(validation_X, validation_y)):
                    feed_dict_validation = {self.validation_inputs: validation_Xb,
                                            self.target: self._adjust_ground_truth(validation_yb)}
                    logger.debug(
                        '6. Loading batch %d validation data done.' % batch_num)

                    if (epoch - 1) % summary_every == 0 and self.is_summary:
                        logger.debug(
                            '7. Running validation steps with summary...')
                        validation_predictions_e, validation_loss_e, summary_str_validate = sess.run(
                            [self.validation_predictions, self.validation_loss,
                                validation_batch_summary_op],
                            feed_dict=feed_dict_validation)
                        validation_writer.add_summary(
                            summary_str_validate, epoch)
                        validation_writer.flush()
                        logger.debug(
                            '7. Running validation steps with summary done.')
                        if verbose > 3:
                            logger.debug(
                                "Epoch %d, Batch %d validation loss: %s" % (epoch, batch_num, validation_loss_e))
                            logger.debug("Epoch %d, Batch %d validation predictions: %s" % (
                                epoch, batch_num, validation_predictions_e))
                    else:
                        logger.debug(
                            '7. Running validation steps without summary...')
                        validation_predictions_e, validation_loss_e = sess.run(
                            [self.validation_predictions, self.validation_loss],
                            feed_dict=feed_dict_validation)
                        logger.debug(
                            '7. Running validation steps without summary done.')
                    validation_losses.append(validation_loss_e)
                    batch_validation_sizes.append(len(validation_Xb))

                    for i, (_, metric_function) in enumerate(self.validation_metrics_def):
                        metric_score = metric_function(
                            validation_yb, validation_predictions_e)
                        batch_validation_metrics[i].append(metric_score)
                    logger.debug('8. Validation batch %d done' % batch_num)

                epoch_validation_loss = np.average(
                    validation_losses, weights=batch_validation_sizes)
                for i, (_, _) in enumerate(self.validation_metrics_def):
                    epoch_validation_metrics.append(
                        np.average(batch_validation_metrics[i], weights=batch_validation_sizes))

                # Write validation epoch summary every epoch
                logger.debug('9. Writing epoch validation summary...')
                if self.is_summary:
                    summary_str_validate = sess.run(validation_epoch_summary_op, feed_dict={
                                                    self.epoch_loss: epoch_validation_loss, self.validation_metric_placeholders: epoch_validation_metrics})
                    validation_writer.add_summary(summary_str_validate, epoch)
                    validation_writer.flush()
                logger.debug('9. Writing epoch validation summary done.')

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
                if verbose > 0:
                    logger.info("Learning rate: %f " % learning_rate_value)
                logger.debug('10. Epoch done. [%d]' % epoch)
            if self.is_summary:
                train_writer.close()
                validation_writer.close()

    def _setup_summaries(self):
        with tf.name_scope('summaries'):
            self.epoch_loss = tf.placeholder(
                tf.float32, shape=[], name="epoch_loss")

            # Training summaries
            tf.summary.scalar('learning rate', self.learning_rate,
                              collections=[TRAINING_EPOCH_SUMMARIES])
            tf.summary.scalar('training (cross entropy) loss', self.epoch_loss,
                              collections=[TRAINING_EPOCH_SUMMARIES])
            if len(self.inputs.get_shape()) == 4:
                summary.summary_image(self.inputs, 'inputs', max_images=10, collections=[
                                      TRAINING_BATCH_SUMMARIES])
            for key, val in self.training_end_points.iteritems():
                summary.summary_activation(val, name=key, collections=[
                                           TRAINING_BATCH_SUMMARIES])
            summary.summary_trainable_params(['scalar', 'histogram', 'norm'], collections=[
                                             TRAINING_BATCH_SUMMARIES])
            summary.summary_gradients(self.grads_and_vars, [
                                      'scalar', 'histogram', 'norm'], collections=[TRAINING_BATCH_SUMMARIES])

            # Validation summaries
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

    def _setup_optimizer(self):
        self.learning_rate = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder")
        # Keep old variable around to load old params, till we need this
        self.obsolete_learning_rate = tf.Variable(
            1.0, trainable=False, name="learning_rate")
        optimizer = self._optimizer(self.learning_rate, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        self.grads_and_vars = optimizer.compute_gradients(
            self.regularized_training_loss, tf.trainable_variables())
        if self.clip_norm:
            self.grads_and_vars = _clip_grad_norms(self.grads_and_vars)
        self.optimizer_step = optimizer.apply_gradients(self.grads_and_vars)

    def _optimizer(self, lr, optname='momentum', decay=0.9, momentum=0.9, epsilon=1e-08, beta1=0.9, beta2=0.999):
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
        return opt

    def _setup_predictions_and_loss(self, loss_type='kappa_log'):
        if self.classification:
            self._setup_classification_predictions_and_loss(loss_type=loss_type)
        else:
            self._setup_regression_predictions_and_loss()

    def _setup_classification_predictions_and_loss(self, loss_type='kappa_log'):
        self.training_end_points = self.model(is_training=True, reuse=None)
        self.inputs = self.training_end_points['inputs']
        training_logits, self.training_predictions = self.training_end_points[
            'logits'], self.training_end_points['predictions']
        self.validation_end_points = self.model(is_training=False, reuse=True)
        self.validation_inputs = self.validation_end_points['inputs']
        validation_logits, self.validation_predictions = self.validation_end_points[
            'logits'], self.validation_end_points['predictions']
        with tf.name_scope('loss'):
            if loss_type == 'kappa_log':
                with tf.name_scope('predictions'):
                    self.target = tf.placeholder(
                        tf.int32, shape=(None, self.num_classes))
                training_loss = kappa_log_loss_clipped(self.training_predictions, self.target, y_pow=2,
                                                       label_smoothing=self.label_smoothing, num_classes=self.num_classes, batch_size=self.training_iterator.batch_size)
                self.validation_loss = kappa_log_loss_clipped(
                    self.validation_predictions, self.target, num_classes=self.num_classes, batch_size=self.training_iterator.batch_size)
            else:
                with tf.name_scope('predictions'):
                    self.target = tf.placeholder(tf.int32, shape=(None,))
                training_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=training_logits, labels=self.target))

                self.validation_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=validation_logits, labels=self.target))

            l2_loss = tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))
            self.regularized_training_loss = training_loss + \
                l2_loss * self.cnf.get('l2_reg', 0.0)

    def _setup_regression_predictions_and_loss(self):
        self.training_end_points = self.model(is_training=True, reuse=None)
        self.inputs = self.training_end_points['inputs']
        self.training_predictions = self.training_end_points['predictions']
        self.validation_end_points = self.model(is_training=False, reuse=True)
        self.validation_inputs = self.validation_end_points['inputs']
        self.validation_predictions = self.validation_end_points['predictions']
        with tf.name_scope('predictions'):
            self.target = tf.placeholder(tf.float32, shape=(None, 1))
        with tf.name_scope('loss'):
            training_loss = tf.reduce_mean(
                tf.square(tf.subtract(self.training_predictions, self.target)))

            self.validation_loss = tf.reduce_mean(
                tf.square(tf.subtract(self.validation_predictions, self.target)))

            l2_loss = tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))
            self.regularized_training_loss = training_loss + \
                l2_loss * self.cnf.get('l2_reg', 0.0)

    def _adjust_ground_truth(self, y):
        if self.loss_type == 'kappa_log':
            return np.eye(self.num_classes)[y]
        else:
            return y if self.classification else y.reshape(-1, 1).astype(np.float32)


def _load_variables(sess, saver, weights_from):
    logger.info("---Loading session/weights from %s..." % weights_from)
    try:
        saver.restore(sess, weights_from)
    except Exception as e:
        logger.info(
            "Unable to restore entire session from checkpoint. Error: %s." % e.message)
        logger.info("Doing selective restore.")
        try:
            reader = tf.train.NewCheckpointReader(weights_from)
            names_to_restore = set(reader.get_variable_to_shape_map().keys())
            variables_to_restore = [v for v in tf.all_variables() if v.name[
                :-2] in names_to_restore]
            logger.info("Loading %d variables: " % len(variables_to_restore))
            for var in variables_to_restore:
                logger.info("Loading: %s %s)" % (var.name, var.get_shape()))
                restorer = tf.train.Saver([var])
                try:
                    restorer.restore(sess, weights_from)
                except Exception as e:
                    logger.info("Problem loading: %s -- %s" %
                                (var.name, e.message))
                    continue
            logger.info("Loaded session/weights from %s" % weights_from)
        except Exception:
            logger.info(
                "Couldn't load session/weights from %s; starting from scratch" % weights_from)
            sess.run(tf.initialize_all_variables())


def _create_summary_writer(summary_dir, sess):
    # if os.path.exists(summary_dir):
    #     shutil.rmtree(summary_dir)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
        os.mkdir(summary_dir + '/training')
        os.mkdir(summary_dir + '/validation')

    train_writer = tf.summary.FileWriter(
        summary_dir + '/training', graph=sess.graph)
    val_writer = tf.summary.FileWriter(
        summary_dir + '/validation', graph=sess.graph)
    return train_writer, val_writer


def variable_summaries(var, name, collections, extensive=True):
    if extensive:
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean,
                          collections=collections, name='var_mean_summary')
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev,
                          collections=collections, name='var_std_summary')
        tf.scalar_summary('max/' + name, tf.reduce_max(var),
                          collections=collections, name='var_max_summary')
        tf.scalar_summary('min/' + name, tf.reduce_min(var),
                          collections=collections, name='var_min_summary')
    return tf.histogram_summary(name, var, collections=collections, name='var_histogram_summary')


def _print_layer_shapes(end_points):
    logger.info("\nModel layer output shapes:")
    for k, v in end_points.iteritems():
        logger.info("%s - %s" % (k, v.get_shape()))


def _clip_grad_norms(gradients_to_variables, max_norm=10):
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
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        grads_and_vars.append((grad, var))
    return grads_and_vars


def clip_grad_global_norms(tvars, loss, opt, global_norm=1, gate_gradients=1, gradient_noise_scale=4.0, GATE_GRAPH=2, grad_loss=None, agre_method=None, col_grad_ops=False):
    """Clips the gradients by the given value.

    Args:
        tvars: trainable variables used for gradint updates
        loss: total loss of the network
        opt: optimizer
        global_norm: the maximum global norm

    Returns:
        A list of clipped gradient to variable pairs.
     """
    var_refs = [v.ref() for v in tvars]
    grads = tf.gradients(loss, var_refs, grad_ys=grad_loss, gate_gradients=(
        gate_gradients == 1), aggregation_method=agre_method, colocate_gradients_with_ops=col_grad_ops)
    if gradient_noise_scale > 1:
        grads = add_scaled_noise_to_gradients(
            list(zip(grads, tvars)), gradient_noise_scale=gradient_noise_scale)
    if gate_gradients == GATE_GRAPH:
        grads = tf.tuple(grads)
    grads, _ = tf.clip_by_global_norm(grads, global_norm)
    grads_and_vars = list(zip(grads, tvars))
    return grads_and_vars


def multiply_gradients(grads_and_vars, gradient_multipliers):
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
        if var in gradient_multipliers or var.op.name in gradient_multipliers:
            key = var if var in gradient_multipliers else var.op.name
            if grad is None:
                raise ValueError('Requested multiple of `None` gradient.')

            if isinstance(grad, tf.IndexedSlices):
                tmp = grad.values * \
                    tf.constant(gradient_multipliers[key], dtype=grad.dtype)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad *= tf.constant(gradient_multipliers[key], dtype=grad.dtype)
        multiplied_grads_and_vars.append((grad, var))
    return multiplied_grads_and_vars


def add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale=10.0):
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
