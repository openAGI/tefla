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


class GenerativeLearner(Base):
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
        super(GenerativeLearner, self).__init__(model, cnf, **kwargs)

    def fit(self, data_set, num_classes=1, weights_from=None, start_epoch=1, summary_every=199, model_name='multiclass_ss', weights_dir='weights'):
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
                self._setup_summaries()
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
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
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
            self._load_weights(sess, weights_from)

        self.total_network_params()
        self.write_graph(sess.graph_def, weights_dir)
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
            e_train_losses = []
            d_real_train_losses = []
            d_fake_train_losses = []
            batch_train_sizes = []
            for batch_num, (Xb, yb) in enumerate(self.training_iterator(training_X, training_y)):
                while Xb.shape[0] != self.cnf['batch_size_train']:
                    batch_pad = - Xb.shape[0] + self.cnf['batch_size_train']
                    Xb = np.vstack((Xb, Xb[0:batch_pad, :, :, :]))
                    yb = np.hstack((yb, yb[0:batch_pad]))
                feed_dict_train = {self.inputs: Xb,
                                   self.learning_rate_d: learning_rate_value, self.learning_rate_g: learning_rate_value, self.learning_rate_e: learning_rate_value}
                log.debug('1. Loading batch %d data done.' % batch_num)
                if epoch % summary_every == 0 and self.is_summary:
                    log.debug('2. Running training steps with summary...')
                    _, _d_loss, d_loss_real, d_loss_fake, summary_str_train = sess.run(
                        [self.train_op_d, self.tower_loss_d, self.tower_loss_d_real, self.tower_loss_d_fake, training_batch_summary_op], feed_dict=feed_dict_train)
                    _, _g_loss = sess.run(
                        [self.train_op_g, self.tower_loss_g], feed_dict=feed_dict_train)
                    _, _e_loss = sess.run([self.train_op_e, self.tower_loss_e
                                           ], feed_dict=feed_dict_train)
                    train_writer.add_summary(summary_str_train, epoch)
                    train_writer.flush()
                    log.debug(
                        '2. Running training steps with summary done.')
                    log.debug("Epoch %d, Batch %d D_loss_real: %s, D_loss_fake: %s,D_loss_class: %s, G_loss: %s" % (
                        epoch, batch_num, _d_loss_real, _d_loss_fake, _d_loss_class, _g_loss))
                else:
                    log.debug(
                        '2. Running training steps without summary...')
                    _, _d_loss_real, _d_loss_fake, _d_loss = sess.run(
                        [self.train_op_d, self.tower_loss_d_real, self.tower_loss_d_fake, self.tower_loss_d], feed_dict=feed_dict_train)
                    _, _g_loss = sess.run([self.train_op_g, self.tower_loss_g
                                           ], feed_dict=feed_dict_train)
                    _, _e_loss = sess.run([self.train_op_e, self.tower_loss_e
                                           ], feed_dict=feed_dict_train)
                    log.debug(
                        '2. Running training steps without summary done.')

                d_train_losses.append(_d_loss)
                g_train_losses.append(_g_loss)
                e_train_losses.append(_e_loss)
                d_real_train_losses.append(_d_loss_real)
                d_fake_train_losses.append(_d_loss_fake)
                batch_train_sizes.append(len(Xb))
                learning_rate_value = self.lr_policy.batch_update(
                    learning_rate_value, batch_iter_idx)
                batch_iter_idx += 1
                log.debug('4. Training batch %d done.' % batch_num)
            d_avg_loss = np.average(
                d_train_losses, weights=batch_train_sizes)
            g_avg_loss = np.average(
                g_train_losses, weights=batch_train_sizes)
            e_avg_loss = np.average(
                e_train_losses, weights=batch_train_sizes)
            d_real_avg_loss = np.average(
                d_real_train_losses, weights=batch_train_sizes)
            d_fake_avg_loss = np.average(
                d_fake_train_losses, weights=batch_train_sizes)
            log.debug("Epoch %d [(%s) images, %6.1fs]:, D_avg_loss: %s, G_avg_loss %s, E-avg_loss %s" %
                      (epoch, np.sum(batch_train_sizes), time.time() - tic, d_avg_loss, g_avg_loss, e_avg_loss))
            print("Epoch %d [(%s) images, %6.1fs]:, D_avg_loss: %s, G_avg_loss %s, E-avg_loss %s" %
                  (epoch, np.sum(batch_train_sizes), time.time() - tic, d_avg_loss, g_avg_loss, e_avg_loss))
            # Plot training loss every epoch
            log.debug('5. Writing epoch summary...')
            if self.is_summary:
                summary_str_train = sess.run(training_epoch_summary_op, feed_dict={
                                             self.epoch_loss: d_avg_loss, self.epoch_loss_g: g_avg_loss, self.learning_rate_d: learning_rate_value, self.learning_rate_g: learning_rate_value})
                train_writer.add_summary(summary_str_train, epoch)
                train_writer.flush()
            log.debug('5. Writing epoch summary done.')

            log.info(
                "Epoch %d [(%s) images, %6.1fs]: d-loss: %.3f, d-real-loss: %.3f, d-fake-loss: %.3f, g-loss: %.3f, e-loss: %.3f" %
                (epoch, np.sum(batch_train_sizes), time.time() - tic,
                 d_avg_loss, d_real_avg_loss, d_fake_avg_loss, g_avg_loss, e_avg_loss)
            )
            epoch_info = dict(
                epoch=epoch,
                d_loss=d_avg_loss,
                g_loss=g_avg_loss,
                e_loss=e_avg_loss,
            )

            training_history.append(epoch_info)
            saver.save(sess, "%s/model-epoch-%d.ckpt" % (weights_dir, epoch))

            learning_rate_value = self.lr_policy.epoch_update(
                learning_rate_value, training_history)
            z_test = self.model.get_z(
                [self.cnf['batch_size_test'], 512], None, name='g_test')
            end_points_G_val = self.model.generator(
                z_test, False, True, batch_size=self.cnf['batch_size_test'])

            util.save_images('generated_images.jpg',
                             sess.run(end_points_G_val['softmax']), width=self.model.image_size[0], height=self.model.image_size[1])

            G = sess.run(end_points_G_val['softmax'])
            cv2.imwrite('generated_image.jpg', G[0, :, :, :] * 50 + 128)

            # Learning rate step decay
        if self.is_summary:
            train_writer.close()
            validation_writer.close()

    def _tower_loss_semi_supervised(self, scope, is_training, reuse, model, inputs, gpu_idx=0, num_classes=11, is_fm_loss=False, gpu_id=0):
        batch_size_train = self.cnf['batch_size_train']
        batch_size_val = self.cnf['batch_size_test']
        z_input = self.model.get_z([batch_size_train, 512], reuse, name='g_z')
        end_points_G = model.generator(
            z_input, is_training, reuse, batch_size_train)
        end_points_E = model.encoder(
            inputs, is_training, reuse)
        z_input_new = model.get_z([batch_size_train, 512], reuse, name='e_Z')
        z_e_input = end_points_E['e_logits1'] + \
            z_input_new * end_points_E['e_logits2']
        end_points_G_E = model.generator(
            z_e_input, is_training, True, batch_size_train)

        if gpu_idx == 0:
            G_means = tf.reduce_mean(
                end_points_G['softmax'], axis=0, keep_dims=True)
            G_vars = tf.reduce_mean(
                tf.square(end_points_G['softmax'] - G_means), axis=0, keep_dims=True)
            G = tf.Print(end_points_G['softmax'], [tf.reduce_mean(G_means), tf.reduce_mean(
                G_vars)], "generator mean and average var", first_n=1)
            G_E_means = tf.reduce_mean(
                end_points_G_E['softmax'], axis=0, keep_dims=True)
            G_E_vars = tf.reduce_mean(
                tf.square(end_points_G_E['softmax'] - G_E_means), axis=0, keep_dims=True)
            G_E = tf.Print(end_points_G_E['softmax'], [tf.reduce_mean(G_E_means), tf.reduce_mean(
                G_E_vars)], "generator mean and average var", first_n=1)
            inputs_means = tf.reduce_mean(inputs, axis=0, keep_dims=True)
            inputs_vars = tf.reduce_mean(
                tf.square(inputs - inputs_means), axis=0, keep_dims=True)
            inputs = tf.Print(inputs, [tf.reduce_mean(inputs_means), tf.reduce_mean(
                inputs_vars)], "image mean and average var", first_n=1)

        try:
            joint_G = tf.concat([G, G_E], 0)
        except Exception:
            joint_G = tf.concat(0, [G, G_E])
        print(joint_G.get_shape())
        end_points_D_IMG = model.discriminator(
            inputs, is_training, reuse, num_classes=num_classes)
        end_points_D_G = model.discriminator(
            joint_G, is_training, True, num_classes=num_classes)

        # For printing layers shape
        if gpu_id == 0:
            self.training_end_points = end_points_D_G
            self.training_end_points.update(end_points_G)
            self.training_end_points.update(end_points_E)
            # self._print_layer_shapes(self.training_end_points, log)

        recon_vs_gan = 1e-6
        d_conv4_2_e = tf.slice(
            end_points_D_G['d_conv4_2'], [batch_size_train, 0, 0, 0], [batch_size_train, 8, 8, 192])
        like_loss = tf.reduce_mean(tf.square(end_points_D_IMG[
            'd_conv4_2'] - d_conv4_2_e)) / 2.
        kl_loss = tf.reduce_mean(-end_points_E['e_logits2'] + .5 * (-1 + tf.exp(
            2. * end_points_E['e_logits2']) + tf.square(end_points_E['e_logits1'])))

        d_label_smooth = self.cnf.get('d_label_smooth',  0.05)
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=end_points_D_IMG['logits'], labels=tf.ones_like(end_points_D_IMG['logits']) - d_label_smooth))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=end_points_D_G['logits'], labels=tf.zeros_like(end_points_D_G['logits'])))
        d_loss = d_loss_fake + d_loss_real

        g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=end_points_D_G['logits'], labels=tf.ones_like(end_points_D_G['logits'])))
        g_loss = g_loss1 + recon_vs_gan * like_loss
        e_loss = kl_loss + like_loss
        return d_loss, g_loss, e_loss, d_loss_real, d_loss_fake, kl_loss, like_loss

    def _get_vars_semi_supervised(self):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('d_')]
        g_vars = [var for var in t_vars if var.name.startswith('g_')]
        e_vars = [var for var in t_vars if var.name.startswith('e_')]
        for x in d_vars:
            assert x not in g_vars and x not in e_vars
        for x in g_vars:
            assert x not in d_vars and x not in e_vars
        for x in e_vars:
            assert x not in d_vars and x not in g_vars
        for x in t_vars:
            assert x in g_vars or x in d_vars or x in e_vars

        return {'d_vars': d_vars, 'g_vars': g_vars, 'e_vars': e_vars}

    def _process_tower_grads(self, d_optimizer, g_optimizer, e_optimizer, model, num_classes=1, is_training=True, reuse=None, update_ops=None):
        tower_grads_d = []
        tower_grads_g = []
        tower_grads_e = []
        tower_loss_d = []
        tower_loss_g = []
        tower_loss_e = []
        tower_loss_d_real = []
        tower_loss_d_fake = []
        tower_loss_kl = []
        tower_loss_like = []
        if self.cnf.get('num_gpus', 1) > 1:
            images_gpus = tf.split(
                self.inputs, self.cnf.get('num_gpus', 1), axis=0)
        else:
            images_gpus = []
            images_gpus.append(self.inputs)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(self.cnf.get('num_gpus', 1)):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (self.cnf.get('TOWER_NAME', 'tower'), i)) as scope:
                        d_loss, g_loss, e_loss, d_loss_real, d_loss_fake, kl_loss, like_loss = self._tower_loss_semi_supervised(
                            scope, is_training, reuse, model, images_gpus[i], num_classes=num_classes, gpu_id=i)
                        tf.get_variable_scope().reuse_variables()
                        # reuse = True

                        # global_update_ops =
                        # set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                        global_update_ops = tf.get_collection(
                            tf.GraphKeys.UPDATE_OPS)
                        if update_ops is None:
                            update_ops = global_update_ops
                        else:
                            update_ops = set(update_ops)
                        # Make sure update_ops are computed before total_loss.
                        if update_ops:
                            with tf.control_dependencies(update_ops):
                                barrier = tf.no_op(name='update_barrier')
                                d_loss = control_flow_ops.with_dependencies(
                                    [barrier], d_loss)
                                g_loss = control_flow_ops.with_dependencies(
                                    [barrier], g_loss)
                                e_loss = control_flow_ops.with_dependencies(
                                    [barrier], e_loss)
                        t_vars = self._get_vars_semi_supervised()
                        if self.clip_by_global_norm:
                            capped_d_grads = self._clip_grad_global_norms(
                                t_vars['d_vars'], d_loss, d_optimizer, gradient_noise_scale=0.0)
                            capped_g_grads = self._clip_grad_global_norms(
                                t_vars['g_vars'], g_loss, g_optimizer, gradient_noise_scale=0.0)
                            capped_e_grads = self._clip_grad_global_norms(
                                t_vars['e_vars'], e_loss, e_optimizer, gradient_noise_scale=0.0)
                        else:
                            capped_d_grads = d_optimizer.compute_gradients(
                                d_loss, t_vars['d_vars'])
                            capped_g_grads = g_optimizer.compute_gradients(
                                g_loss, t_vars['g_vars'])
                            capped_e_grads = g_optimizer.compute_gradients(
                                e_loss, t_vars['e_vars'])
                        tower_grads_d.append(capped_d_grads)
                        tower_grads_g.append(capped_g_grads)
                        tower_grads_e.append(capped_e_grads)
                        tower_loss_d.append(d_loss)
                        tower_loss_g.append(g_loss)
                        tower_loss_e.append(e_loss)
                        tower_loss_d_real.append(d_loss_real)
                        tower_loss_d_fake.append(d_loss_fake)
                        tower_loss_kl.append(kl_loss)
                        tower_loss_like.append(like_loss)

        grads_and_vars_d = self._average_gradients(tower_grads_d)
        grads_and_vars_g = self._average_gradients(tower_grads_g)
        grads_and_vars_e = self._average_gradients(tower_grads_e)
        return grads_and_vars_d, grads_and_vars_g, grads_and_vars_e, sum(tower_loss_d), sum(tower_loss_g), sum(tower_loss_e), sum(tower_loss_d_real), sum(tower_loss_d_fake), sum(tower_loss_kl), sum(tower_loss_like)

    def _setup_model_loss(self, update_ops=None, num_classes=1):
        self.learning_rate_d = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder_generator")
        self.learning_rate_g = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder_discriminator")
        self.learning_rate_e = tf.placeholder(
            tf.float32, shape=[], name="learning_rate_placeholder_encoder")

        d_optimizer = self._optimizer(self.learning_rate_d, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        g_optimizer = self._optimizer(self.learning_rate_g, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        e_optimizer = self._optimizer(self.learning_rate_e, optname=self.cnf.get(
            'optname', 'momentum'), **self.cnf.get('opt_kwargs', {'decay': 0.9}))
        # Get images split the batch across GPUs.
        assert self.cnf['batch_size_train'] % self.cnf.get(
            'num_gpus', 1) == 0, ('Batch size must be divisible by number of GPUs')

        self.inputs = tf.placeholder(tf.float32, shape=(
            None, self.model.image_size[0], self.model.image_size[0], 3), name="input")
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        capped_d_grads, capped_g_grads, capped_e_grads, self.tower_loss_d, self.tower_loss_g, self.tower_loss_e, self.tower_loss_d_real, self.tower_loss_d_fake, self.tower_loss_kl, self.tower_loss_like = self._process_tower_grads(
            d_optimizer, g_optimizer, e_optimizer, self.model, num_classes=num_classes, is_training=True, reuse=None)
        if self.gradient_multipliers is not None:
            with tf.name_scope('multiply_grads'):
                capped_d_grads = self._multiply_gradients(
                    capped_d_grads, self.gradient_multipliers)
        apply_d_gradient_op = d_optimizer.apply_gradients(
            capped_d_grads, global_step=global_step)
        apply_g_gradient_op = g_optimizer.apply_gradients(
            capped_g_grads, global_step=global_step)
        apply_e_gradient_op = e_optimizer.apply_gradients(
            capped_e_grads, global_step=global_step)
        self.train_op_d = control_flow_ops.with_dependencies(
            [apply_d_gradient_op], self.tower_loss_d)
        self.train_op_g = control_flow_ops.with_dependencies(
            [apply_g_gradient_op], self.tower_loss_g)
        self.train_op_e = control_flow_ops.with_dependencies(
            [apply_e_gradient_op], self.tower_loss_e)
