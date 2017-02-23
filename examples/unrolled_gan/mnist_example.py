"""
original code: https://github.com/iclrreproducer/unrolled_gan
improved version: Mrinal Haloi
contact: mrinalhaloi11@gmail.com
"""
import os
from os.path import expanduser
import numpy as np
import tensorflow as tf
from tefla.core.optimizer import RMSPropunroll
from tefla.core.layers import batch_norm_tf as batch_norm, prelu, fully_connected as dense
from examples.datasets import mnist

from utils import SessionWrap, save_images


def generator(z, is_training, reuse, n_hid=500, isize=28 * 28, use_bn=False):
    fc = dense(z, n_hid, is_training, reuse, w_regularizer=None, name='fc1')
    if use_bn:
        fc = tf.nn.relu(batch_norm(fc, is_training=is_training,
                                   reuse=reuse, name='fc1/bn'))
    else:
        fc = tf.nn.relu(hid, name='fc1/prelu')
    fc = dense(fc, n_hid, is_training, reuse, w_regularizer=None, name='fc2')
    if use_bn:
        fc = tf.nn.relu(batch_norm(fc, is_training=is_training,
                                   reuse=reuse, name='fc2/bn'))
    else:
        fc = rtf.nn.elu(fc, name='fc2/prelu')
    out = tf.nn.sigmoid(dense(fc, isize, is_training, reuse,
                              w_regularizer=None, name='g_out'))
    return out


def discriminator(x, z_size, is_training, reuse, n_hid=500, isize=28 * 28):
    fc = dense(x, n_hid, is_training, reuse, w_normalized=True, name='fc1')
    fc = tf.nn.relu(fc, name='fc1/relu')
    fc = dense(fc, n_hid, is_training, reuse, w_normalized=True, name='fc2')
    fc = tf.nn.relu(fc, name='fc2/relu')
    out = dense(fc, 1, is_training, reuse, name='d_out')
    return out


def discriminator_from_params(x, params, is_training, reuse, isize=28 * 28, n_hid=100):
    fc = dense(x, n_hid, is_training, reuse, w_normalized=True,
               name='fc1', params=params[:2])
    fc = tf.nn.relu(fc, name='fc1/relu')
    fc = dense(fc, n_hid, is_training, reuse, w_normalized=True, name='fc2',
               params=params[2:4])
    fc = tf.nn.relu(fc, name='fc2/relu')
    out = dense(fc, 1, is_training, reuse, name='d_out', params=params[4:])
    return out


def train(loss_d, loss_g, opt_d, opt_g, feed_x, feed_z, z_gen, n_steps, batch_size,
          d_steps=1, g_steps=1, d_pretrain_steps=1,
          session=None, callbacks=[], coord=None):
    with SessionWrap(session) as sess:
        tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())
        for t in range(n_steps):
            for i in range(d_steps):
                z = z_gen()
                _, curr_loss_d = sess.run([opt_d, loss_d], feed_dict={
                    feed_z: z})
            if t > d_pretrain_steps:
                for i in range(g_steps):
                    z = z_gen()
                    _, curr_loss_g = sess.run([opt_g, loss_g], feed_dict={
                        feed_z: z})
            else:
                curr_loss_g = 0.
            for callback in callbacks:
                callback(t, curr_loss_d, curr_loss_g)


def main():
    g = tf.Graph()

    lookahead = 5
    use_bn = True
    g_lr = 0.005
    d_lr = 0.0001

    eps = 1e-6
    batch_size = 128
    z_size = 100
    g_steps = 1
    d_steps = 1
    steps = 100000
    d_pretrain_steps = 1
    isize = 28 * 28
    coord = tf.train.Coordinator()
    with g.as_default():
        z_tf = tf.placeholder(tf.float32, shape=(batch_size, z_size))
        mnistdata = mnist.MnistData(one_hot_labels=False)
        x_tf = tf.reshape(mnistdata.train_batch(
            batch_size)[0], shape=(-1, 28 * 28))
        with tf.variable_scope('G') as scope:
            x_gen = generator(z_tf, True, None, use_bn=use_bn)
            g_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            g_prior = 0.
            for param in g_params:
                g_prior += 0. * tf.reduce_sum(tf.square(param))

        with tf.variable_scope('D') as scope:
            disc_out = discriminator(tf.concat(
                [x_tf, x_gen], 0), z_size, True, None)
            disc_real = disc_out[:batch_size, :]
            disc_fake = disc_out[batch_size:, :]
            d_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([disc_real, disc_fake], 0),
                                                                        labels=tf.concat([tf.ones_like(disc_real),
                                                                                          tf.zeros_like(disc_fake)], 0)))

        optimizer_d = RMSPropunroll(learning_rate=d_lr)
        opt_d = optimizer_d.minimize(loss_d, var_list=d_params)

        opt_vars = None
        next_d_params = d_params
        if lookahead > 0:
            for i in range(lookahead):
                disc_out_g = discriminator_from_params(
                    tf.concat([x_tf, x_gen], 0), next_d_params, True, None)
                disc_real_g = disc_out_g[:batch_size, :]
                disc_fake_g = disc_out_g[batch_size:, :]
                loss_d_tmp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([disc_real_g, disc_fake_g], 0),
                                                                                    labels=tf.concat([tf.ones_like(disc_real_g),
                                                                                                      tf.zeros_like(disc_fake_g)], 0)))

                grads = tf.gradients(loss_d_tmp, next_d_params)
                next_d_params, opt_vars = optimizer_d.unroll(
                    grads, next_d_params, opt_vars=opt_vars)
        else:
            disc_out_g = discriminator_from_params(
                tf.concat([x_tf, x_gen], 0), next_d_params, True, None)
            disc_fake_g = disc_out_g[batch_size:, :]
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake_g, labels=tf.ones_like(disc_fake_g)))

        loss_generator = loss_g
        optimizer_g = RMSPropunroll(learning_rate=g_lr)

        opt_g = optimizer_g.minimize(loss_generator, var_list=g_params)

        session = tf.Session()

        def z_gen():
            return np.random.uniform(-1, 1, size=(batch_size, z_size))

        z_vis = z_gen()

        def logging(t, curr_loss_d, curr_loss_g):
            if t % 100 == 0:
                print("{} loss D = {} loss G = {}".format(
                    t, curr_loss_d, curr_loss_g))
            if t % 500 == 0:
                save_images('samples.png', session.run(
                    x_gen, feed_dict={z_tf: z_vis}))

        train(loss_d, loss_generator, opt_d, opt_g, x_tf, z_tf, z_gen,
              steps, batch_size,
              g_steps=g_steps, d_steps=d_steps, d_pretrain_steps=d_pretrain_steps,
              session=session,
              callbacks=[logging], coord=coord)
        coord.request_stop()
        coord.join(stop_grace_period_secs=0.05)


if __name__ == '__main__':
    main()
