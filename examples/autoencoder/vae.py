"""Categorical VAE with Gumbel-Softmax

simple impementations of https://arxiv.org/abs/1611.01144
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tefla.core.layers import gumbel_softmax, fully_connected, _flatten as flatten, stack
import numpy as np

Bernoulli = tf.contrib.distributions.Bernoulli


def encoder(x, K=10, N=30):
    # variational posterior q(y|x), i.e. the encoder (shape=(batch_size,200))
    net = stack(x, fully_connected, [512, 256], True, None, name='fcencoder')
    # unnormalized logits for N separate K-categorical distributions
    # (shape=(batch_size*N,K))
    logits_y = tf.reshape(fully_connected(
        net, K * N, True, None, activation=None, name='logitsencoder'), [-1, K])
    q_y = tf.nn.softmax(logits_y)
    log_q_y = tf.log(q_y + 1e-20)

    return logits_y, q_y, log_q_y


def decoder(tau, logits_y, K, N):
    y = tf.reshape(gumbel_softmax(logits_y, tau, hard=False), [-1, N, K])
    # generative model p(x|y), i.e. the decoder (shape=(batch_size,200))
    net = stack(flatten(y), fully_connected, [256, 512], True, None, name='fcdecoder')
    logits_x = fully_connected(net, 784, True, None, activation=None, name='logitsdecoder')
    # (shape=(batch_size,784))
    p_x = Bernoulli(logits=logits_x)
    return p_x


def create_train_op(x, lr, q_y, log_q_y, p_x, K, N):
    kl_tmp = tf.reshape(q_y * (log_q_y - tf.log(1.0 / K)), [-1, N, K])
    KL = tf.reduce_sum(kl_tmp, [1, 2])
    elbo = tf.reduce_sum(p_x.log_prob(x), 1) - KL

    loss = tf.reduce_mean(-elbo)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    return train_op, loss


def train():
    K = 10  # number of classes
    N = 30  # number of categorical distributions

    # Encoder
    # input image x (shape=(batch_size,784))
    inputs = tf.placeholder(tf.float32, [None, 784], name='inputs')
    tau = tf.placeholder(tf.float32, [], name='temperature')
    lr = tf.placeholder(tf.float32, [], name='lr_value')
    # get data
    data = input_data.read_data_sets('/tmp/', one_hot=True).train
    logits_y, q_y, log_q_y = encoder(inputs, K=10, N=30)
    p_x = decoder(tau, logits_y, K, N)
    train_op, loss = create_train_op(inputs, lr, q_y, log_q_y, p_x, K, N)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    BATCH_SIZE = 100
    NUM_ITERS = 50000
    tau0 = 1.0  # initial temperature
    np_temp = tau0
    np_lr = 0.001
    ANNEAL_RATE = 0.00003
    MIN_TEMP = 0.5

    dat = []
    sess = tf.InteractiveSession()
    sess.run(init_op)
    for i in range(1, NUM_ITERS):
        np_x, np_y = data.next_batch(BATCH_SIZE)
        _, np_loss = sess.run([train_op, loss], {inputs: np_x, lr: np_lr, tau: np_temp})
        if i % 100 == 1:
            dat.append([i, np_temp, np_loss])
        if i % 1000 == 1:
            np_temp = np.maximum(tau0 * np.exp(-ANNEAL_RATE * i), MIN_TEMP)
            np_lr *= 0.9
        if i % 5000 == 1:
            print('Step %d, ELBO: %0.3f' % (i, -np_loss))


if __name__ == '__main__':
    train()
