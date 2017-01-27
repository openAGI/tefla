import numpy as np
from random import shuffle
import tensorflow as tf
from tefla.core.rnn_cell import LSTMCell
from tefla.core.layers import fully_connected as fc


class GenerateData(object):

    def __init__(self, num_examples=2**20, seq_length=20, is_shuffle=True, num_test_examples=10000):
        random_data = ['{0:0{seq_length}b}'.format(
            i, seq_length=seq_length) for i in range(num_examples)]
        if is_shuffle:
            shuffle(random_data)
        random_data = [map(int, i) for i in random_data]
        nparray_data = []
        for i in random_data:
            temp_list = []
            for j in i:
                temp_list.append([j])
            nparray_data.append(np.array(temp_list))
        self.data = nparray_data

        self.labels = []
        for i in self.data:
            count = 0
            for j in i:
                if j[0] == 1:
                    count += 1
            temp_list = ([0] * 21)
            temp_list[count] = 1
            self.labels.append(temp_list)

        self.traindata = self.data[num_test_examples:]
        self.trainlabels = self.labels[num_test_examples:]
        self.testdata = self.data[:num_test_examples]
        self.testlabels = self.labels[:num_test_examples]
        self.batch_id = 0

    def next(self, batch_size, train=True):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.traindata[self.batch_id:min(self.batch_id +
                                                       batch_size, len(self.data))])
        batch_labels = (self.trainlabels[self.batch_id:min(self.batch_id +
                                                           batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels


def model(input_data, target, num_hidden=34, is_training=True, reuse=None):
    cell = LSTMCell(num_hidden, reuse)
    val, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)
    logit = fc(last, int(target.get_shape()[1]), is_training, reuse)
    prediction = tf.nn.softmax(logit)
    loss = - \
        tf.reduce_sum(
            target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    error = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(error, tf.float32))
    return prediction, error, train_op


def train(data, batch_size, n_iters_per_epoch, output_size=21, seq_length=20, data_dimension=1, epoch=1000):
    input_data = tf.placeholder(tf.float32, [None, seq_length, data_dimension])
    target = tf.placeholder(tf.float32, [None, output_size])
    prediction, error, train_op = model(input_data, target)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        step = 0
        for step in range(n_iters_per_epoch * epoch):
            batch_x, batch_y = data.next(batch_size)
            sess.run(train_op, feed_dict={input_data: batch_x, target: batch_y})
            print('batch %s' % str(step))
            testdata = data.testdata
            testlabels = data.testlabels
            predict_error = sess.run(
                error, {input_data: testdata, target: testlabels})
            print('Batch {:2d} error {:3.1f}%'.format(
                step + 1, 100 * predict_error))


if __name__ == '__main__':
    batch_size = 100
    data = GenerateData(num_test_examples=10000)
    n_iters_per_epoch = int(len(data.traindata)) / batch_size
    epoch = 1000
    train(data, batch_size, n_iters_per_epoch, epoch=epoch)
