import tensorflow as tf
from tefla.core.special_layers import spatialtransformer
from tefla.core.layers import fully_connected as fc
from tefla.core.layers import conv2d, dropout, softmax, prelu
import numpy as np

"""
Example

Spatial Tranformer Network usage example

"""


class GetData(object):

    def __init__(self):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        self.traindata = mnist.train.images
        self.trainlabels = mnist.train.labels
        self.testdata = mnist.test.images
        self.testlabels = mnist.test.labels
        self.batch_id = 0
        self.batch_id_test = 0

    def next(self, batch_size, train=True):
        if self.batch_id == len(self.traindata):
            self.batch_id = 0
        if self.batch_id + batch_size <= len(self.traindata):
            batch_data = (self.traindata[self.batch_id:min(self.batch_id +
                                                           batch_size, len(self.traindata))])
            batch_labels = (self.trainlabels[self.batch_id:min(self.batch_id +
                                                               batch_size, len(self.traindata))])
            self.batch_id = min(self.batch_id + batch_size, len(self.traindata))
        else:
            batch_data = (self.traindata[-64:])
            batch_labels = (self.trainlabels[-64:])
            self.batch_id = min(self.batch_id + batch_size, len(self.traindata))
        return batch_data, batch_labels

    def next_test(self, batch_size, train=True):
        if self.batch_id_test == len(self.testdata):
            self.batch_id_test = 0
        if self.batch_id + batch_size <= len(self.testdata):
            batch_data = (self.testdata[self.batch_id_test:min(self.batch_id_test +
                                                               batch_size, len(self.testdata))])
            batch_labels = (self.testlabels[self.batch_id_test:min(self.batch_id_test +
                                                                   batch_size, len(self.testdata))])
            self.batch_id_test = min(self.batch_id_test +
                                     batch_size, len(self.testdata))
        else:
            batch_data = (self.testdata[-64:])
            batch_labels = (self.testlabels[-64:])
            self.batch_id_test = min(self.batch_id_test +
                                     batch_size, len(self.testdata))
        return batch_data, batch_labels


def model(x, y, batch_size, is_training=True, reuse=None):
    with tf.variable_scope('model', reuse=reuse):
        x_tensor = tf.reshape(x, [-1, 28, 28, 1])

        fc1 = fc(x, 20, is_training, reuse, name='fc1', activation=None)
        fc1 = tf.tanh(fc1)
        fc1 = dropout(fc1, is_training, drop_p=0.5)
        fc2 = fc(fc1, 6, is_training, reuse, use_bias=False, name='fc2')
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        fc2_b = tf.Variable(initial_value=initial, name='fc2/b')

        fc2 = tf.nn.bias_add(fc2, bias=fc2_b)
        fc2 = tf.tanh(fc2)
        h_trans = spatialtransformer(x_tensor, fc2, batch_size=batch_size)

        conv1 = conv2d(h_trans, 16, is_training, reuse,
                       activation=prelu, name='conv1')
        conv2 = conv2d(conv1, 16, is_training, reuse, stride=(
            2, 2), activation=prelu, name='conv2')
        fcmain = fc(conv2, 1024, is_training, reuse,
                    name='fc', activation=prelu)
        fcmain = dropout(fcmain, is_training, drop_p=0.5)
        logits = fc(fcmain, 10, is_training, reuse,
                    name='logits', activation=None)
        prediction = softmax(logits, 'prediction')

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        opt = tf.train.AdamOptimizer()
        optimizer = opt.minimize(loss)
        # grads = opt.compute_gradients(loss, [fc2_b])

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        return accuracy, loss, optimizer


def main():
    data = GetData()
    batch_size_train = 64
    batch_size_test = 64
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    accuracy, loss, optimizer = model(x, y, batch_size_train,
                                      is_training=True, reuse=None)
    n_epochs = 500
    num_train_examples = data.traindata.shape[0]
    num_test_examples = data.testdata.shape[0]
    import math
    n_iters_per_epoch_train = int(
        math.ceil(num_train_examples / batch_size_train))
    n_iters_per_epoch_test = int(math.ceil(num_test_examples / batch_size_test))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        for epoch in range(n_epochs):
            epoch_acc = 0
            for step in range(n_iters_per_epoch_train):
                batch_x, batch_y = data.next(batch_size_train)
                sess.run([loss, optimizer], feed_dict={x: batch_x, y: batch_y})
            for step in range(n_iters_per_epoch_test):
                batch_x_test, batch_y_test = data.next_test(batch_size_test)
                epoch_acc += sess.run(
                    accuracy, {x: batch_x_test, y: batch_y_test})
            epoch_acc = epoch_acc / n_iters_per_epoch_test
            print('Epoch {:2d} Accuracy {:3.1f}%'.format(
                epoch + 1, 100 * epoch_acc))


if __name__ == '__main__':
    main()
