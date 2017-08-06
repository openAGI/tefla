# -------------------------------------------------------------------#
# Tool to Visualize activations
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
import click
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.misc
from tefla.utils import util
from tefla.da.standardizer import AggregateStandardizerTF


def plot_activation(activations):
    filters = activations.shape[3]
    plt.figure(1, figsize=(200, 200))
    n_columns = 1
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(activations[0, :, :, i],
                   interpolation="nearest", cmap="gray")
    plt.show()


def get_activation(sess, layer, inputs, data):
    activations = sess.run(layer, {inputs: data})
    # activations = activations[:, :, :, 0]
    # plot_activation(activations)
    return activations


@click.command()
@click.option('--model', default='models/c_multiclass.py', show_default=True, help='Relative path to model.')
@click.option('--model_cnf', default=None, show_default=True,
              help='Relative path to training config file.')
@click.option('--layer_name', default='conv2_1', show_default=True, help='Relative path to training config file.')
@click.option('--data_path', help='Directory with Test Images')
@click.option('--weights_from', help='Path to initial weights file.')
def load_model(model, model_cnf, weights_from, layer_name, data_path):
    model_def = util.load_module(model)
    cnf = util.load_module(model_cnf).cnf
    standardizer = cnf['standardizer']
    model = model_def.model
    sess = tf.Session()
    try:
        end_points = model(is_training=False, reuse=None)
        saver = tf.train.Saver()
        print('Loading weights from: %s' % weights_from)
        saver.restore(sess, weights_from)
    except Exception:
        print('not loaded')

    inputs = end_points['inputs']
    layer = end_points[layer_name]
    data = tf.read_file(data_path)
    data = tf.to_float(tf.image.decode_jpeg(data))
    data = standardizer(data, False)
    data = tf.transpose(data, perm=[1, 0, 2])
    data_ = data.eval(session=sess)
    data_ = scipy.misc.imresize(data_, size=(
        448, 448), interp='cubic')
    data_ = np.expand_dims(data_, 0)
    acti = get_activation(sess, layer, inputs, data_)
    acti = np.mean(acti, 3).squeeze()
    # acti = np.asarray(acti.transpose(1, 0), dtype=np.float32)
    plt.imshow(acti)
    plt.show()
    # print(acti)
    """
    # upsampled = sess.run(end_points['upscore32'], {inputs: data_})
    # upsampled = np.argmax(upsampled, axis=3)
    # np.save(upsampled, 'upsampled.npy')
    # print(upsampled.shape)
    # plt.imshow(np.squeeze(upsampled), cmap=plt.cm.Greys)
    # plt.show()
    """


if __name__ == '__main__':
    load_model()
