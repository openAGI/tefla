import tensorflow as tf
import scipy.misc
import numpy as np
from PIL import Image
import click
import tensorflow.contrib.slim.nets as nets
from models import inception_resnet as inception
incept = inception.model


def process_image(image_filename):
    im_tf = tf.gfile.FastGFile(image_filename, 'rb').read()
    im_tf = tf.image.decode_jpeg(im_tf, channels=3)
    im_tf = tf.image.convert_image_dtype(im_tf, dtype=tf.float32)
    im_tf = tf.image.central_crop(im_tf, central_fraction=0.875)
    im_tf = tf.expand_dims(im_tf, 0)
    im_tf = tf.image.resize_bilinear(im_tf, [299, 299], align_corners=False)
    im_tf = tf.squeeze(im_tf, [0])
    im_tf = tf.subtract(im_tf, 0.5)
    im_tf = tf.multiply(im_tf, 2.0)
    im_tf = tf.expand_dims(im_tf, 0)
    return im_tf


@click.command()
@click.option('--weights_from', default='', show_default=True,
              help="Path to the trained weights file")
@click.option('--image_filename', default='', show_default=True,
              help="Image to run predictions.")
def main(image_filename, weights_from):
    try:
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, 299, 299, 3))
        endpoints = incept(inputs, False, None, num_classes=1001)
        pred = endpoints['Predictions']
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, weights_from)

        im_tf = process_image(image_filename)
    except Exception:
        print('Errors occured')
    im_tf_ = im_tf.eval(session=sess)
    pred_ = sess.run(pred, {inputs: im_tf_})
    pred = np.squeeze(pred_)
    try:
        node_to_string = open(
            'examples/classification/imagenet1000_clsid_to_human.txt', 'r').readlines()
    except Exception:
        print('Erros reading synset')
    top_k = pred.argsort()[-5:][::-1]
    for node_id in top_k:
        # Google trained model has class_id shift
        human_string = node_to_string[node_id - 1].strip().split()[1]
        score = pred[node_id]
        print('%s (score = %.5f)' % (human_string, score))


if __name__ == '__main__':
    main()
