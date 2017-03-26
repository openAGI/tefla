import tensorflow as tf
import scipy.misc
import numpy as np
import click

from models.vgg import vgg_19 as model


VGG_MEAN = [103.939, 116.779, 123.68]


def process_image(image_filename):
    im_tf = tf.gfile.FastGFile(image_filename, 'rb').read()
    im_tf = tf.image.decode_jpeg(im_tf, channels=3)
    im_tf = tf.image.convert_image_dtype(im_tf, dtype=tf.float32)
    im_tf = tf.image.central_crop(im_tf, central_fraction=0.875)
    im_tf = tf.expand_dims(im_tf, 0)
    im_tf = tf.image.resize_bilinear(im_tf, [224, 224], align_corners=False)
    im_tf = tf.squeeze(im_tf, [0])
    im_tf = tf.subtract(im_tf, 0.5)
    im_tf = tf.multiply(im_tf, 2.0)
    im_tf = im_tf * 255.0
    r_, g_, b_ = tf.split(im_tf, 3, axis=2)
    r_ = r_ - VGG_MEAN[2]
    g_ = b_ - VGG_MEAN[1]
    b_ = b_ - VGG_MEAN[0]
    im_tf = tf.concat([r_, g_, b_], axis=2)
    im_tf = tf.expand_dims(im_tf, 0)
    return im_tf


@click.command()
@click.option('--weights_from', default='', show_default=True,
              help="Path to the trained weights file")
@click.option('--image_filename', default='', show_default=True,
              help="Image to run predictions.")
def main(image_filename, weights_from):
    vgg_19 = model(False, None)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, weights_from)
    try:
        im_tf = process_image(image_filename)
    except Exception:
        print('Error processing image file')
    im_tf_ = im_tf.eval(session=sess)
    inputs = vgg_19['inputs']

    prediction = vgg_19['predictions']

    pred = sess.run(prediction, {inputs: im_tf_})
    pred = np.squeeze(pred)
    try:
        node_to_string = open(
            'examples/classification/imagenet1000_clsid_to_human.txt', 'r').readlines()
    except Exception:
        print('Erros reading synset')
    top_k = pred.argsort()[-5:][::-1]
    for node_id in top_k:
        human_string = node_to_string[node_id].strip().split()[1]
        score = pred[node_id]
        print('%s (score = %.5f)' % (human_string, score))


if __name__ == '__main__':
    main()
