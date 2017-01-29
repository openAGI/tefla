import numpy as np
import tensorflow as tf
from scipy.misc import imsave
import numbers


class SessionWrap(object):

    def __init__(self, session=None):
        self.session = session
        if session is None:
            self.release_session = True
        else:
            self.release_session = False

    def __enter__(self):
        if self.session is None:
            self.session = tf.Session()
        return self.session

    def __exit__(self, *args):
        if self.release_session:
            self.session.close()


def save_images(fname, flat_img, width=28, height=28, sep=3):
    N = flat_img.shape[0]
    pdim = int(np.ceil(np.sqrt(N)))
    image = np.zeros((pdim * (width + sep), pdim * (height + sep)))
    for i in range(N):
        row = int(i / pdim) * (height + sep)
        col = (i % pdim) * (width + sep)
        image[row:row + width, col:col +
              height] = flat_img[i].reshape(width, height)
    imsave(fname, image)
