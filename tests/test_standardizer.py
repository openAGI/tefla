import tensorflow as tf
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from tefla.da.standardizer import AggregateStandardizer, AggregateStandardizerTF
from tefla.da.standardizer import SamplewiseStandardizer, SamplewiseStandardizerTF


@pytest.fixture(autouse=True)
def _reset_graph():
    tf.reset_default_graph()


def test_np_tf_aggregate():
    standardizer = AggregateStandardizer(
        mean=np.array([108.64628601, 75.86886597, 54.34005737],
                      dtype=np.float32),
        std=np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32),
        u=np.array([[-0.56543481, 0.71983482, 0.40240142],
                    [-0.5989477, -0.02304967, -0.80036049],
                    [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32),
        ev=np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32),
        sigma=0.5
    )
    standardizertf = AggregateStandardizerTF(
        mean=np.array([108.64628601, 75.86886597, 54.34005737],
                      dtype=np.float32),
        std=np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32),
        u=np.array([[-0.56543481, 0.71983482, 0.40240142],
                    [-0.5989477, -0.02304967, -0.80036049],
                    [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32),
        ev=np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32),
        sigma=0.5
    )

    sess = tf.Session()

    # im_tf = tf.read_file('/home/ishant/Downloads/segmentation2.jpg')
    # im_tf = tf.image.decode_jpeg(im_tf)
    # im_np = np.asarray(im_tf.eval(session=sess), dtype=np.float32).transpose(2, 1, 0)
    im_np = np.random.normal(50.0, 2.5,
                             size=(200, 200, 3))
    im_np = np.clip(im_np, 0.0, 255.0)
    im_tf = im_np
    im_np = np.asarray(im_np.transpose(2, 1, 0), dtype=np.float32)
    im_st = standardizer(im_np, False)
    im_st = im_st.transpose(1, 2, 0)
    im_tf = tf.transpose(im_tf, perm=[1, 0, 2])
    im_sttf = standardizertf(tf.to_float(im_tf), False)
    im_ = im_sttf.eval(session=sess)
    # print(im_st[188, 113, :])
    # print(im_[188, 113, :])
    assert_array_almost_equal(im_st, im_)


def test_np_tf_samplewise():
    sttf = SamplewiseStandardizerTF(clip=6)
    st = SamplewiseStandardizer(clip=6)
    sess = tf.Session()
    im_np = np.random.normal(50.0, 2.5,
                             size=(200, 200, 3))
    im_np = np.clip(im_np, 0.0, 255.0)
    im_tf = im_np
    im_np = np.asarray(im_np.transpose(2, 1, 0), dtype=np.float32)
    im_st = st(im_np, False)
    im_st = im_st.transpose(1, 2, 0)
    im_tf = tf.transpose(im_tf, perm=[1, 0, 2])
    im_sttf = sttf(tf.to_float(im_tf), False)
    im_ = im_sttf.eval(session=sess)
    assert_array_almost_equal(im_st, im_, decimal=4)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_np_tf_aggregate()
