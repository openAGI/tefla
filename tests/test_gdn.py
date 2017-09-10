from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from tefla.core.gdn import gdn


class GDNTest(tf.test.TestCase):

    def setUp(self):
        super(GDNTest, self).setUp()

    def _runGDN(self, x, shape, inverse, reuse=None, name='gdn'):
        inputs = tf.placeholder(tf.float32, shape)
        outputs = gdn(inputs, reuse, inverse=inverse, name=name)
        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            y, = sess.run([outputs], {inputs: x})
        return y

    def testUnknownDim(self):
        x = np.random.uniform(size=(1, 2, 3, 4))
        with self.assertRaises(ValueError):
            self._runGDN(x, 4 * [None], False, name='unknowndims')

    def testChannelsLast(self):
        for ndim in [3, 4, 5]:
            x = np.random.uniform(size=(1, 2, 3, 4)[:ndim])
            y = self._runGDN(x, x.shape, False, name='channellast_' + str(ndim))
            self.assertEqual(x.shape, y.shape)
            self.assertAllClose(
                y, x / np.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

    def testWrongDims(self):
        x = np.random.uniform(size=(1, 2, 3, 4, 3, 2)[:6])
        with self.assertRaises(ValueError):
            self._runGDN(x, x.shape, False, name='wrongdims')

    def testIGDN(self):
        x = np.random.uniform(size=(1, 2, 3, 4))
        y = self._runGDN(x, x.shape, True, name='testigdn')
        self.assertEqual(x.shape, y.shape)
        self.assertAllClose(
            y, x * np.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()
