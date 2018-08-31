import numpy as np
import tensorflow as tf

from tefla.core.special_layers import group_norm


class TestGroupNorm(tf.test.TestCase):

 def testGroupNorm(self):
    x = np.random.rand(5, 7, 3, 16)
    with self.test_session() as session:
      y = group_norm(tf.constant(x, dtype=tf.float32))
      session.run(tf.global_variables_initializer())
      res = session.run(y)
      self.assertEqual(res.shape, (5, 7, 3, 16))


if __name__=='__main__':
  tf.test.main()
