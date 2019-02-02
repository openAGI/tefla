import tensorflow as tf

from tefla.core.dir_dataset import DataSet
from tefla.core.iter_ops import create_training_iters
from tefla.da.standardizer import NoOpStandardizer
from tefla.utils import util


class TFDataset():

  def __init__(self,
               model_cnf,
               data_dir,
               image_size,
               crop_size,
               channel_dim=3,
               start_epoch=1,
               parallel=True):
    self.cnf = util.load_module(model_cnf).cnf
    self.image_size = image_size
    self.crop_size = crop_size
    self.channel_dim = channel_dim
    self.data_set = DataSet(data_dir, image_size)
    self.standardizer = self.cnf.get('standardizer', NoOpStandardizer())
    self.training_iter, self.validation_iter = create_training_iters(
        self.cnf,
        self.data_set,
        self.standardizer, [crop_size, crop_size],
        start_epoch,
        parallel=parallel)

    self.training_X = self.data_set.training_X
    self.training_y = self.data_set.training_y
    self.validation_X = self.data_set.validation_X
    self.validation_y = self.data_set.validation_y

  def gen(self, is_training=True):
    if is_training:
      for batch_num, (Xb, yb) in enumerate(self.training_iter(self.training_X, self.training_y)):
        yield Xb, yb
    else:
      for batch_num, (Xb, yb) in enumerate(
          self.validation_iter(self.validation_X, self.validation_y)):
        yield Xb, yb

  def input_fn(self, is_training):
    ds = tf.data.Dataset.from_generator(
        self.gen, (tf.float32, tf.int64),
        output_shapes=(tf.TensorShape([None, self.crop_size, self.crop_size, self.channel_dim]),
                       tf.TensorShape([None])),
        args=([is_training]))
    return ds.make_one_shot_iterator().get_next()
