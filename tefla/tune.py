import numpy as np
import argparse, json
import tensorflow as tf
from tefla.core.dir_dataset import DataSet
from tefla.core.iter_ops import create_training_iters
from tefla.core.learning import SupervisedLearner
from tefla.da.standardizer import NoOpStandardizer
from tefla.utils import util
from tefla.core.hyperband import Hyperband
from tefla.utils.hyperband_utils import mnist_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model.py', 
    help='Relative path to model.')
parser.add_argument('--demo', default=False, 
    help='Whether to run a demo uning MNIST.')
parser.add_argument('--training_cnf', default='training_cnf.py', 
    help='Relative path to training config file.')
parser.add_argument('--tuning_cnf', default='tuning_cnf.py', 
    help='Relative path to tuning config file.')
parser.add_argument('--max_iter', default=82, type=int, 
    help='maximum iterations per configuration')
parser.add_argument('--eta', default=3, type=int, 
    help='defines configuration downsampling rate (default = 3)')
parser.add_argument('--data_dir', default=None,
    help='Path to training directory.')
parser.add_argument('--parallel', default=True, 
    help='parallel or queued.')
parser.add_argument('--start_epoch', default=1, type=int, 
    help='Epoch number from which to resume training.')
parser.add_argument('--num_classes', default=10, type=int, 
    help='Number of classes to use for training.')
parser.add_argument('--gpu_memory_fraction', default=0.92, 
    help='Epoch number from which to resume training.')
parser.add_argument('--weights_from', default=None, 
    help='Path to initial weights file.')
parser.add_argument('--weights_dir', default='./saved_model', 
    help='Path to save weights file.')
parser.add_argument('--resume_lr', default=0.01,
    help='Path to initial weights file.')
parser.add_argument('--loss_type', default='cross_entropy', 
    help='Loss fuction type.')
parser.add_argument('--weighted', default=False, 
    help='Whether to use weighted loss.')
parser.add_argument('--log_file_name', default='train_seg.log', 
    help='Log file name.')
parser.add_argument('--is_summary', default=False, 
    help='Path to initial weights file.')
parser.add_argument('--verbose', default=1, type=int, 
    help='verbose level')

def try_config(args, cnf):
  """For trying configurations on the custom model given model and cnf.
  """
  model_def = util.load_module(args.model)
  model = model_def.model
  
  if args.demo:
    data_set = mnist_dataset()
    cnf['batch_size_train'] = 32
    cnf['batch_size_test'] = 32
    
    learner = SupervisedLearner(
        model,
        cnf,
        classification=cnf['classification'],
        num_classes=10,
        verbosity=args.verbose)
    
    _early_stop, _loss = learner.fit(data_set, summary_every=399)
    return {
    'early_stop': _early_stop,
    'loss': _loss
    }


  data_set = DataSet(
    args.data_dir,
    model_def.image_size[0],
    mode=cnf.get('mode'),
    multilabel=cnf.get('multilabel', False))

  standardizer = cnf.get('standardizer', NoOpStandardizer())
  cutout = cnf.get('cutout', None)

  training_iter, validation_iter = create_training_iters(
      cnf,
      data_set,
      standardizer,
      model_def.crop_size,
      args.start_epoch,
      parallel=args.parallel,
      cutout=cutout,
      data_balancing=cnf.get('data_balancing', False))
  learner = SupervisedLearner(
      model,
      cnf,
      training_iterator=training_iter,
      validation_iterator=validation_iter,
      resume_lr=args.resume_lr,
      classification=cnf['classification'],
      gpu_memory_fraction=args.gpu_memory_fraction,
      num_classes=args.num_classes,
      is_summary=args.is_summary,
      loss_type=args.loss_type,
      weighted=args.weighted,
      log_file_name=args.log_file_name, 
      verbosity=args.verbose)
  
  _early_stop, _loss = learner.fit(data_set, weights_from=args.weights_from, start_epoch=args.start_epoch, 
                                   weights_dir=args.weights_dir, summary_every=399)
  return {
  'early_stop': _early_stop,
  'loss': _loss
  }
  
def main():
  hb = Hyperband(try_config, args)
  results = hb.run()
  print(json.dumps(results, sort_keys=True, indent=4))
  np.save('./results.npy', results)

if __name__ == "__main__":
  args = parser.parse_args()
  main()