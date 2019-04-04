'''hyperparameter tuning with hyperband'''
import pickle
import click
from tefla.core.dir_dataset import DataSet
from tefla.core.iter_ops import create_training_iters
from tefla.core.learning import SupervisedLearner
from tefla.da.standardizer import NoOpStandardizer
from tefla.utils import util
from tefla.core.hyperband import Hyperband

# pylint: disable=no-value-for-parameter
@click.command()
@click.option('--model', default=None, show_default=True, help='Relative path to model.')
@click.option(
    '--training_cnf',
    default=None, show_default=True, help='Relative path to training config file.')
@click.option(
    '--tuning_cnf', default=None, show_default=True, help='Relative path to training config file.')
@click.option('--data_dir', default=None, show_default=True, help='Path to training directory.')
@click.option(
    '--results_dir',
    default='./results.txt',
    show_default=True,
    help='Path to hyperband results directory.')
@click.option('--parallel', default=True, show_default=True, help='parallel or queued.')
@click.option(
    '--start_epoch',
    default=1,
    show_default=True,
    help='Epoch number from which to resume training.')
@click.option(
    '--num_classes', default=5, show_default=True, help='Number of classes to use for training.')
@click.option(
    '--gpu_memory_fraction',
    default=0.92,
    show_default=True,
    help='Epoch number from which to resume training.')
@click.option(
    '--weights_from', default=None, show_default=True, help='Path to initial weights file.')
@click.option('--weights_dir', default=None, show_default=True, help='Path to save weights file.')
@click.option('--resume_lr', default=0.01, show_default=True, help='Path to initial weights file.')
@click.option('--loss_type', default='cross_entropy', show_default=True, help='Loss fuction type.')
@click.option('--weighted', default=False, show_default=True, help='Whether to use weighted loss.')
@click.option('--log_file_name', default='train_seg.log', show_default=True, help='Log file name.')
@click.option(
    '--is_summary', default=False, show_default=True, help='Path to initial weights file.')
@click.option('--verbose', default=1, show_default=True, help='Verbose level.')

# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def main(model, training_cnf, tuning_cnf, data_dir, results_dir, parallel, start_epoch,
         weights_from, weights_dir, resume_lr, gpu_memory_fraction, num_classes,
         is_summary, loss_type, weighted, log_file_name, verbose):
  """main function to call hyperband
  """
  args = {
      'model': model,
      'training_cnf': training_cnf,
      'tuning_cnf': tuning_cnf,
      'data_dir': data_dir,
      'parallel': parallel,
      'start_epoch': start_epoch,
      'weights_from': weights_from,
      'weights_dir': weights_dir,
      'resume_lr': resume_lr,
      'gpu_memory_fraction': gpu_memory_fraction,
      'num_classes': num_classes,
      'is_summary': is_summary,
      'loss_type': loss_type,
      'weighted': weighted,
      'log_file_name': log_file_name,
      'verbose': verbose,
  }
  hyperband = Hyperband(try_config, args)
  results = hyperband.run()
  with open(results_dir, "wb") as fil:
    pickle.dump(results, fil)
  print('Hyperband restults are saved to {}'.format(
      results_dir))


def try_config(args, cnf):
  """For trying out configurations on the custom model.
  """
  model_def = util.load_module(args['model'])
  model = model_def.model

  if args['weights_from']:
    weights_from = str(args['weights_from'])
  else:
    weights_from = args['weights_from']

  data_set = DataSet(
      args['data_dir'],
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
      args['start_epoch'],
      parallel=args['parallel'],
      cutout=cutout,
      data_balancing=cnf.get('data_balancing', False))
  learner = SupervisedLearner(
      model,
      cnf,
      training_iterator=training_iter,
      validation_iterator=validation_iter,
      resume_lr=args['resume_lr'],
      classification=cnf['classification'],
      gpu_memory_fraction=args['gpu_memory_fraction'],
      num_classes=args['num_classes'],
      is_summary=args['is_summary'],
      loss_type=args['loss_type'],
      weighted=args['weighted'],
      log_file_name=args['log_file_name'],
      verbosity=args['verbose'],
      is_early_stop=cnf.get('is_early_stop', True))

  _early_stop, _loss = learner.fit(
      data_set,
      weights_from=weights_from,
      start_epoch=args['start_epoch'],
      weights_dir=args['weights_dir'],
      summary_every=399)
  return {'early_stop': _early_stop, 'loss': _loss}


if __name__ == "__main__":
  main()
