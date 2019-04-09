'''hyperband
'''
from random import random
from math import log, ceil
from time import time, ctime
import json
import numpy as np
from hyperopt.pyll.stochastic import sample
from tefla.utils import util
from tefla.utils.hyperband_utils import get_config, handle_integers


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
class Hyperband(object):
  """Hyperband main logic
  Args:
      args: command line arguments regarding training
      training_cnf: dict, training configs
      tuning_cnf: dict, tuning configs
      max_iter: maximum iterations per configuration
      eta: defines configuration downsampling rate (default = 3)
      s_max: number of unique executions of Successive Halving
      B: total number of iterations (without reuse) per execution of Succesive Halving (n,r)
      try_params_function: function for trying out configurations.
  """
  def __init__(self, try_params_function, args):
    self.args = args
    self.training_cnf = util.load_module(self.args['training_cnf']).cnf
    self.tuning_cnf = util.load_module(self.args['tuning_cnf']).space
    self.max_iter = self.tuning_cnf.get('hyperband', 81).get('max_iter')
    self.eta = self.tuning_cnf.get('hyperband', 3).get('eta')
    self.logeta = lambda x: log(x) / log(self.eta)
    self.s_max = int(self.logeta(self.max_iter))
    self.B = (self.s_max + 1) * self.max_iter
    self.results = []
    self.counter = 0
    self.best_loss = np.inf
    self.best_counter = -1
    self.try_params = try_params_function

  def get_params(self):
    """getting parameters from search space
    """
    params = sample(self.tuning_cnf)
    return handle_integers(params)

  def run(self, skip_last=0):
    """runs configs for just an iteration or two at first, to get a taste of how they perform.
    Then it takes the best performers and runs them longer.

    Returns:
      dict, all the tested hyperparameters and their performances
    """
    for s in reversed(range(self.s_max + 1)):
      # Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
      n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
      # initial number of iterations per config
      r = self.max_iter * self.eta ** (-s)
      # n random configurations
      T = [self.get_params() for i in range(n)]
      for i in range((s + 1) - int(skip_last)):
        # Run each of the n configs for <iterations>
        # and keep best (n_configs / eta) configurations
        n_configs = n * self.eta ** (-i)
        n_iterations = r * self.eta ** (i)
        print("\n*** {} configurations x {:.1f} iterations each".format(
            n_configs, n_iterations))
        val_losses = []
        early_stops = []
        for t in T:
          # Run each of the n_i configs for r_i iterations and keep best n_i/eta
          self.training_cnf.update(t)
          current_config = get_config(t)
          print('current configuration: \n', json.dumps(current_config, indent=4, sort_keys=True))
          self.training_cnf['num_epochs'] = int(n_iterations)
          self.training_cnf['optname'] = current_config['optname']
          self.counter += 1
          print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
              self.counter, ctime(), self.best_loss, self.best_counter))
          start_time = time()
          # calling try_params function to run the current config
          result = self.try_params(self.args, self.training_cnf)
          assert 'loss' in result
          seconds = int(round(time() - start_time))
          print("\n{} seconds.".format(
              seconds))
          loss = result['loss']
          val_losses.append(loss)
          early_stop = result.get('early_stop', False)
          early_stops.append(early_stop)
          # keeping track of the best result so far (for display only)
          # could do it be checking results each time, but hey
          if loss < self.best_loss:
            self.best_loss = loss
            self.best_counter = self.counter
          # append all info to result dict.
          result['counter'] = self.counter
          result['seconds'] = seconds
          result['params'] = current_config
          result['iterations'] = n_iterations
          self.results.append(result)
        # select a number of best configurations for the next loop
        # filter out early stops, if any
        indices = np.argsort(val_losses)
        T = [T[i] for i in indices if not early_stops[i]]
        T = T[0:int(n_configs / self.eta)]
    # End Finite Horizon Successive Halving with (n,r)
    return self.results
