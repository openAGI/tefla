from __future__ import division, print_function, absolute_import

import abc
import six
import math
import bisect
import numpy as np
from . import logger as log


@six.add_metaclass(abc.ABCMeta)
class AbstractInitialLr(object):
    """Initial Lr class"""

    def __init__(self, initial_lr):
        self._base_lr = initial_lr
        self._start_epoch = 1
        super(AbstractInitialLr, self).__init__()

    @property
    def base_lr(self):
        """Returns Base/start learning rate"""
        return self._base_lr

    @base_lr.setter
    def base_lr(self, base_lr):
        """Set Base/start learning rate"""
        self._base_lr = base_lr

    @property
    def initial_lr(self):
        """
        a property
        Returns the updated learning rate
        """
        return self.batch_update(self._base_lr, self.start_epoch * self._n_iters_per_epoch)

    @property
    def start_epoch(self):
        """Returns start_epoch"""
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, start_epoch):
        """Set start epoch"""
        self._start_epoch = start_epoch

    @property
    def n_iters_per_epoch(self):
        return self._n_iters_per_epoch

    @n_iters_per_epoch.setter
    def n_iters_per_epoch(self, n_iters_per_epoch):
        self._n_iters_per_epoch = n_iters_per_epoch

    @abc.abstractmethod
    def batch_update(self, learning_rate, iter_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def epoch_update(self, learning_rate, training_history):
        raise NotImplementedError


class NoDecayPolicy(AbstractInitialLr):
    """Returns learning rate without any decay. Learning rate stays constant over the training
    """

    def __str__(self):
        return 'NoDecayPolicy(rate=%s)' % str(self.initial_lr)

    def __repr__(self):
        return str(self)

    def batch_update(self, learning_rate, iter_idx):
        return learning_rate

    def epoch_update(self, learning_rate, training_history):
        return learning_rate


class StepDecayPolicy(AbstractInitialLr):
    """Training learning rate schedule based on  inputs dict with epoch number as keys

    Args:
        schedule: a dict, epoch number as keys and learning rate as values
        start_epoch: training start epoch number
    """

    def __init__(self, schedule, start_epoch=1):
        self.schedule = schedule
        super(StepDecayPolicy, self).__init__(schedule[0])

    def batch_update(self, learning_rate, iter_idx):
        return learning_rate

    def epoch_update(self, learning_rate, training_history):
        """
        Update the learning rate as per schedule based on training epoch number

        Args:
            learning_rate: previous epoch learning rate
            training_histoty: a dict with epoch, training loss, validation loss as keys

        Returns:
            updated learning rate
        """
        epoch_info = training_history[-1]
        epoch = epoch_info['epoch']
        new_learning_rate = learning_rate
        if epoch in self.schedule.keys():
            new_learning_rate = self.schedule[epoch]
        return new_learning_rate

    def __str__(self):
        return 'StepDecayPolicy(schedule=%s)' % str(self.schedule)

    def __repr__(self):
        return str(self)


class AdaptiveDecayPolicy(AbstractInitialLr):
    """Adaptive training learning rate policy based on validation losses

    If validation loss doesnt decreases over last 5 epoch then decay the learning rate by a
    multiplication factor 0.9; e.g. current_lr*0.9
    """

    def batch_update(self, learning_rate, iter_idx):
        return learning_rate

    def epoch_update(self, learning_rate, training_history, verbose=-1):
        """
        Update the learning rate as per schedule based on training epoch number

        Args:
            learning_rate: previous epoch learning rate
            training_histoty: a dict with epoch, training loss, validation loss as keys
        Returns:
            updated learning rate
        """
        epochs_info = training_history[-6:]
        updated_lr = learning_rate
        if len(epochs_info) > 1:
            last_val_loss = epochs_info[0]['validation_loss']
            if last_val_loss > min(map(lambda vl: vl['validation_loss'], epochs_info[1:])):
                updated_lr = learning_rate * 0.9
                if verbose > -1:
                    log.info("Learning rate changed to: %f " % updated_lr)
        return updated_lr

    def __str__(self):
        return 'AdaptiveDecayPolicy(schedule=%s)' % str(self.initial_lr)

    def __repr__(self):
        return str(self)


class AdaptiveUpDownDecayPolicy(AbstractInitialLr):
    """Adaptive training learning rate policy based on validation losses

    If validation loss doesnt decreases over last 10 epoch then decay the learning rate by a
    multiplication factor 0.9; e.g. current_lr*0.9
    Here we will count the number of upward and downward movements of validation loss,
    if validation losses increases for 6 times (for 10 epoch) then decay the current learning rate
    """

    def batch_update(self, learning_rate, iter_idx):
        return learning_rate

    def epoch_update(self, learning_rate, training_history, sess, verbose=-1):
        """
        Update the learning rate as per schedule based on training epoch number

        Args:
            learning_rate: previous epoch learning rate
            training_histoty: a dict with epoch, training loss, validation loss as keys
        Returns:
            updated learning rate
        """
        # training_history = dict(training_loss:t_loss_val,
        # validation_loss:v_loss_val, epoch:epoch_val)
        epochs_info = training_history[-10:]
        updated_lr = learning_rate
        if len(epochs_info) > 1:
            val_losses = map(lambda vl: vl['validation_loss'], epochs_info)
            pair_losses = zip(val_losses, val_losses[1:])
            up_downs = map(lambda pair: pair[0] < pair[1], pair_losses)
            ups = len(filter(lambda up_down: up_down < 0, up_downs))
            if ups > 6:
                updated_lr = learning_rate * 0.9
                if verbose > -1:
                    log.info("Learning rate changed to: %f " % updated_lr)
        return updated_lr

    def __str__(self):
        return 'AdaptiveUpDownDecayPolicy(schedule=%s)' % str(self.initial_lr)

    def __repr__(self):
        return str(self)


class PolyDecayPolicy(AbstractInitialLr):
    """Polynomial learning rate decay policy

    the effective learning rate follows a polynomial decay, to be
    zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)

    Args:
        base_lr: a float, starting learning rate
        power: a float, decay factor
        max_epoch: a int, max training epoch
        n_iters_per_epoch: number of interations per epoch, e.g. total_training_samples/batch_size
    """

    def __init__(self, base_lr, power=10.0, max_epoch=500, n_iters_per_epoch=1094):
        self.power = power
        self.max_epoch = max_epoch
        self._n_iters_per_epoch = n_iters_per_epoch
        super(PolyDecayPolicy, self).__init__(base_lr)

    def batch_update(self, learning_rate, iter_idx):
        """Update learning rate after every training batch

        it follows a polynomial decay policy

        Args:
            learning_rate: current batch learning rate
            iter_idx: iteration number,
                e.g. number_of_iterations_per_batch*epoch+current_batch_iteration_number

        Returns:
            updated_lr
        """
        updated_lr = self._base_lr * \
            math.pow(1 - iter_idx / float(self.max_epoch *
                                          self._n_iters_per_epoch), self.power)
        return updated_lr

    def epoch_update(self, learning_rate, training_history):
        return learning_rate

    def __str__(self):
        return 'PolyDecayPolicy(initial rate=%f, power=%f, max_epoch=%d)' % (self.initial_lr, self.power,
                                                                             self.max_epoch)

    def __repr__(self):
        return str(self)


class InvDecayPolicy(AbstractInitialLr):
    """Inverse learning rate decay policy

    return base_lr * (1 + gamma * iter) ^ (- power)

    args:
        base_lr: a float, starting learning rate
        gamma: a float, decay factor
        max_epoch: a int, max training epoch
        n_iters_per_epoch: number of interations per epoch, e.g. total_training_samples/batch_size
    """

    def __init__(self, base_lr, gamma=0.96, power=10.0, max_epoch=500, n_iters_per_epoch=1094):
        self.gamma = gamma
        self.power = power
        self.max_epoch = max_epoch
        self._n_iters_per_epoch = n_iters_per_epoch
        super(PolyDecayPolicy, self).__init__(base_lr)

    def batch_update(self, learning_rate, iter_idx):
        """
        Update learning rate after every training batch
        It follows a inverse decay policy

        Args:
            learning_rate: current batch learning rate
            iter_idx: iteration number,
                e.g. number_of_iterations_per_batch*epoch+current_batch_iteration_number

        Returns:
            updated_lr
        """
        updated_lr = self._base_lr * \
            math.pow(1 + self.gamma * iter_idx, - self.power)
        return updated_lr

    def epoch_update(self, learning_rate, training_history):
        return learning_rate

    def __str__(self):
        return 'InvDecayPolicy(initial rate=%f, power=%f, max_epoch=%d)' % (self.initial_lr, self.power, self.max_epoch)

    def __repr__(self):
        return str(self)


class CosineDecayPolicy(AbstractInitialLr):
    """Cosine learning rate decay policy

    the effective learning rate follows a polynomial decay, to be
    zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)

    Args:
        base_lr: a float, starting learning rate
        cycle_steps: a float, decay steps one cycle, learning rate decayed by inv(cycle_steps)
    """

    def __init__(self, base_lr, cycle_steps=100000, **kwargs):
        self._cycle_steps = cycle_steps
        super(CosineDecayPolicy, self).__init__(base_lr)

    def batch_update(self, learning_rate, iter_idx):
        """Update learning rate after every training batch

        it follows a polynomial decay policy

        Args:
            learning_rate: current batch learning rate
            iter_idx: iteration number,
                e.g. number_of_iterations_per_batch*epoch+current_batch_iteration_number

        Returns:
            updated_lr
        """
        updated_lr = self._base_lr * \
            05 * (1 + np.cos(np.pi * (iter_idx %
                                      self._cycle_steps) / self._cycle_steps))
        return updated_lr

    def epoch_update(self, learning_rate, training_history):
        return learning_rate

    def __str__(self):
        return 'CosineDecayPolicy(initial rate=%f, cycle_steps=%f)' % (self.initial_lr, self._cycle_steps)

    def __repr__(self):
        return str(self)


class CyclicDecayPolicy(AbstractInitialLr):
    """Cyclic learning rate decay policy

    the effective learning rate follows a polynomial decay, to be
    zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)

    Args:
        base_lr: a float, starting learning rate
        cycle_steps: a float, decay steps one cycle, learning rate decayed by inv(cycle_steps)
    """

    def __init__(self, base_lr, cycle_steps=100000, **kwargs):
        self._cycle_steps = cycle_steps
        super(CyclicDecayPolicy, self).__init__(base_lr)

    def batch_update(self, learning_rate, iter_idx):
        """Update learning rate after every training batch

        it follows a polynomial decay policy

        Args:
            learning_rate: current batch learning rate
            iter_idx: iteration number,
                e.g. number_of_iterations_per_batch*epoch+current_batch_iteration_number

        Returns:
            updated_lr
        """
        cycle_position = iter_idx % (2 * self._cycle_steps)
        cycle_position = (cycle_position - self._cycle_steps) / \
            float(self._cycle_steps)
        cycle_position = 1.0 - np.abs(cycle_position)
        updated_lr = self._base_lr * (cycle_position + 0.1) * 3.0
        return updated_lr

    def epoch_update(self, learning_rate, training_history):
        return learning_rate

    def __str__(self):
        return 'CyclicDecayPolicy(initial rate=%f, cycle_steps=%f)' % (self.initial_lr, self._cycle_steps)

    def __repr__(self):
        return str(self)
