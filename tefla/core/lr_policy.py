from __future__ import division, print_function, absolute_import

import logging
import math
import bisect

logger = logging.getLogger('tefla')


class InitialLrMixin(object):
    def __init__(self, initial_lr):
        self._base_lr = initial_lr

    @property
    def base_lr(self):
        return self._base_lr

    @base_lr.setter
    def base_lr(self, base_lr):
        self._base_lr = base_lr

    @property
    def start_epoch(self):
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, start_epoch):
        self._start_epoch = start_epoch


class NoBatchUpdateMixin(object):
    def batch_update(self, learning_rate, iter_idx):
        return learning_rate


class NoEpochUpdateMixin(object):
    def epoch_update(self, learning_rate, training_history):
        return learning_rate

    @property
    def n_iter_per_epoch(self):
        return self._n_iter_per_epoch

    @n_iter_per_epoch.setter
    def n_iter_per_epoch(self, n_iter_per_epoch):
        self._n_iter_per_epoch = n_iter_per_epoch


class NoDecayPolicy(InitialLrMixin, NoBatchUpdateMixin, NoEpochUpdateMixin):
    def __str__(self):
        return 'NoDecayPolicy(rate=%s)' % str(self.initial_lr)

    def __repr__(self):
        return str(self)


class StepDecayPolicy(InitialLrMixin, NoBatchUpdateMixin):
    def __init__(self, schedule, start_epoch=1):
        self.schedule = schedule
        self._start_epoch = 1
        super(StepDecayPolicy, self).__init__(schedule[0])

    def epoch_update(self, learning_rate, training_history):
        epoch_info = training_history[-1]
        epoch = epoch_info['epoch']
        new_learning_rate = learning_rate
        if epoch in self.schedule.keys():
            new_learning_rate = self.schedule[epoch]
        return new_learning_rate

    @property
    def initial_lr(self):
        # if self.start_epoch == 1:
        return self._base_lr
        # else:
        #    step_epochs = self.schedule.keys()
        #    step_epochs = bisect.insort(step_epochs, self._start_epoch)
        #    return self.schedule[step_epochs[step_epochs.index(self._start_epoch)]]

    def __str__(self):
        return 'StepDecayPolicy(schedule=%s)' % str(self.schedule)

    def __repr__(self):
        return str(self)


class PolyDecayPolicy(InitialLrMixin, NoEpochUpdateMixin):
    def __init__(self, base_lr, power=10.0, max_epoch=500, n_iter_per_epoch=1094):
        self.power = power
        self._start_epoch = 1
        self.max_epoch = max_epoch
        self._n_iter_per_epoch = n_iter_per_epoch
        super(PolyDecayPolicy, self).__init__(base_lr)

    def batch_update(self, learning_rate, iter_idx):
        updated_lr = self._base_lr * math.pow(1 - iter_idx / float(self.max_epoch * self._n_iter_per_epoch), self.power)
        return updated_lr

    @property
    def initial_lr(self):
        return self.batch_update(self._base_lr, self.start_epoch * self.n_iter_per_epoch)

    def __str__(self):
        return 'PolyDecayPolicy(initial rate=%f, power=%f, max_epoch=%d)' % (self.initial_lr, self.power,
                                                                             self.max_epoch)

    def __repr__(self):
        return str(self)


class InvDecayPolicy(InitialLrMixin, NoEpochUpdateMixin):
    def __init__(self, base_lr, gamma=9.0, power=10.0, max_epoch=500, n_iter_per_epoch=1094):
        self.gamma = gamma
        self.power = power
        self.start_epoch = 1
        self.max_epoch = max_epoch
        self._n_iter_per_epoch = n_iter_per_epoch
        super(PolyDecayPolicy, self).__init__(base_lr)

    def batch_update(self, learning_rate, iter_idx):
        updated_lr = self.base_lr * math.pow(1 + self.gamma * iter_idx, - self.power)
        return updated_lr

    @property
    def initial_lr(self):
        return self.batch_update(self.base_lr, self.start_epoch * self._n_iter_per_epoch)

    def __str__(self):
        return 'InvDecayPolicy(initial rate=%f, power=%f, max_epoch=%d)' % (self.initial_lr, self.power, self.max_epoch)

    def __repr__(self):
        return str(self)
