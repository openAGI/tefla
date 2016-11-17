from __future__ import division, print_function, absolute_import

import logging
import math

logger = logging.getLogger('tefla')


class InitialLrMixin(object):
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr


class NoBatchUpdateMixin(object):
    def batch_update(self, learning_rate, iter_idx, n_iter_per_epoch):
        return learning_rate


class NoEpochUpdateMixin(object):
    def epoch_update(self, learning_rate, training_history):
        return learning_rate


class NoDecayPolicy(InitialLrMixin, NoBatchUpdateMixin, NoEpochUpdateMixin):
    def __str__(self):
        return 'NoDecayPolicy(rate=%s)' % str(self.initial_lr)

    def __repr__(self):
        return str(self)


class StepDecayPolicy(NoBatchUpdateMixin):
    def __init__(self, schedule):
        self.schedule = schedule
        self.initial_lr = self.schedule[0]

    def epoch_update(self, learning_rate, training_history):
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


class PolyDecayPolicy(InitialLrMixin, NoEpochUpdateMixin):
    def __init__(self, initial_lr, power=10.0, max_epoch=500):
        self.power = power
        self.max_epoch = max_epoch
        super(PolyDecayPolicy, self).__init__(initial_lr)

    def batch_update(self, learning_rate, iter_idx, n_iter_per_epoch):
        new_learning_rate = self.initial_lr * math.pow(1 - iter_idx / float(self.max_epoch * n_iter_per_epoch),
                                                       self.power)
        return new_learning_rate

    def __str__(self):
        return 'PolyDecayPolicy(initial rate=%f, power=%f, max_epoch=%d)' % (self.initial_lr, self.power,
                                                                             self.max_epoch)

    def __repr__(self):
        return str(self)
