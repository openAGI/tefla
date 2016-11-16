from __future__ import division, print_function, absolute_import

import logging

logger = logging.getLogger('tefla')


class NoDecayPolicy(object):
    def __init__(self, lr):
        self.lr = lr

    def initial_lr(self):
        return self.lr

    def update(self, learning_rate, training_history, sess, verbose):
        pass


class StepDecayPolicy(object):
    def __init__(self, schedule):
        self.schedule = schedule

    def initial_lr(self):
        return self.schedule[0]

    def update(self, learning_rate, training_history, sess, verbose):
        epoch_info = training_history[-1]
        epoch = epoch_info['epoch']
        if epoch in self.schedule.keys() and self.schedule[epoch] is not 'stop':
            sess.run(learning_rate.assign(self.schedule[epoch]))
            if verbose > -1:
                logger.info("Learning rate changed to: %f " % sess.run(learning_rate))
