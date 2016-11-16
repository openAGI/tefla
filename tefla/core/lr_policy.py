from __future__ import division, print_function, absolute_import

import math
import logging

logger = logging.getLogger('tefla')


class NoDecayPolicy(object):
    def __init__(self, lr):
        self.lr = lr

    def initial_lr(self):
        return self.lr

    def update(self, learning_rate, training_history, sess, verbose):
        pass

    def __str__(self):
        return 'NoDecayPolicy(rate=%s)' % str(self.lr)

    def __repr__(self):
        return str(self)


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

    def __str__(self):
        return 'StepDecayPolicy(schedule=%s)' % str(self.schedule)

    def __repr__(self):
        return str(self)


class AdaptiveDecayPolicy(NoDecayPolicy):
    def update(self, learning_rate, training_history, sess, verbose=-1):
        epochs_info = training_history[-6:]
        if len(epochs_info) > 1:
            last_val_loss = epochs_info[0]['validation_loss']
            if last_val_loss > min(map(lambda vl: vl['validation_loss'], epochs_info[1:])):
                sess.run(learning_rate.assign(learning_rate* 0.9))
                if verbose > -1:
                    logger.info("Learning rate changed to: %f " % sess.run(learning_rate))

    def __str__(self):
        return 'AdaptiveDecayPolicy(schedule=%s)' % str(self.schedule)

    def __repr__(self):
        return str(self)


class AdaptiveUpDownDecayPolicy(NoDecayPolicy):
    def update(self, learning_rate, training_history, sess, verbose=-1):
        epochs_info = training_history[-10:]
        if len(epochs_info) > 1:
            val_losses = map(lambda vl: vl['validation_loss'], epochs_info)
            pair_losses = zip(val_losses, val_losses[1:])
            up_downs = map(lambda pair: pair[0] < pair[1], pair_losses)
            ups = len(filter(lambda up_down: up_down < 0, up_downs))
            if ups > 7:
                sess.run(learning_rate.assign(learning_rate* 0.9))
                if verbose > -1:
                    logger.info("Learning rate changed to: %f " % sess.run(learning_rate))

    def __str__(self):
        return 'AdaptiveDecayPolicy(schedule=%s)' % str(self.schedule)

    def __repr__(self):
        return str(self)


class PolyDecayPolicy(NoDecayPolicy):
    def update(self, learning_rate, sess, iter_idx, power=10.0, max_epoch=500, n_iter_per_epoch=1000, verbose=-1):
        updated_lr = learning_rate * math.pow(1 - iter_idx / float(max_epoch * n_iter_per_epoch), power)
        sess.run(learning_rate.assign(updated_lr))
        if verbose > -1:
            logger.info("Learning rate changed to: %f " % sess.run(learning_rate))

    def __str__(self):
        return 'AdaptiveDecayPolicy(schedule=%s)' % str(self.schedule)

    def __repr__(self):
        return str(self)


class InvDecayPolicy(NoDecayPolicy):
    def update(self, learning_rate, sess, iter_idx, gamma=9.0, power=10.0, verbose=-1):
        updated_lr = learning_rate * math.pow(1 + gamma * iter_idx, -power)
        sess.run(learning_rate.assign(updated_lr))
        if verbose > -1:
            logger.info("Learning rate changed to: %f " % sess.run(learning_rate))

    def __str__(self):
        return 'AdaptiveDecayPolicy(schedule=%s)' % str(self.schedule)

    def __repr__(self):
        return str(self)
