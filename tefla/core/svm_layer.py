from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sk
import sklearn.model_selection


class exp2var(object):

    def __init__(self, loc=0.0, scale=1.0):
        self.dist = sp.stats.uniform(loc=loc, scale=scale)
        self.loc = loc
        self.scale = scale

    def rvs(self, **kwargs):
        u = self.dist.rvs(**kwargs)
        return 2.0**u


class SVMLayer(object):

    def __init__(self, linear=False, dual=True, max_iter=10000, min_gamma=-24, scale_gamma=8):
        if linear:
            self.parameters = {
                'dual': [dual],
                'C': exp2var(loc=-16.0, scale=32.0),
                'multi_class': ['ovr'],
                'random_state': [0],
                'max_iter': [max_iter],
            }
            self.classifier = sk.svm.LinearSVC()
        else:
            self.parameters = {
                'C': exp2var(loc=-16.0, scale=32.0),
                'gamma': exp2var(loc=min_gamma, scale=scale_gamma),
                'kernel': ['rbf'],
                'decision_function_shape': ['ovr'],
                'random_state': [0],
            }
            self.classifier = sk.svm.SVC()

    def hyperoptimizer(self, scoring='roc_auc', max_iter=10, n_jobs=1, group=True):
        return sk.model_selection.RandomizedSearchCV(self.classifier, self.parameters,
                                                     n_iter=max_iter, scoring=scoring, fit_params=None, n_jobs=n_jobs, iid=True, refit=True,
                                                     cv=sk.model_selection.GroupKFold(
                                                         n_splits=3) if group else None,
                                                     verbose=2, random_state=0)
