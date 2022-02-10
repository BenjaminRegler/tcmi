# -*- coding: utf-8 -*-
"""
@package    tcmi.estimators

@copyright  Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
"""

import os
import re
import tempfile
import subprocess

import numpy as np
import sklearn as sk

from . import entropy


class DependenceEstimator(sk.base.BaseEstimator, sk.base.RegressorMixin):
    """A general-purpose dependence estimator for measuring mutual information.
    """
    
    def __init__(self, method='tcmi', n_iter=50000, cache=False, n_jobs=None,
                 verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
        """Initialize dependence estimator.
        """
        self.method = method
        self.n_iter = n_iter
        self.cache = cache
        
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fit_params = fit_params
        self.pre_dispatch = pre_dispatch

    def fit(self, x, y):
        """Fit dependence estimator.
        """
        x, y = sk.utils.check_X_y(x, y, multi_output=False, y_numeric=True)
        size = y.shape[-1]

        self._variables = (x, y)

    def predict(self, x):
        """Predict dependence using mutual information.
        """
        if not self._variables:
            raise RuntimeError('Estimator not fitted yet.')
        return self.score(*self._variables)

    def score(self, x, y=None):
        """Score mutual dependence.
        """
        method = self.method.lower()

        score = 0
        if method == 'tcmi':
            if not isinstance(x, (tuple, list, np.ndarray)):
                x = x.to_numpy()
            if isinstance(x, np.ndarray):
                x = x.T
                
            x = tuple(np.asarray(variable) for variable in x)
            kwargs = dict(n_jobs=self.n_jobs, verbose=self.verbose, adjust=0.1,
                          pre_dispatch=self.pre_dispatch, cache=self.cache)
            score = entropy.cumulative_mutual_information(y, x, **kwargs)

        return np.clip(score, 0, 1)
