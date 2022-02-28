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

        # Save data
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

        # Get method
        method = self.method.lower()
        if method == 'mcde':
            method = 'mwp'

        score = 0
        if method == 'tcmi':
            if not isinstance(x, (tuple, list, np.ndarray)):
                x = x.to_numpy()
            if isinstance(x, np.ndarray):
                x = x.T
                
            # Compute cumulative mutual information
            x = tuple(np.asarray(variable) for variable in x)
            kwargs = dict(n_jobs=self.n_jobs, verbose=self.verbose, adjust=0.1,
                          pre_dispatch=self.pre_dispatch, cache=self.cache)
            score = entropy.cumulative_mutual_information(y, x, **kwargs)
        else:
            # Get executable
            current_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(os.path.dirname(current_path),
                                'assets', 'mcde.jar')

            # Compute contrast measure with sepcified method
            with tempfile.NamedTemporaryFile(suffix='.csv') as file:
                filename = file.name

                # Save data
                data = np.column_stack((y, x))
                np.savetxt(filename, data, delimiter=',')

                # Run command
                command = 'java -jar "{:s}" -t EstimateDependency -p 1 -f {:s} ' \
                          '-a {:s} -m {:d}'.format(path, filename, method, self.n_iter)
        
                output = subprocess.check_output(command, shell=True)
                output = output.decode()

                # Extract dependency estimation (score)
                match = re.search(r'^\d+(?:.\d+)?$', output, re.M)
                score = np.float_(0)
                if match:
                    score = np.float_(match.group(0))

        # Make sure dependence estimator is always in the range [0, 1]
        return np.clip(score, 0, 1)
