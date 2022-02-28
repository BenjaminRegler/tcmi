# -*- coding: utf-8 -*-
"""
@package    tcmi.model_selection

@copyright  Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
"""

import warnings
import numpy as np
import scipy.stats as st

from sklearn.utils import check_random_state
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import cross_validate as _cross_validate

from . import metrics


class SortedStratifiedKFold(StratifiedKFold):
    '''Stratified K-Fold cross validator.
    '''
    def __init__(self, n_splits=3, test_size=0.1, side='right', shuffle=False,
                 random_state=None):
        """Constructor.
        """
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

        self.test_size = test_size
        self.side = side

    def split(self, x, y, groups=None):
        """Split data set.
        """
        # Prepare data split
        size = len(y)
        offset = int(size * self.test_size)
        idx = np.argsort(y, kind='mergesort')

        # Hide data from stratified k-fold
        if self.side == 'left':
            train_indices, test_indices = idx[offset:], idx[:offset]
        elif self.side == 'right':
            train_indices, test_indices = idx[:size-offset], idx[-offset:]
            if offset == 0:
                test_indices = np.array([], dtype=np.int_)
        elif self.side == 'uniform':
            indices = [np.arange(size)]
            test_indices = []
            while indices and len(test_indices) < offset:
                interval = indices.pop(0)
                n = len(interval)
                if n == 0:
                    continue
                
                split = int(n // 2)
                test_indices.append(interval[split])
                indices.extend((interval[:split], interval[split+1:]))

            test_indices = np.array(test_indices, dtype=np.int_)
            train_indices = np.setdiff1d(idx, test_indices)
        else:
            train_indices = idx[offset:size-offset]
            test_indices = np.concatenate((idx[:offset], idx[size-offset:]))

        # Split data
        y_train = y[train_indices]

        # Construct iterator
        y_cat = equal_width(y_train, bins=self.n_splits, seed=None)
        for train, test in super().split(y_cat, y_cat, groups):                
            yield train_indices[train], np.concatenate((train_indices[test], test_indices))


class RepeatedSortedStratifiedKFold(_RepeatedSplits):
    """Repeated Stratified K-Fold cross validator.
    """
    def __init__(self, n_splits=5, n_repeats=10, test_size=0.1, side='right',
                 random_state=None):
        super().__init__(SortedStratifiedKFold, n_repeats=n_repeats, random_state=random_state,
                         n_splits=n_splits, test_size=test_size, side=side)


def split_data(x, y, test_size=0.2, random_state=None, shuffle=True):
    """Split data.
    """
    y_cat = equal_percentiles(y, bins=test_size)
    return train_test_split(x, y, test_size=test_size, shuffle=shuffle,
                            random_state=random_state)


def equal_percentiles(y, bins='auto'):
    """Equal-percentiles discretization.
    """
    n = len(y)
    
    # Extract cutpoints
    bins = np.int(n / _estimate_bins(y, bins=bins))
    cutpoints = np.linspace(0, 100, num=bins)

    # Compute percentiles and make sure that they are unique
    percentiles = np.percentile(y, cutpoints)
    percentiles = np.unique(percentiles)

    # Discretize data
    return np.digitize(y, percentiles[:-1])


def equal_width(y, bins='auto', seed=None):
    """Equal-width discretization.
    """
    n = len(y)
    bins = _estimate_bins(y, bins=bins)
    categories = np.empty(n, dtype='u4')
    
    div, mod = divmod(n, bins)
    categories[:n - mod] = np.repeat(range(div), bins)

    size = len(categories[n - mod:])
    if size > 0:
        if seed is None:
            categories[-size:] = div - 1
        else:
            rng = check_random_state(seed)
            categories[-size:] = rng.choice(range(div), size)

    # Run argsort twice to get the rank of each y value
    return categories[np.argsort(np.argsort(y, kind='mergesort'), kind='mergesort')]


def _estimate_bins(x, bins='auto'):
    """Estimate bin width.
    """
    size = len(x)
    
    # Convert relative to absolute number of bins
    if np.isreal(bins) and 0 < bins < 1:
        bins = np.floor(bins * size)
        
    # Find optimal number of bins
    elif bins == 'auto' or not isinstance(bins, (int, np.integer)):
        if size > 1000:
            # Compute interquartile range
            iqr = np.subtract.reduce(np.percentile(x, [75, 25]))

            # Compute Freedman Diaconis Estimator
            fd = 2 * iqr / size ** (1 / 3)
            bins = np.ceil(np.ptp(x) / fd)
        else:
            # Use Sturges estimator for small datasets
            bins = np.ceil(np.log2(size) + 1)
    
    return np.int(bins)


def cross_validate(estimator, x, y, scoring=None, cv='warn', **kwargs):
    """Cross-validate model.
    """
    if cv is None or cv == 1:
        # Get score metrics
        scorers = _check_multimetric_scoring(estimator, scoring=scoring)
        if not isinstance(scorers, dict):
             scorers = scorers[0]
        indices = np.arange(len(y))

        # Fit estimator to data
        scores = _fit_and_score(estimator, x, y, scorers, indices, indices,
                                kwargs.get('verbose', False),
                                None, kwargs.get('fit_params', None),
                                return_train_score=False, return_times=False,
                                return_estimator=False, error_score=False)
        if "test_scores" in scores:
             scores = scores["test_scores"]

        # Collect statistics
        stats = {}
        for name in scorers:
            stats['test_%s' % name] = np.array([scores[name], ])
    else:
        # Cross-validate as usual
        stats = _cross_validate(estimator, x, y, cv=cv, scoring=scoring,
                                **kwargs)

    return stats


def evaluate(model, x, y, cv='warn', scoring='', fit_params=None, n_jobs=None,
             key=''):
    """Evaluate model and compute statistics.
    """
    if not scoring:
        if np.issubdtype(y.dtype, np.integer):
            scoring =_SCORINGS['classification']
        else:
            scoring = _SCORINGS['regression']
    elif not isinstance(scoring, (list, tuple)):
        scoring = (scoring, )

    # Get scorers
    scorers = {}
    for scorer in scoring:
        name = metrics.get_name(scorer)
        scorers[name] = metrics.get_scorer(scorer)

    # Compute model
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        stats = cross_validate(model, x, y, cv=cv, scoring=scorers,
                               fit_params=fit_params, n_jobs=n_jobs)

    # Filter results
    for key in list(stats):
        if not key.startswith('test_'):
            del stats[key]
            continue

        # Get value
        raw_value = stats.pop(key)

        # Correct sign
        key = key[5:]
        scorer = scorers[key]
        raw_value *= scorer._sign

        # Make sure there are enough data points for statistics
        stat_value = raw_value
        if stat_value.size == 1:
            stat_value = np.concatenate((stat_value, stat_value))

        # Compute statistics
        stats.update({
            key: raw_value,
            '{:s}_mean'.format(key): stat_value.mean(),
            '{:s}_std'.format(key): stat_value.std(),
        })

    return stats


def estimate_error(values, confidence=0.95, bootstrapping=1000, scale=None):
    """Estimate errors based on bootstrapping to find confidence interval.
    """
    values = np.asarray(values, dtype=float)
    
    if bootstrapping > 0:
        estimates = []
        for _ in range(bootstrapping):
            idx = np.random.randint(0, len(values), values.shape)
            estimates.append(np.take(values, idx).mean())

        # Determine error bounds based on given confidence interval alpha
        le, ue = (1 - confidence) / 2, (1 + confidence) / 2

        # Get error bounds
        median = np.quantile(estimates, 0.5)
        errors = np.quantile(estimates, [le, ue])

        # Shift errors
        errors -= median
    else:
        # Estimate error via t-Student distribution
        errors = st.t.interval(confidence, len(values) - 1, scale=st.sem(values))
    
    # Shift error bounds to scale
    errors = np.nan_to_num(errors)
    if scale:
        errors += scale

    # Error bounds are always centered around mean/median
    return errors


def get_statistics(model, x, y, cv='warn', fit_params=None, alpha=0.95, verbose=0,
                   n_jobs=None):
    """Generate statistic summary for model 
    """
    # Cross-validate model
    stats = evaluate(model, x, y, cv=cv, fit_params=fit_params, n_jobs=n_jobs)

    if verbose:
        # Filter keys
        keys = [key for key in stats if not key.endswith('_mean') and \
                not key.endswith('_std')]
        width = max(len(key) for key in keys)

        # Get name of target variable
        target = getattr(y, 'name', 'n/a')
        print('\n------------------------------'
              '\nTarget: {target:s}'
              '\nRange = [{vmin:.2f}, {vmax:.2f}]\n'
              .format(vmin=np.min(y, axis=0), vmax=np.max(y, axis=0),
                      target=target))

        for key in sorted(keys, key=lambda x: (len(x), x)):
            value = stats[key + '_mean']
            le, ue = estimate_error(stats[key], confidence=alpha)

            print('{:{width}s} = {:8.2f} +- {:.2f} [{:.2f}, {:.2f}]'
                    .format(key, value, stats[key + '_std'], le, ue,
                            width=width))
    
    return stats


_SCORINGS = {
    'classification': (
        'accuracy',
        'f1_micro',
        'precision_micro',
        'recall_micro'
    ),
    'regression': (
        'r2',
        'neg_root_mean_squared_error',
        'neg_mean_absolute_error',
        'neg_maximum_absolute_error',
        'neg_relative_root_mean_squared_error',
        'neg_relative_mean_absolute_error',
        'neg_relative_maximum_absolute_error'
    )
}
