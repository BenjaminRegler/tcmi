# -*- coding: utf-8 -*-
"""
@package    python.subspace_search

@copyright  Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
"""

import time
import heapq
import bisect
import joblib
import pickle
import hashlib
import warnings
import itertools
import collections

import numpy as np
import scipy.stats

from . import utils

from sklearn import metrics
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import cross_validate as _cross_validate


def cross_validate(estimator, x, y, scoring=None, cv='warn', **kwargs):
    """Cross-validate model.
    """
    if cv is None or cv == 1:
        scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)
        indices = np.arange(len(y))

        scores, = _fit_and_score(estimator, x, y, scorers, indices, indices,
                                kwargs.get('verbose', False),
                                None, kwargs.get('fit_params', None),
                                return_train_score=False, return_times=False,
                                return_estimator=False, error_score=False)

        stats = {}
        for name in scorers:
            stats['test_%s' % name] = np.array([scores[name], ])
    else:
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

    scorers = {}
    for scorer in scoring:
        name = _get_name(scorer)
        scorers[name] = metrics.make_scorer(_mutual_information)

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        stats = cross_validate(model, x, y, cv=cv, scoring=scorers,
                               fit_params=fit_params, n_jobs=n_jobs)

    for key in list(stats):
        if not key.startswith('test_'):
            del stats[key]
            continue

        raw_value = stats.pop(key)

        key = key[5:]
        scorer = scorers[key]
        raw_value *= scorer._sign

        stat_value = raw_value
        if stat_value.size == 1:
            stat_value = np.concatenate((stat_value, stat_value))

        stats.update({
            key: raw_value,
            '{:s}_mean'.format(key): stat_value.mean(),
            '{:s}_std'.format(key): stat_value.std(),
        })

    return stats


def check_subspace(x, subspace, threshold=0.95):
    """Check subspace before computation.
    """
    filtered_subspace = [feature.strip('+-|') for feature in subspace]
    alias = filtered_subspace[-1]
    key = subspace[-1]

    if alias in filtered_subspace[:-1]:
        return False

    rejection = 1 - threshold
    data = utils.prepare_data(x[[key]], [], copy=True)
    for column, reference in data.items():
        for feature in subspace[:-1]:
            if np.corrcoef(x[feature], reference)[0, 1]**2 > threshold:
                return False

            value = scipy.stats.ks_2samp(x[feature], reference).pvalue
            if value > rejection:
                False

    return True


def _compute_subspace(task, settings):
    """Simple wrapper to compute a subspace task.
    """
    kwargs = settings.copy()
    i, key, features, params = task
    callback, model, x, y = [kwargs.pop(k) for k in ('callback', 'model', 'x', 'y')]

    stats = callback(model, x[features], y, key=key, **kwargs)
    return i, features, stats


def _mutual_information(y_true, y_pred):
    """Mutual information score.
    """

    return y_pred


def _get_name(scoring):
    """Get name of score.
    """
    return 'mutual_information'



def get_subspaces(data, target, model, cv='warn', alpha=0.95, beta=0, tol=5e-3, 
                  subspace=(), depth=-1, scoring='mutual_information_score', fit_params=None,
                  return_score=False, verbose=0, n_jobs=1, pre_dispatch='2*n_jobs',
                  cache=False):
    """Get subspaces.
    """
    x, y = data.drop(target, axis=1), data[target]
    scorer = metrics.make_scorer(_mutual_information)
    sign = scorer._sign

    if depth < 0:
        depth = x.shape[-1]

    if not isinstance(fit_params, dict):
        fit_params = {}

    keys = [key for key in sorted(x) if key not in subspace]
    queue = collections.deque([{
        'stats': None,
        'subspace': subspace,
        'keys': keys,
        'path': (np.float(0), )
    }])

    nodes = []
    bounds = {}
    safe_features = dict()
    subspaces = dict()
    optimal = (np.finfo(np.float).min, np.float_(0), len(subspace), queue[0])

    counter = 0
    eps = np.finfo(np.float_).eps

    evaluate_model = evaluate

    if subspace:
        features = list(subspace)
        key = _get_key(subspace)
        size = len(subspace)
        
        params = fit_params.copy()
        if 'eval_set' in params:
            params['eval_set'] = [(x_test[features], y_test)
                                  for x_test, y_test in params['eval_set']]

        stats = evaluate_model(model, x[features], y, cv=cv, key=key,
                               fit_params=params, scoring=scoring,
                               n_jobs=n_jobs)
        score_mean, score_std = _get_score(scoring, scorer, stats, alpha=alpha)

        queue[0].update({
            'stats': stats,
            'path': queue[0]['path'] + (sign * score_mean, )
        })
        optimal = (score_mean, score_std, len(subspace), queue[0])
        bounds[size] = score_mean - score_std - eps

    n_cores = 1
    if cv is None or cv == 1:
        n_cores = n_jobs

    processor = joblib.Parallel(n_jobs=n_cores, pre_dispatch=pre_dispatch)
    callback = joblib.delayed(_compute_subspace)

    kwargs = dict(callback=evaluate_model, model=model, x=x, y=y, cv=cv,
                  scoring=scoring, n_jobs=n_jobs)

    while queue:
        node = queue.popleft()
        node_mean, node_std = _get_score(scoring, scorer, node['stats'],
                                         alpha=alpha)
        
        subspace = list(node['subspace'])
        key = _get_key(subspace)
        size = len(subspace)

        delta = (node_mean - node_std) - (optimal[0] - optimal[1])
        if key in subspaces or (delta < 0 and size > optimal[2]):
            continue

        if verbose:
            print('Subspace: {{{:s}}}  ->  [{:.2f} +- {:.2f}]'
                  .format(','.join(subspace), node_mean, node_std))

        subspaces[key] = safe_features.pop(key, 0)
        counter += 1
        
        keys = node['keys']
        scores = np.zeros((len(keys), 2), dtype=np.float_)
        statistics = [None for i in range(scores.shape[0])]

        tasks = []
        for i, key in enumerate(keys):
            features = subspace + [key, ]

            key = _get_key(features)
            if key in subspaces or key in safe_features or not check_subspace(x, features):
                scores[i] = (-np.inf, np.float_(0))
                continue

            safe_features[key] = -1
            params = fit_params.copy()
            if 'eval_set' in params:
                params['eval_set'] = [(x_test[features], y_test)
                                      for x_test, y_test in params['eval_set']]

            task = (i, key, features, params)
            tasks.append(task)
            safe_features[key] = -1

        for i, features, stats in processor(callback(task, kwargs)
                                            for task in tasks):
            if verbose:
                print('{:<85s}'.format(','.join(features)), end=' ')

            scores[i] = _get_score(scoring, scorer, stats, alpha=alpha)
            statistics[i] = stats

            if verbose:
                print('[Score: {:5.2f} +- {:5.2f}]'
                      .format(sign * scores[i][0], scores[i][1]))
    
        indices = sorted(_get_node(score[1] - score[0], keys[i], i)
                         for i, score in enumerate(scores))
        keys, indices = tuple(zip(*[(keys[i], i) for *_, i in indices]))

        next_bound = bounds.get(size, bounds.get(size - 1, -np.inf))
        lower_bound = max(bounds.get(size - 1, -np.inf),
                          node_mean - node_std + tol)
        
        if verbose:
            print('Bound: {:.2f} -> {:.2f}'
                  .format(sign * lower_bound, sign * next_bound))

        upper_bound = np.subtract(*scores[indices[0]])
        if np.isfinite(lower_bound) and np.isfinite(upper_bound):
            bound = lower_bound + beta * (upper_bound - lower_bound)
        else:
            bound = lower_bound
        
        for i, j in enumerate(indices):
            score_mean, score_std = scores[j]
            
            if score_mean < bound:
                break

            delta = (score_mean - score_std) - (optimal[0] - optimal[1])
            difference = (score_mean - score_std) - (node_mean - node_std)
            if difference < 0 or np.abs(difference) < tol:
                continue

            child_node = {
                'stats': statistics[j],
                'keys': keys[:i] + keys[i + 1:],
                'subspace': tuple(subspace + [keys[i], ]),
                'path': node['path'] + (sign * score_mean, )
            }

            key = _get_key(child_node['subspace'])
            if key in subspaces:
                print('Something went wrong.\n', key, keys, scores)
                continue
            
            nodes.append(_get_node(score_std - score_mean, counter, child_node))
            
            length = len(child_node['subspace'])
            if delta > tol or (np.abs(delta) < tol and length < optimal[2]):
                optimal = (score_mean, score_std, length, child_node.copy())
            
            index = len(nodes) - 1
            if child_node['keys'] and length < depth and difference > tol and \
                   not np.isclose(score_mean - score_std, 1):
                queue.append(child_node)
                safe_features[key] = index
            else:
                subspaces[key] = index
            counter += 1
        
        score_mean, score_std = scores[indices[0]]
        bounds[size] = max(next_bound, score_mean - score_std - eps)

        if verbose:
            print('[{:s} - {:d} subspaces remaining]\n'
                  .format(time.strftime("%Y-%m-%d %H:%M:%S"), len(queue)))

    if verbose:
        print('\nBalance search tree')

    paths = {}
    size = len(nodes)
    safe_features = set()

    cursor = 0
    while cursor < len(nodes):
        *_, index, node = nodes[cursor]
        subspace = node['subspace']
        cursor += 1

        key = _get_key(subspace)
        if key in safe_features:
            continue

        weight = None
        depth = len(subspace)
        if depth > 1:
            node_mean, node_std = _get_score(scoring, scorer, node['stats'],
                                             alpha=alpha)
            
            for keys in itertools.combinations(subspace, depth - 1):
                features = sorted(keys)
                key = _get_key(features)

                if key not in subspaces:
                    params = fit_params.copy()                    
                    if 'eval_set' in params:
                        params['eval_set'] = [(x_test[features], y_test)
                                              for x_test, y_test in params['eval_set']]
                    
                    if verbose:
                        print('{:<90s}'.format(','.join(features)), end=' ')

                    stats = evaluate_model(model, x[features], y, cv=cv, key=key,
                                           fit_params=params, scoring=scoring,
                                           n_jobs=n_jobs)

                    score_mean, score_std = _get_score(scoring, scorer, stats,
                                                       alpha=alpha)
                    if verbose:
                        print('[Score: {:5.2f} +- {:5.2f}]'
                              .format(sign * score_mean, score_std))
                    
                    path = paths.get(key, (np.float(0), ))
                    
                    child_node = {
                        'stats': stats,
                        'subspace': keys,
                        'path': path + (sign * score_mean, )
                    }

                    delta = (score_mean - score_std) - (node_mean - node_std)
                    if np.abs(delta) < tol:
                        nodes.append(_get_node(score_std - score_mean, counter,
                                               child_node))
                        paths[key] = child_node['path']
                        subspaces[key] = len(nodes) - 1
                        weight = np.inf
                        counter += 1
                else:
                    subset = nodes[subspaces.get(key)]
                    stats = subset[-1]['stats']

                    score_mean, score_std = _get_score(scoring, scorer, stats,
                                                       alpha=alpha)

                    delta = (score_mean - score_std) - (node_mean - node_std)
                    if np.abs(delta) < tol:
                        weight = np.inf

        if weight is not None:
            nodes[cursor - 1] = (weight, index, node)
        elif depth > 1:
            for keys in itertools.combinations(subspace, depth - 1):
                key = _get_key(keys)
                safe_features.add(key)

    heapq.heapify(nodes)
    safe_features.clear()
    subspaces = []
    subsets = []

    cursor = np.inf
    while nodes:
        *score, node = heapq.heappop(nodes)
        node.pop('path', None)
        node.pop('keys', None)

        subspace = sorted(node['subspace'])
        key = _get_key(subspace)
        depth = len(subspace)

        if key in safe_features or score[0] > 0 or depth > cursor:
            continue

        bucket = None
        for size, subset in subsets:
            if depth <= size:
                break

            if subset.issubset(subspace):
                bucket = subset
                break

        if bucket is None:
            bisect.insort_left(subsets, (depth, set(subspace)))            
            subspaces.append(node)
            safe_features.add(key)
            cursor = depth
            
    del subsets
    del nodes
    
    result = (subspaces, )    
    if return_score:
        del optimal[-1]['keys']
        result += (optimal[-1], )
    return result if len(result) > 1 else result[0]


def _get_score(key, scorer, stats=None, alpha=0.95):
    """Get score.
    """
    if stats is None:
        stats = {}

    key = _get_name(key)
    mean = scorer._sign * stats.get(key + '_mean', -scorer._sign * np.inf)

    values = stats.get(key, None)
    std = np.float_(0)
    
    return mean, std


def _get_node(value, counter, node, start=1, stop=4):
    """Get node.
    """
    return tuple(np.around(value, decimals=num)
                 for num in range(start, stop + 1)) + (counter, node)


def _get_key(subspace):
    """Get key.
    """
    return ','.join(sorted(subspace, key=lambda k: (k.strip('-+|'), len(k), k)))


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
