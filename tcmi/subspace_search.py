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
from . import metrics
from .model_selection import cross_validate, evaluate


def check_subspace(x, subspace, threshold=0.95):
    """Check subspace before computation.
    """
    filtered_subspace = [feature.strip('+-|') for feature in subspace]
    alias = filtered_subspace[-1]
    key = subspace[-1]

    # Check that key is not a variant of a feature in the subspace
    if alias in filtered_subspace[:-1]:
        return False

    # Check correlations between feature and subspace
    rejection = 1 - threshold
    data = utils.prepare_data(x[[key]], [], copy=True)
    for column, reference in data.items():
        for feature in subspace[:-1]:
            # Linear correlations
            if np.corrcoef(x[feature], reference)[0, 1]**2 > threshold:
                return False

            # Nonlinear correlations
            value = scipy.stats.ks_2samp(x[feature], reference).pvalue
            if value > rejection:
                False

    # All tests have been passed and feature is ok
    return True


def _compute_subspace(task, settings):
    """Simple wrapper to compute a subspace task.
    """
    # Unpack task
    kwargs = settings.copy()
    i, key, features, params = task
    callback, model, x, y = [kwargs.pop(k) for k in ('callback', 'model', 'x', 'y')]

    # Evaluate model
    stats = callback(model, x[features], y, key=key, **kwargs)
    return i, features, stats


def get_subspaces(data, target, model, cv='warn', alpha=0.95, beta=0, tol=5e-3, 
                  subspace=(), depth=-1, scoring='mutual_information_score', fit_params=None,
                  return_score=False, verbose=0, n_jobs=1, pre_dispatch='2*n_jobs',
                  cache=False):
    """Get subspaces.
    """
    # Split data
    x, y = data.drop(target, axis=1), data[target]
    scorer = metrics.get_scorer(scoring)
    sign = scorer._sign

    # Setup depth
    if depth < 0:
        depth = x.shape[-1]

    # Make sure fit_params is a dictionary
    if not isinstance(fit_params, dict):
        fit_params = {}

    # Setup priority queue
    keys = [key for key in sorted(x) if key not in subspace]
    queue = collections.deque([{
        'stats': None,
        'subspace': subspace,
        'keys': keys,
        'path': (np.float(0), )
    }])

    # Prepare subspace search
    nodes = []
    bounds = {}
    safe_features = dict()
    subspaces = dict()
    optimal = (np.finfo(np.float).min, np.float_(0), len(subspace), queue[0])

    counter = 0
    eps = np.finfo(np.float_).eps

    # Cache evaluations
    evaluate_model = evaluate

    # Compute score for initial subspace
    if subspace:
        features = list(subspace)
        key = _get_key(subspace)
        size = len(subspace)
        
        params = fit_params.copy()
        if 'eval_set' in params:
            params['eval_set'] = [(x_test[features], y_test)
                                  for x_test, y_test in params['eval_set']]

        # Evaluate model
        stats = evaluate_model(model, x[features], y, cv=cv, key=key,
                               fit_params=params, scoring=scoring,
                               n_jobs=n_jobs)
        score_mean, score_std = _get_score(scoring, scorer, stats, alpha=alpha)

        # Update root node
        queue[0].update({
            'stats': stats,
            'path': queue[0]['path'] + (sign * score_mean, )
        })
        optimal = (score_mean, score_std, len(subspace), queue[0])
        bounds[size] = score_mean - score_std - eps

    # Use all CPUs for subspace search without CV
    n_cores = 1
    if cv is None or cv == 1:
        n_cores = n_jobs

    # Initialize joblib
    processor = joblib.Parallel(n_jobs=n_cores, pre_dispatch=pre_dispatch)
    callback = joblib.delayed(_compute_subspace)

    kwargs = dict(callback=evaluate_model, model=model, x=x, y=y, cv=cv,
                  scoring=scoring, n_jobs=n_jobs)

    # Perform branch-and-bound
    while queue:
        node = queue.popleft()
        node_mean, node_std = _get_score(scoring, scorer, node['stats'],
                                         alpha=alpha)
        
        subspace = list(node['subspace'])
        key = _get_key(subspace)
        size = len(subspace)

        # Skip subspace, if there is no improvement in the score
        delta = (node_mean - node_std) - (optimal[0] - optimal[1])
        if key in subspaces or (delta < 0 and size > optimal[2]):
            continue

        # Print information about subspace
        if verbose:
            print('Subspace: {{{:s}}}  ->  [{:.2f} +- {:.2f}]'
                  .format(','.join(subspace), node_mean, node_std))

        # Add subspace to list
        subspaces[key] = safe_features.pop(key, 0)
        counter += 1
        
        # Compute models
        keys = node['keys']
        scores = np.zeros((len(keys), 2), dtype=np.float_)
        statistics = [None for i in range(scores.shape[0])]

        tasks = []
        for i, key in enumerate(keys):
            features = subspace + [key, ]

            # Use sorted subspace to avoid duplicates in tree exploration
            key = _get_key(features)
            if key in subspaces or key in safe_features or not check_subspace(x, features):
                scores[i] = (-np.inf, np.float_(0))
                continue

            safe_features[key] = -1
            params = fit_params.copy()
            if 'eval_set' in params:
                params['eval_set'] = [(x_test[features], y_test)
                                      for x_test, y_test in params['eval_set']]

            # Delay model evaluation
            task = (i, key, features, params)
            tasks.append(task)
            safe_features[key] = -1

        # Evaluate models
        for i, features, stats in processor(callback(task, kwargs)
                                            for task in tasks):
            if verbose:
                print('{:<85s}'.format(','.join(features)), end=' ')

            scores[i] = _get_score(scoring, scorer, stats, alpha=alpha)
            statistics[i] = stats

            if verbose:
                print('[Score: {:5.2f} +- {:5.2f}]'
                      .format(sign * scores[i][0], scores[i][1]))
    
        # Sort models in descending order of score (stable)
        indices = sorted(_get_node(score[1] - score[0], keys[i], i)
                         for i, score in enumerate(scores))
        keys, indices = tuple(zip(*[(keys[i], i) for *_, i in indices]))

        # Skip subspaces smaller than specified bound
        next_bound = bounds.get(size, bounds.get(size - 1, -np.inf))
        lower_bound = max(bounds.get(size - 1, -np.inf),
                          node_mean - node_std + tol)

        # Determine lower bound
        if not np.isfinite(lower_bound):
            lower_bound = np.min(scores.T[0]) - tol
        
        if verbose:
            print('Bound: {:.2f} -> {:.2f}'
                  .format(sign * lower_bound, sign * next_bound))

        # Perform constrained beam search
        upper_bound = np.subtract(*scores[indices[0]])
        if np.isfinite(lower_bound) and np.isfinite(upper_bound):
            bound = lower_bound + beta * (upper_bound - lower_bound)
        else:
            bound = lower_bound
        
        # Enlarge feature subspace
        for i, j in enumerate(indices):
            score_mean, score_std = scores[j]
            
            # Stop criteria of branch-and-bound algorithm
            if score_mean < bound:
                break

            # Skip subspace, if there is no improvement in the score
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
            
            # Store solution (use counter as tie breaker)
            nodes.append(_get_node(score_std - score_mean, counter, child_node))
            
            # Update optimal value
            length = len(child_node['subspace'])
            if delta > tol or (np.abs(delta) < tol and length < optimal[2]):
                optimal = (score_mean, score_std, length, child_node.copy())
            
            # Compute models for feature subspaces (in the next round)
            index = len(nodes) - 1
            if child_node['keys'] and length < depth and difference > tol and \
                   not np.isclose(score_mean - score_std, 1):
                queue.append(child_node)
                safe_features[key] = index
            else:
                # Add completed subspace to list
                subspaces[key] = index
            counter += 1
        
        # Update bound
        score_mean, score_std = scores[indices[0]]
        bounds[size] = max(next_bound, score_mean - score_std - eps)

        # Show information about subspace search
        if verbose:
            print('[{:s} - {:d} subspaces remaining]\n'
                  .format(time.strftime("%Y-%m-%d %H:%M:%S"), len(queue)))

    if verbose:
        print('\nBalance search tree')

    # Balance search tree
    paths = {}
    size = len(nodes)
    safe_features = set()

    # Backward removal of subsets
    cursor = 0
    while cursor < len(nodes):
        *_, index, node = nodes[cursor]
        subspace = node['subspace']
        cursor += 1

        # Check if feature subspace is aready safe
        key = _get_key(subspace)
        if key in safe_features:
            continue

        # Perform backward search
        weight = None
        depth = len(subspace)
        if depth > 1:
            node_mean, node_std = _get_score(scoring, scorer, node['stats'],
                                             alpha=alpha)
            
            for keys in itertools.combinations(subspace, depth - 1):
                # Keep order of features for balancing subsets
                features = sorted(keys)
                key = _get_key(features)

                # Check models with smaller subspaces
                if key not in subspaces:
                    params = fit_params.copy()                    
                    if 'eval_set' in params:
                        params['eval_set'] = [(x_test[features], y_test)
                                              for x_test, y_test in params['eval_set']]
                    
                    # Compute score
                    if verbose:
                        print('{:<90s}'.format(','.join(features)), end=' ')

                    # Evaluate model
                    stats = evaluate_model(model, x[features], y, cv=cv, key=key,
                                           fit_params=params, scoring=scoring,
                                           n_jobs=n_jobs)

                    score_mean, score_std = _get_score(scoring, scorer, stats,
                                                       alpha=alpha)
                    if verbose:
                        print('[Score: {:5.2f} +- {:5.2f}]'
                              .format(sign * score_mean, score_std))
                    
                    # Get solution path
                    path = paths.get(key, (np.float(0), ))
                    
                    # Store solution
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
                    # Make sure larger subspaces of equal scores are pruned
                    subset = nodes[subspaces.get(key)]
                    stats = subset[-1]['stats']

                    # Get score
                    score_mean, score_std = _get_score(scoring, scorer, stats,
                                                       alpha=alpha)

                    delta = (score_mean - score_std) - (node_mean - node_std)
                    if np.abs(delta) < tol:
                        weight = np.inf

        # Update score
        if weight is not None:
            nodes[cursor - 1] = (weight, index, node)
        elif depth > 1:
            # Make sure to protect subspaces from coincidentally being removed
            for keys in itertools.combinations(subspace, depth - 1):
                key = _get_key(keys)
                safe_features.add(key)

    # Truncate list of subspaces
    heapq.heapify(nodes)
    safe_features.clear()
    subspaces = []
    subsets = []

    cursor = np.inf
    while nodes:
        *score, node = heapq.heappop(nodes)
        node.pop('path', None)
        node.pop('keys', None)

        # Specify unique key identifier for node
        subspace = sorted(node['subspace'])
        key = _get_key(subspace)
        depth = len(subspace)

        # Ignore duplicate or larger subspaces with lower scores
        if key in safe_features or score[0] > 0 or depth > cursor:
            continue

        # Filter subspaces (i.e, remove complex subspaces that have lower
        # score than the score of its subset subspaces)            
        bucket = None
        for size, subset in subsets:
            # Stop early, if there is no chance for finding a subset
            if depth <= size:
                break

            # Check subspace
            if subset.issubset(subspace):
                bucket = subset
                break

        if bucket is None:
            # Save node and add subspace for duplicate search
            bisect.insort_left(subsets, (depth, set(subspace)))            
            subspaces.append(node)
            safe_features.add(key)
            cursor = depth
            
    # Free memory
    del subsets
    del nodes
    
    # Construct result
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

    # Get mean score
    key = metrics.get_name(key)
    mean = scorer._sign * stats.get(key + '_mean', -scorer._sign * np.inf)

    # Estimate error as standard deviation is not reliable for small number
    # of samples
    values = stats.get(key, None)
    std = np.float_(0)

    if values is not None and values.size > 1:
        le, ue = model_selection.estimate_error(values, confidence=alpha, bootstrapping=0)
        std = max(-le, ue)
    
    # Score and standard deviation
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

