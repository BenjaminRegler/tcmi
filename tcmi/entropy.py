# -*- coding: utf-8 -*-
"""
@package    tcmi.entropy

@copyright  Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
"""

import joblib
import functools
import itertools
import collections
import numpy as np

from .utils import ndindex, chunk_iterator
from scipy.stats import hypergeom as _hypergeom


def cumulative_mutual_information(y, x, adjust=0, cache=False, n_jobs=None,
                                  verbose=0, pre_dispatch='2*n_jobs',
                                  return_scores=False):
    """Compute the cumulative mutual information shared by X and Y.
    """
    eps = np.finfo(np.float_).eps
    kwargs = dict(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch,
                  return_vector=True)

    x = clean_data(x, allow_duplicates=True)

    hce, hce_corr, ce, counts = conditional_cumulative_entropy(y, *x,
                                                               return_counts=True,
                                                               **kwargs)
    hce_corr = hce_corr.sum(axis=1)
    hce = hce.sum(axis=1)

    hce0 = cumulative_baseline_correction(y, *x, correction=counts,
                                          cache=cache, **kwargs)
    hce0 = hce0.sum(axis=1)
    ce0 = cumulative_entropy(y)

    score = 1 - hce / ce.sum(axis=1)
    score0 = 1 - hce0 / ce0

    score_corr = np.zeros_like(score)
    if np.any(hce_corr > eps):
        score_corr = np.clip(1 - hce / np.maximum(hce_corr, eps), 0, 1)

    total_score = np.clip(score - score0 - score_corr, 0, 1)

    result = total_score.mean()
    if return_scores:
        result = (result, (total_score, score, score_corr, score0))
    return result


def cumulative_entropy(y, return_vector=False, return_inverse=False,
                       direction=0):
    """Computes the cumulative entropy of y.
    """
    index = _direction_to_index(direction)
    
    inverse = False
    if return_inverse:
        y, inverse, probability = np.unique(y, return_inverse=True,
                                            return_counts=True)
    else:
        y, probability = np.unique(y, return_counts=True)

    dy = np.diff(y)
    size = y.size

    entropies = []
    directions = ((index, ) if isinstance(index, int) else (-1, 1))
    for direction in directions:
        ce = np.zeros(size, dtype=np.float_)

        if direction == -1:
            p = np.add.accumulate(probability, dtype=np.float_)
            p /= p[-1]

            start = (1 if return_vector else size - 1)

            for i in range(start, size):
                vector = p[:i]
                ce[i] -= np.inner(dy[:i], vector * np.log2(vector))
            entropies.append(ce if return_vector else ce[-1])

        elif direction == 1:
            counts = probability.sum()
            p = np.add.accumulate(probability[::-1], dtype=np.float_)[::-1]
            p /= counts

            stop = (size if return_vector else 2)

            for i in range(1, stop):
                vector = p[i:]
                ce[i - 1] -= np.inner(dy[i - 1:], vector * np.log2(vector))
            entropies.append(ce if return_vector else ce[0])

        else:
            raise RuntimeError('Unknown direction "{:d}".'.format(direction))

    result = (np.array(entropies) if len(entropies) > 1 else entropies[0])
    if return_inverse:
        result = (result, inverse)
    return result


def _compute_entropy(y, x, x_sorted, dimensions, masks, offset=0):
    """Compute entropy.
    """
    cache = (None, True)
    shape = (2, y.size)
    counts = np.zeros(shape, dtype=np.int_)
    entropy = np.zeros_like(counts, dtype=np.float)
    entropy0 = np.zeros_like(counts, dtype=np.float)
    entropy_corr = np.zeros_like(counts, dtype=np.float)

    local_storage = {}
    for i, dimension in enumerate(itertools.product(*dimensions)):
        key, mask = cache

        if key != dimension[:-1]:
            mask = np.logical_and.reduce(
                [masks[i][j] for i, j in enumerate(dimension[:-1])])
            cache = (dimension[:-1], mask)

        mask = np.logical_and(mask, masks[-1][dimension[-1]])

        key = np.packbits(mask).tobytes()

        bucket = local_storage.get(key, [])
        if bucket:
            for slot, index, value, value0, value_corr in bucket:
                counts[slot, index] += 1
                entropy[slot, index] -= value
                entropy0[slot, index] -= value0
                entropy_corr[slot, index] -= value_corr

            continue

        bucket = []
        for slot, submask in enumerate(mask, offset):
            size = submask.sum()
            index = max(size - 1, 0)

            counts[slot, index] += 1
            if size < 2:
                bucket.append((slot, index, np.float(0), np.float(0), np.float(0)))
                continue

            yc = np.compress(submask, y)

            umask = np.ones(size + 1, dtype=np.bool_)
            umask[1:-1] = (yc[1:] != yc[:-1])
            yu = np.compress(umask[:-1], yc)

            px = []
            for variable in x[slot]:
                value = np.compress(submask, variable)

                if np.any(value != value[0]):
                    px.append(
                        np.searchsorted(np.sort(value), value, side='right'))

            if not px:
                px.append(np.full(size, size, dtype=np.uintp))
            px = np.minimum.reduce(px)

            if slot == 1:
                py = np.searchsorted(yc, yc, side='left')
                np.subtract(size, py, out=py)

                pxy = np.minimum(px, py)
                py = py / py[0]

                dy = np.ediff1d(yu, to_begin=0)
            else:
                py = np.searchsorted(yc, yc, side='right')

                pxy = np.minimum(px, py)
                py = py / py[-1]

                dy = np.ediff1d(yu, to_end=0)

            if yu.size < yc.size:
                delta_y = np.zeros(size, dtype=np.float_)
                idx = np.nonzero(umask[:-1])[0]
                delta_y[idx] = dy
                dy = delta_y
                del delta_y

            value0 = np.einsum('i,i,i', dy, py, np.log2(py))
            value = np.einsum('i,i,i', dy, pxy / size, np.log2(pxy / px))

            px_s = np.sort(px)
            pxy_s = np.sort(pxy)
            if slot:
                px_s = px_s[::-1]
                pxy_s = pxy_s[::-1]
            value_corr = np.einsum('i,i,i', dy, pxy_s / size, np.log2(pxy_s / px_s))

            value_corr = np.minimum(value_corr, value)

            entropy[slot, index] -= value
            entropy0[slot, index] -= value0
            entropy_corr[slot, index] -= value_corr

            bucket.append((slot, index, value, value0, value_corr))

        local_storage[key] = bucket

    return entropy, entropy0, entropy_corr, counts


def conditional_cumulative_entropy(y, *x, direction=0, cache=False,
                                   early_stopping_rounds=10, threshold=1e-3,
                                   n_jobs=None, verbose=0, pre_dispatch='2*n_jobs',
                                   return_counts=False, return_vector=False):
    """Computes the conditional cumulative entropy of y given x.
    """
    y = np.asarray(y)

    order = y.argsort(kind='mergesort')

    y = np.take(y, order)
    x = tuple(np.take(value, order) for value in x)
    x = clean_data(x, allow_duplicates=True, return_original=False,
                   return_reverse=True)

    x_sorted = []
    for variables in x:
        variables = tuple(np.sort(v) for v in variables)
        x_sorted.append(variables)

    masks, dimensions = _get_selection_masks(x[0], direction=direction,
                                             return_dimension=True)

    offset = _direction_to_index(direction)
    if not isinstance(offset, int):
        offset = 0

    shape = (2, y.size)
    counts = np.zeros(shape, dtype=np.int_)
    entropy = np.zeros_like(counts, dtype=np.float)
    entropy0 = np.zeros_like(counts, dtype=np.float)
    entropy_corr = np.zeros_like(counts, dtype=np.float)

    processor = joblib.Parallel(n_jobs=n_jobs, backend='multiprocessing',
                                verbose=verbose, pre_dispatch=pre_dispatch)
    callback = joblib.delayed(_compute_entropy)

    chunks = processor._effective_n_jobs()
    iterator = chunk_iterator(ndindex(*dimensions, raw=True, grouped=True,
                                      multi_index=True), chunks)

    buffer = collections.deque(maxlen=early_stopping_rounds)
    buffer.extend((i, 0, -1) for i in range(2))

    counter = 0
    total = np.prod(dimensions)
    eps = np.finfo(np.float).eps

    for i, indices in enumerate(iterator):
        levels, indices = tuple(zip(*indices))

        results = processor(callback(y, x, x_sorted, index, masks, offset=offset)
                            for index in indices)

        for level, values in zip(levels, results):
            entropy += values[0]
            entropy0 += values[1]
            entropy_corr += values[2]
            counts += values[3]

            total_counts = np.count_nonzero(counts, axis=1)
            ce = entropy0.sum(axis=1) / (total_counts + eps)
            hce = entropy.sum(axis=1) / (total_counts + eps)

            xx, yy, scores = tuple(zip(*buffer))

            progress = counts.sum() / total
            score = np.mean(hce / (ce + eps))
            buffer.append((progress, score, np.mean(yy)))

            model = np.poly1d(np.polyfit(xx, yy, 1))
            value = model(1)

            model = np.poly1d(np.polyfit(xx, scores, 1))
            value0 = model(1)

            if 0 <= value <= 1 and 0 <= value0 <= 1 and \
                    abs(value - value0) < threshold:
                counter += 1

                if counter > early_stopping_rounds > 0:
                    break
            else:
                counter = 0

        if counter > early_stopping_rounds > 0:
            break

    results = (entropy, entropy_corr, entropy0)
    if not return_vector:
        results = tuple(vector.sum(axis=1) for vector in results)
    if return_counts:
        results += (counts, )

    index = _direction_to_index(direction)
    return tuple(result[index] for result in results)


def cumulative_baseline_correction(y, *x, direction=0, correction=None, cache=False,
                                   n_jobs=None, verbose=0, pre_dispatch='2*n_jobs',
                                   return_vector=False, return_baseline=False):
    """Computes the correction of chance baseline correction of y given x.
    """
    index = _direction_to_index(direction)

    y = np.asarray(y)
    weights = []

    order = y.argsort(kind='mergesort')
    x = tuple(np.take(value, order) for value in x)
    variables = clean_data(x, allow_duplicates=True, return_original=False,
                           return_reverse=True)

    for variable in variables:
        weight = _get_permutation_weights(*variable)
        weights.append(weight)

    if correction is not None:
        counts = correction / correction.T[-1][:, np.newaxis]

        weight = np.arange(y.size + 1, dtype=np.int_)
        np.power(weight, len(x), out=weight)
        weight = np.diff(weight)[::-1]

        mask = np.logical_and(counts == 0, weights)

        weights = counts / weight
        weights[mask] = 1

    processor = joblib.Parallel(n_jobs=n_jobs, backend='multiprocessing',
                                verbose=verbose, pre_dispatch=pre_dispatch)
    callback = joblib.delayed(compute_cumulative_baseline)

    baseline = processor(callback(y, i + 1, return_vector=False, direction=0)
                         for i in range(y.size))
    baseline = np.column_stack(baseline)

    corrections = []
    for i, weight in enumerate(weights):
        correction = weight[1:] * baseline[i][1:]
        corrections.append(correction / weight.sum())

    if return_vector:
        corrections = np.atleast_2d(corrections)
    else:
        corrections = [np.sum(correction) for correction in corrections]

    results = corrections[index]
    if return_baseline:
        results = (results, baseline)
    return results


def compute_cumulative_baseline(y, component, direction=0, return_entropy=False,
                                return_vector=False):
    """Computes the correction of chance baseline correction of response.
    """
    index = _direction_to_index(direction)

    y = np.sort(y, kind='mergesort')
    if component < 2:
        score = np.zeros(2, dtype=np.float_)

        result = (score[index], )
        if return_entropy:
            entropy = np.zeros((2, component), dtype=np.float_)
            result += (entropy[index], )
        if return_vector:
            entropy = np.zeros((2, component), dtype=np.float_)
            result += (entropy, )

        return result[0] if len(result) == 1 else result

    n = y.size
    b = component

    numbers = np.arange(b + 1, dtype=np.int_)
    ce = np.zeros((2, numbers.size), dtype=np.float_)

    ce[0, 1:] = (numbers[1:] / b) * np.log2(numbers[1:] / b)
    ce[1, 1:] = ce[0, 1:][::-1]

    entropy = np.zeros((2, n), dtype=np.float_, order='f')
    fraction = b / n

    size = n - b + 1

    ij = np.arange(n, b - 2, -1, dtype=np.int_)
    ij[0] = 0

    np.add.accumulate(ij, out=ij)

    matrix = np.zeros((ij[-1], 2), dtype=np.float_)
    y = np.column_stack((y, -y[::-1]))

    partial_entropy = np.zeros((n, 2), dtype=np.float_)
    vector = np.zeros((2, b), dtype=np.float_)
    scores = np.zeros(2, dtype=np.float_)

    for i in range(1, n):
        left = max(1, i + b - n)
        right = 1 + min(i, b)
        nij = numbers[left:right]

        probability = _hypergeom.pmf(b - nij, n - 1, n - i, b - 1)


        cutoff = (-1 if b < right else None)
        jx = max(0, i - (n - b + 1))

        for j, mij in enumerate(b - nij[:cutoff]):
            number = (n - i) - (mij - 1)

            m = np.int_(mij - 1)
            weights = 1 - m / np.arange(n - i, m, -1)
            weights[0] = mij / (n - i)

            np.multiply.accumulate(weights, out=weights)

            delta = y[i:number + i] - y[i - 1]

            value = delta * weights[:, None] * ce[0, left + j] * probability[j]
            offset = min(number, n - i)
            s = slice(i, i + offset)

            partial_entropy[s] -= value[:offset]
            scores -= value.sum(axis=0)

            k = j + jx
            cursor = i - k - 1
            l, r = ij[cursor:cursor + 2]

            if k > 0:
                factor = matrix[l + i - cursor - 2]
                partial_entropy[s] += np.outer(weights[:offset], factor)

                if i - k < size:
                    matrix[r:ij[cursor + 2]] += matrix[l + 1:r]

            matrix[l:r] = partial_entropy[cursor + 1:]

            value = partial_entropy[s].T
            vector[:, k + 1] += value.sum(axis=-1)

            entropy[:, s] += value
            partial_entropy.fill(0)

    entropy *= fraction
    vector *= fraction
    scores *= fraction

    result = (scores[index], )
    if return_entropy:
        result += (entropy[index], )
    if return_vector:
        result += (vector[index], )
    return result if len(result) > 1 else result[0]


def clean_data(x, allow_duplicates=False, return_original=True,
               return_reverse=False):
    """Clean data.
    """
    order = []
    values = []
    reverse = []
    duplicates = set()

    length = len(x)
    for i, vector in enumerate(x, 1):
        unique_vector = np.unique(vector)
        size = unique_vector.size

        if size > 1 or length == i:
            vs = np.sort(vector)
            unique_vector = np.searchsorted(vs, vector, side='right')

            fingerprint = unique_vector.tobytes()
            reversed_fingerprint = unique_vector[::-1].tobytes()

            if fingerprint not in duplicates:
                if not allow_duplicates:
                    duplicates.add(fingerprint)
                    duplicates.add(reversed_fingerprint)

                values.append(vector if return_original else unique_vector)
                order.append(size)

                if return_reverse:
                    unique_vector = np.searchsorted(vs, vector, side='left')
                    reverse.append(vector[::-1] if return_original
                                   else vs.size - unique_vector)

    order = np.argsort(order, kind='mergesort')[::-1]

    result = (tuple(values[i] for i in order), )
    if return_reverse:
        result += (tuple(reverse[i] for i in order), )
    return result if len(result) > 1 else result[0]


def _direction_to_index(direction):
    """Map direction identifier to index.
    """
    directions = {-1: 0, 0: slice(None), 1: 1, '<=': 0, '<=>': slice(None), '>=': 1}

    if direction not in directions:
        raise RuntimeError('Unknown direction "{:d}".'.format(direction))

    return directions[direction]


def _get_permutation_weights(*x):
    """Get permutation weights.
    """
    x = [np.sort(v, kind='mergesort') for v in x]
    size = x[0].size

    #
    #
    dtype = 'S' + str(len(x) * x[0].itemsize)

    data = np.column_stack(x)
    data = data.view(dtype=dtype).flatten()

    block = np.ones(size + 1, dtype=np.bool_)
    block[1:-1] = data[1:] != data[:-1]
    block = np.nonzero(block)[0]

    weights = np.zeros(size, dtype=np.bool_)
    for i in range(1, block.size):
        left, right = block[i - 1:i + 1]
        width = right - left

        if width > 1 and min(width, size - right) > 0:
            for j in range(left + 1, right):
                limits = []
                for variable in x:
                    limit = np.searchsorted(variable, variable[j], side='right')
                    limits.append(size - limit)

                if len(limits) > 1 and min(limits) > 0:
                    weights[j - 1] = 1

        weights[right - 1] = 1

    return weights


def _get_selection_masks(x, direction=0, return_dimension=False):
    """Get selection masks of x.
    """
    index = _direction_to_index(direction)
    operators = [np.less_equal, np.greater_equal]
    if isinstance(index, int):
        operators = (operators[index], )

    masks = []
    dimensions = []

    for variable in x:
        values = np.unique(variable)

        mask = []
        for operator in operators:
            mask.append(tuple(operator(variable, value) for value in values))

        mask = np.stack(mask, axis=1)

        dimensions.append(values.size)
        masks.append(mask)

    return (masks, dimensions) if return_dimension else masks
