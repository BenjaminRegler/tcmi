# -*- coding: utf-8 -*-
"""
@package    tcmi.utils

@copyright  Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
"""

import itertools
import numpy as np
import pandas as pd

from joblib import hashing

# Mappings
_MAPPINGS = {
    '|{}|': [np.abs],
    '-{}': [np.negative],
    '-|{}|': [np.abs, np.negative]
}


def get_fingerprint(x):
    """Computes the fingerprint of a Numpy vector.
    """
    fingerprint = np.searchsorted(np.sort(x), x, side='left')
    return compute_hash(fingerprint)


def compute_hash(x):
    """Hash object.
    """
    return hashing.hash(x, hash_name='md5')


def is_numeric(obj):
    """Check if object is numeric.
    """
    
    # Check numpy object and pandas dataframes 
    flag = bool(isinstance(obj, np.ndarray) and obj.dtype.kind in 'OSU')
    if isinstance(obj, pd.Series):
        flag |= pd.api.types.is_categorical_dtype(obj)
    elif isinstance(obj, pd.DataFrame):
        for key in obj.columns:
            flag |= is_numeric(obj[key])
    
    # General check (may not cover all cases)
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs) and not flag


def prepare_data(data, target, copy=False):
    """Prepare data by agumenting feature space.
    """   
    # Make copy
    if copy:
        data = data.copy()

    if isinstance(target, str):
        target = [target]
    
    # Augment feature space
    keys = sorted(data)
    hashes = set()
    
    for key in keys:
        value = data[key]
        if key in target:
            continue

        # Compute hash
        fingerprint = get_fingerprint(value)
        hashes.add(fingerprint)

        # Generate features
        for label, chain in sorted(_MAPPINGS.items(), key=lambda x: x[0]):
            alias = label.replace('{}', '')
            label = label.format(key)

            # Apply function chain
            result = value
            if is_numeric(value):
                for func in chain:
                    result = func(result)
                
            # Compute hash and compare with database
            fingerprint = get_fingerprint(result)
            if fingerprint not in hashes:
                data[label] = result
                hashes.add(fingerprint)
    
    return data


def filter_subsets(subsets, remove_duplicates=False):
    """Filter subsets.
    """
    mappings = sorted(_MAPPINGS, key=lambda x: (-len(x), x))
    
    results = []
    duplicates = set()
    for subset in subsets:        
        # Normalize subspace
        subspace_original = subset['subspace']
        size = len(subspace_original)

        normalized_subspace = subspace_original
        for mapping in mappings:
            prefix, suffix = mapping.split('{}', 1)
            normalized_subspace = tuple(strip(k, prefix, suffix)
                                        for k in normalized_subspace)

        # Filter duplicate keys
        if remove_duplicates:
            subspace = []
            for k in normalized_subspace:
                if k not in subspace:
                    subspace.append(k)
            normalized_subspace = tuple(subspace)

        # Generate key for duplicate search
        key = ','.join(sorted(normalized_subspace))
        if key in duplicates:            continue

        # Process subspace
        duplicates.add(key)
        subset = subset.copy()
        subset.update({
            'subspace': normalized_subspace,
            'subspace_original': subspace_original
        })
        results.append(subset)

    return results


def strip(text, prefix='', suffix=''):
    """Remove substring from the left and right side of the text.
    """
    return strip_right(strip_left(text, prefix), suffix)


def strip_left(text, prefix):
    """Remove substring from the left side of the text.
    """
    if prefix and text.startswith(prefix):
        text = text[len(prefix):]
    return text


def strip_right(text, suffix):
    """Remove substring from the right side of the text.
    """
    if suffix and text.endswith(suffix):
        text = text[:-len(suffix)]
    return text


def wrap_iterator(iterator, wrap=False, index=None):
    """Return wrapped iterator for `yield from` syntax.
    """
    if wrap:
        iterator = iter([iterator]) 
    return (iterator if index is None
            else itertools.zip_longest((), iterator, fillvalue=index))


def chunk_iterator(iterable, n):
    """Group an iterator in chunks of n without padding.
    """
    iterator = iter(iterable)
    return iter(lambda: list(itertools.islice(iterable, n)), [])


def index_split(index, dimension=1, method='symmetric'):
    """Splits an index into parts.
    """
    splits = []
    size = len(index)

    if method == 'adaptive':
        # Dynamically refine index like a mesh
        stack = [index]
        split = []

        while stack:
            indices = stack
            stack = []

            for index in indices:
                size = len(index)
                if size < 3:
                    split.extend(index)
                    continue

                divider = size // 2
                split.append(index[divider])

                stack.append(index[:divider])
                stack.append(index[divider + 1:])

            splits.append(np.array(split))
            split = []

    elif method == 'symmetric':
        # Read index from left and right symmetrically
        for i in range(np.math.ceil(size / 2)):
            a, b = index[i], index[-1-i]
            split = np.array((a, b))

            splits.append(split if a < b else split[0:1])

    elif method == 'interleave':
        # Interweave indices
        divider = max(2, np.sqrt(size).astype(np.int) // dimension)
        step = size // divider + 1

        splits = []
        for i in range(step):
            splits.append(index[i::step])

    else:
        raise KeyError('Unknown split method "{:s}".'.format(method))

    # Return arrays like array_split
    return splits


def ndindex(*indices, method='symmetric', raw=False, grouped=False,
            multi_index=False):
    """An N-dimensional iterator object to index arrays.
    """
    # Split indices into groups
    dimension = len(indices)
    splits = [index_split(np.arange(index), dimension=dimension,
                          method=method) for index in indices]

    # Create pool of indices to iterate over
    pool = [split.pop(0) for split in splits]
    empty = np.array([], dtype=np.int_)

    # Return multi-index?
    iteration = None
    if multi_index:
        iteration = 0

    # Perform initial multi-dimensional indexing
    iterator = (pool if raw else itertools.product(*pool))
    yield from wrap_iterator(iterator, wrap=grouped, index=iteration)

    # Adaptively refine multi-dimensional index
    while True:
        # Reset loop
        loop = False
        if multi_index:
            iteration += 1

        # Get next mesh refinement
        staging = []
        for split in splits:
            flag = len(split) > 0
            loop |= flag

            staging.append(split.pop(0) if flag else empty)

        # Stop iterations
        if not loop:
            break

        for i in range(dimension):
            # Ignore empty dimensions
            if staging[i].size == 0:
                continue

            # Merge indices
            indices = pool.copy()
            for j in range(i + 1, dimension):
                indices[j] = np.concatenate((pool[j], staging[j]))
            indices[i] = staging[i]

            iterator = (indices if raw else itertools.product(*indices))
            yield from wrap_iterator(iterator, wrap=grouped, index=iteration)

        # Enlarge pool
        pool = [np.concatenate(v) for v in zip(pool, staging)]
