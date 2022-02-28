# -*- coding: utf-8 -*-
"""
@package    tcmi.metrics

@copyright  Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
"""

import numpy as np

from sklearn import metrics as _metrics
from sklearn.utils.validation import check_array, check_consistent_length, _num_samples


def get_name(scoring):
    """Get name of score.
    """
    if not isinstance(scoring, str):
        scoring = get_scorer(scoring)._score_func.__name__.replace('_score', '')
    return scoring.replace('neg_', '').replace('maximum_', 'max_')
        

def get_scorer(scoring):
    """Get a scorer from string
    """
    if isinstance(scoring, str) and scoring in _SCORERS:
        scoring = _SCORERS[scoring]
    return _metrics.get_scorer(scoring)


def mutual_information(y_true, y_pred):
    """Mutual information score.
    """

    # This is a simple wrapper for returning the score as given in y_pred
    return y_pred


def root_mean_squared_error(y_true, y_pred, return_error=False):
    """Root-Mean-Square Error.
    """
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    size = y_pred.size

    rmse = np.linalg.norm(y_true - y_pred) / np.sqrt(size)
    result = (rmse, )
    
    if return_error:
        if _num_samples(y_pred) < 9:
            print('Not enough points. {} datapoints given. At least 9 is required'
                  .format(size))
            return

        le = rmse * (1.0 - np.sqrt(1 - 1.96 * np.sqrt(2) / np.sqrt(size - 1)))
        ue = rmse * (np.sqrt(1 + 1.96 * np.sqrt(2) / np.sqrt(size - 1)) - 1)
        result += ((le, ue), )

    return result if return_error else result[0]


def relative_root_mean_squared_error(y_true, y_pred, return_error=False):
    """Root relative-squared error.
    """
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    result = root_mean_squared_error(y_true, y_pred, return_error=return_error)
    
    value = np.linalg.norm(y_true - y_true.mean()) / np.sqrt(y_true.size)
    if np.isclose(value, 0):
        value = 1

    if return_error:
        rmse, (le, ue) = result
        result = (rmse / value, (le / value, ue / value))
    else:
        result /= value 
    return result


def mean_absolute_error(y_true, y_pred, return_error=False):
    """
    Mean Absolute Error (MAE).
    """
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    mae = np.abs(y_true - y_pred).mean()

    result = (mae, )
    if return_error:
        size = y_pred.size
        le =  mae * (1 - np.sqrt(1 - 1.96 * np.sqrt(2) / np.sqrt(size - 1)))
        ue =  mae * (np.sqrt(1 + 1.96 * np.sqrt(2) / np.sqrt(size - 1)) - 1)
        result += ((le, ue), )

    return result if return_error else result[0]


def relative_mean_absolute_error(y_true, y_pred, return_error=False):
    """Relative mean absolute error.
    """
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    result = mean_absolute_error(y_true, y_pred, return_error=return_error)

    value = np.abs(y_true - y_true.mean()).mean()
    if np.isclose(value, 0):
        value = 1

    if return_error:
        mae, (le, ue) = result
        result = (mae / value, (le / value, ue / value))
    else:
        result /= value 
    return result


def maximum_absolute_error(y_true, y_pred, return_error=False):
    """
    Maximum Absolute Error (MaxAE).
    """
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    maxae = np.abs(y_true - y_pred).max()

    result = (maxae, )
    if return_error:
        size = y_pred.size
        le =  maxae * (1 - np.sqrt(1 - 1.96 * np.sqrt(2) / np.sqrt(size - 1)))
        ue =  maxae * (np.sqrt(1 + 1.96 * np.sqrt(2) / np.sqrt(size - 1)) - 1)
        result += ((le, ue), )

    return result if return_error else result[0]


def relative_maximum_absolute_error(y_true, y_pred, return_error=False):
    """Relative maximum absolute error.
    """
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    result = maximum_absolute_error(y_true, y_pred, return_error=return_error)

    value = np.abs(y_true - y_true.mean()).max()
    if np.isclose(value, 0):
        value = 1

    if return_error:
        mae, (le, ue) = result
        result = (mae / value, (le / value, ue / value))
    else:
        result /= value 
    return result


def mean_error(y_true, y_pred, return_error=False):
    """Mean error (ME).
    """
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    delta =  y_true - y_pred
    me = delta.mean()
    
    result = (me, )
    if return_error:
        error = 1.96 * np.std(delta) / np.sqrt(y_pred.size)
        result += ((error, error), )

    return result if return_error else result[0]


def relative_mean_error(y_true, y_pred, return_error=False):
    """Relative mean error.
    """
    y_true, y_pred = _check_reg_targets(y_true, y_pred)
    result = mean_error(y_true, y_pred, return_error=return_error)

    value = np.mean(y_true - y_true.mean())
    if np.isclose(value, 0):
        value = 1

    if return_error:
        me, (le, ue) = result
        result = (me / value, (le / value, ue / value))
    else:
        result /= value 
    return result


def _check_reg_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same regression task.
    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    return y_true, y_pred


# Standard regression scorers
neg_root_mean_squared_error_scorer = _metrics.make_scorer(root_mean_squared_error, greater_is_better=False)
neg_mean_absolute_error_scorer = _metrics.make_scorer(mean_absolute_error, greater_is_better=False)
neg_mean_error_scorer = _metrics.make_scorer(mean_error, greater_is_better=False)
neg_maximum_absolute_error_scorer = _metrics.make_scorer(maximum_absolute_error, greater_is_better=False)

neg_relative_root_mean_squared_error_scorer = _metrics.make_scorer(relative_root_mean_squared_error, greater_is_better=False)
neg_relative_mean_absolute_error_scorer = _metrics.make_scorer(relative_mean_absolute_error, greater_is_better=False)
neg_relative_mean_error_scorer = _metrics.make_scorer(relative_mean_error, greater_is_better=False)
neg_relative_maximum_absolute_error_scorer = _metrics.make_scorer(relative_maximum_absolute_error, greater_is_better=False)

# Information theory
mutual_information_scorer = _metrics.make_scorer(mutual_information)

# Scorer dictionary
_SCORERS = dict(
    # Absolute metrics    
    neg_root_mean_squared_error=neg_root_mean_squared_error_scorer,
    neg_mean_absolute_error=neg_mean_absolute_error_scorer,
    neg_mean_error=neg_mean_error_scorer,
    neg_maximum_absolute_error=neg_maximum_absolute_error_scorer,

    # Aliases
    rmse=neg_root_mean_squared_error_scorer,
    mae=neg_mean_absolute_error_scorer,
    me=neg_mean_error_scorer,
    maxae=neg_maximum_absolute_error_scorer,

    # Relative metrics
    neg_relative_root_mean_squared_error=neg_relative_root_mean_squared_error_scorer,
    neg_relative_mean_absolute_error=neg_relative_mean_absolute_error_scorer,
    neg_relative_mean_error=neg_relative_mean_error_scorer,
    neg_relative_maximum_absolute_error=neg_relative_maximum_absolute_error_scorer,

    # Information theory
    mutual_information_score=mutual_information_scorer
)
