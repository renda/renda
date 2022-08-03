# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

from renda.utils.dict_ import transform_dicts


def sklearn_metric(metric, Y_true, Y_pred):
    transforms = [np.float64, np.squeeze]  # sklearn prefers row vector
    Y_true, Y_pred = transform_dicts(Y_true, Y_pred, transforms=transforms)

    result = OrderedDict()
    for k in Y_true.keys():
        result[k] = metric(Y_true[k], Y_pred[k])

    return result


# =============================================================================
# Classification metrics
# =============================================================================
def classification_error(Y, Y_pred):
    from sklearn.metrics import zero_one_loss

    metric = lambda Y, Y_pred: zero_one_loss(Y, Y_pred) * 100  # noqa: E731

    return sklearn_metric(metric, Y, Y_pred)


# =============================================================================
# Regression metrics
# =============================================================================
def mean_absolute_percentage_error(Y, Y_pred):
    from sklearn.metrics import mean_absolute_percentage_error as MAPE

    metric = lambda Y, Y_pred: MAPE(Y, Y_pred) * 100  # noqa: E731

    return sklearn_metric(metric, Y, Y_pred)
