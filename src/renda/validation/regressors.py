# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

from renda.utils.dict_ import transform_dicts


def sklearn_regressor(regressor, X, Y):
    X, Y = transform_dicts(X, Y, transforms=[np.float64])
    Y = transform_dicts(Y, transforms=[np.squeeze])  # sklearn prefers row vectors

    regressor.fit(X["train"], Y["train"])

    predictions = OrderedDict()
    for k, X_ in X.items():
        predictions[k] = regressor.predict(X_)

    return predictions


def gaussian_process_regressor(X, Y, **kwargs):
    from sklearn.gaussian_process import GaussianProcessRegressor

    regressor = GaussianProcessRegressor(**kwargs)

    return sklearn_regressor(regressor, X, Y)
