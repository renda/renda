# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

from renda.utils.dict_ import transform_dicts


def sklearn_classifier(classifier, X, Y):
    X, Y = transform_dicts(X, Y, transforms=[np.float64])
    Y = transform_dicts(Y, transforms=[np.squeeze])  # sklearn prefers row vectors

    classifier.fit(X["train"], Y["train"])

    predictions = OrderedDict()
    for k, X_ in X.items():
        predictions[k] = classifier.predict(X_)

    return predictions


def lda_classifier(X, Y, **kwargs):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if len(kwargs) == 0:
        kwargs = {"solver": "lsqr"}

    classifier = LinearDiscriminantAnalysis(solver="lsqr")

    return sklearn_classifier(classifier, X, Y)
