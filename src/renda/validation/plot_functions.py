# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Union

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision.utils import make_grid

from renda.utils.dict_ import transform_dicts


def _process_X_and_Y(X, Y):
    if not isinstance(X, dict):
        if isinstance(Y, dict):
            raise TypeError(
                f"If X is a non-dict, Y must also be a non-dict. "
                f"Got {X} for X and {Y} for Y."
            )

        X = {"train": X}
        Y = {"train": Y}

        non_dict_inputs = True
    else:
        non_dict_inputs = False

    return X, Y, non_dict_inputs, "train"


def plot_features(
    X: Union[Dict[str, Any], Any],
    Y: Union[Dict[str, Any], Any],
    feature_names: Optional[Sequence[str]] = None,
    show_figures: bool = False,
) -> Dict[str, Any]:
    """ """
    X, Y, non_dict_inputs, single_key = _process_X_and_Y(X, Y)

    # Ensure required types and shapes
    X = transform_dicts(X, transforms=[np.float64])
    transforms = [np.int64, np.squeeze, lambda a: np.expand_dims(a, axis=1)]
    Y = transform_dicts(Y, transforms=transforms)

    # Data properties
    num_features = X["train"].shape
    if len(num_features) == 1:
        num_features = 1
    else:
        num_features = num_features[1]
    num_classes = len(np.unique(Y["train"]))

    # Feature names
    if feature_names is None:
        feature_names = [f"Feature_{d + 1}" for d in range(num_features)]

    # Colors
    palette = sns.color_palette("tab20")
    palette = palette[0::2] + palette[1::2]  # First regular, then pastel colors
    palette = palette * ((num_classes // 20) + 1)  # Repeat colors
    palette = palette[0:num_classes]  # Reduce to actual number of classes

    # Markers
    markers = [  # index, description
        "o",  # 0, circle
        "s",  # 1, square
        "D",  # 2, diamond
        "P",  # 3, plus (filled)
        "X",  # 4, x (filled)
        "<",  # 5, triangle left
        ">",  # 6, triangle right
        "^",  # 7, triangle up
        "v",  # 8, triangle down
    ]
    markers = np.array(markers)
    markers = markers.repeat(20)  # Change marker every 20 classes
    markers = markers.tolist()
    markers = markers[0:num_classes]  # Reduce to actual number of classes

    # Plotting
    figures = OrderedDict()
    for k in X.keys():
        data = pd.DataFrame(data=X[k], columns=feature_names)
        data["Classes"] = Y[k]

        with matplotlib.rc_context({"interactive": show_figures}):

            if num_features == 1:
                with sns.axes_style("ticks"):
                    matplotlib.pyplot.figure()  # Only needed for kdeplot
                    axes = sns.kdeplot(
                        data=data,
                        x=feature_names[0],
                        hue="Classes",
                        palette=palette,
                        fill=True,
                    )
            elif num_features == 2:
                with sns.axes_style("ticks"):
                    axes = sns.jointplot(
                        data=data,
                        x=feature_names[0],
                        y=feature_names[1],
                        hue="Classes",
                        palette=palette,
                        markers=markers,
                    )
            else:
                with sns.axes_style("ticks"):
                    axes = sns.pairplot(
                        data=data,
                        hue="Classes",
                        palette=palette,
                        markers=markers,
                    )

        figures[k] = axes.figure

    if non_dict_inputs:
        return figures[single_key]
    else:
        return figures


def plot_image_grids(
    X: Union[Dict[str, Any], Any],
    Y: Optional[Union[Dict[str, Any], Any]] = None,
    image_shape: Optional[Sequence[int]] = None,
    image_format: str = "CHW",
    num_rows: int = 8,
    num_columns: int = 8,
) -> Dict[str, Any]:
    X, Y, non_dict_inputs, single_key = _process_X_and_Y(X, Y)

    image_grids = OrderedDict()
    for k in X.keys():
        X_ = X[k]

        if image_shape is not None:
            if not (
                isinstance(image_shape, Sequence)
                and len(image_shape) == 3
                and all(isinstance(e, int) for e in image_shape)
                and all(e > 0 for e in image_shape)
            ):
                raise ValueError(
                    f"Expected a sequence of three positive integers. "
                    f"Got {image_shape}."
                )

            X_ = X_.view(-1, *image_shape)

        if image_format != "CHW":
            if not (
                isinstance(image_format, str)
                and len(image_format) == 3
                and set(image_format) == {"C", "H", "W"}
            ):
                raise ValueError(
                    f"Expected a string of length 3 that is any permutation "
                    f"of the characters 'C', 'H' and 'W'. Got {image_format}."
                )

            B = 0  # Batch dimension
            C = image_format.find("C") + 1  # Channel dimension
            H = image_format.find("H") + 1  # Image height dimension
            W = image_format.find("W") + 1  # Image width dimension
            X_ = X_.permute(B, C, H, W)

        if Y is None:
            generator = torch.Generator().manual_seed(0)
            indices = torch.randperm(X_.shape[0], generator=generator)
            indices = indices[num_rows * num_columns]
            image_stack = X_[indices]
        else:
            # HINT: Ignore num_rows, add a row per class instead
            Y_ = Y[k]
            image_stack = []
            for class_ in Y_.unique():
                image_stack_ = X_[Y_ == class_, :]
                image_stack_ = image_stack_[0:num_columns, :]
                image_stack.append(image_stack_)
            image_stack = torch.cat(image_stack)

        image_grids[k] = make_grid(image_stack, nrow=num_columns)

        if non_dict_inputs:
            return image_grids[single_key]
        else:
            return image_grids
