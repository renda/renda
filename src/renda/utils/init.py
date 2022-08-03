# -*- coding: utf-8 -*-
import math
from typing import Callable, Union

import torch

# Functions from torch.nn.init that are not init functions
_EXCLUDE_FROM_INIT_FNS = [torch.nn.init.calculate_gain]


def is_torch_nn_init_fn_name(name: str) -> bool:
    if not isinstance(name, str):
        raise TypeError(f"Expected string. Got {name}.")

    fn = getattr(torch.nn.init, name, None)
    return fn is not None and fn not in _EXCLUDE_FROM_INIT_FNS


def is_torch_nn_init_fn(fn: Callable) -> bool:
    if not isinstance(fn, Callable):
        raise TypeError(f"Expected callable. Got {fn}.")

    return (
        hasattr(fn, "__module__")
        and fn.__module__ == "torch.nn.init"
        and fn not in _EXCLUDE_FROM_INIT_FNS
    )


def get_name_of_torch_nn_init_fn(fn: Callable) -> str:
    if not isinstance(fn, Callable):
        raise TypeError(f"Expected callable. Got {fn}.")

    if is_torch_nn_init_fn(fn):
        return fn.__name__
    else:
        raise ValueError(f"Cannot find '{fn}' in 'torch.nn.init'.")


def get_torch_nn_init_fn_by_name(name: str) -> Callable:
    if not isinstance(name, str):
        raise TypeError(f"Expected string. Got {name}.")

    if is_torch_nn_init_fn_name(name):
        return getattr(torch.nn.init, name)
    else:
        raise ValueError(f"Cannot find '{name}' in 'torch.nn.init'.")


def torch_default_weight_init_fn(weight: torch.Tensor) -> None:
    # Copied this from PyTorch 1.8.2 (2022-04-21) (comment included)
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


def torch_default_bias_init_fn(bias: torch.Tensor, weight: torch.Tensor) -> None:
    # Copied this from PyTorch 1.8.2 (2022-04-21)
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    # Modified 2022-04-25:
    #   - prevent division by zero
    #   - the following must hold: -bound <= bound
    #   - note: if -bound == bound, bias is inited with zeros
    if fan_in > 0:
        bound = 1 / math.sqrt(fan_in)
    else:
        bound = 0  # pragma: no cover
    torch.nn.init.uniform_(bias, -bound, bound)


def process_init_fn(fn: Union[None, str, Callable]) -> Callable:
    if isinstance(fn, str):
        if is_torch_nn_init_fn_name(fn):
            return get_torch_nn_init_fn_by_name(fn)
        else:
            raise ValueError(f"Cannot find '{fn}' in 'torch.nn.init'.")
    elif isinstance(fn, Callable):
        return fn  # Assume user defined init fn
    elif fn is None:
        return fn
    else:
        raise TypeError(f"Expected string, callable or None. Got {fn}.")


def process_init_fn_kwargs(kwargs: dict) -> dict:
    if not isinstance(kwargs, dict):
        raise TypeError(f"Expected dict. Got {kwargs}.")

    return kwargs
