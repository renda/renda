# -*- coding: utf-8 -*-
import math
from typing import Callable, Union

import torch

# List callables that may be condused with init functions
_NO_INIT_FN = [torch.nn.init.calculate_gain]


def _is_init_fn(fn: Callable) -> bool:
    if not isinstance(fn, Callable):
        raise TypeError(f"Expected callable. Got {fn}.")

    return (
        hasattr(fn, "__module__")
        and fn.__module__ == "torch.nn.init"
        and fn not in _NO_INIT_FN
    )


def _is_init_fn_name(name: str) -> bool:
    if not isinstance(name, str):
        raise TypeError(f"Expected string. Got {name}.")

    fn = getattr(torch.nn.init, name, None)
    return fn is not None and fn not in _NO_INIT_FN


def _get_init_fn(name: str) -> Callable:
    if not isinstance(name, str):
        raise TypeError(f"Expected string. Got {name}.")
    if not _is_init_fn_name(name):
        raise ValueError(f"Cannot find '{name}' in 'torch.nn.init'.")

    return getattr(torch.nn.init, name)


def _get_init_fn_name(fn: Callable) -> str:
    if not isinstance(fn, Callable):
        raise TypeError(f"Expected callable. Got {fn}.")
    if not _is_init_fn(fn):
        raise ValueError(f"Cannot find '{fn}' in 'torch.nn.init'.")

    return fn.__name__


def _default_weight_init_fn(weight: torch.Tensor) -> None:
    # Copied this from PyTorch 1.8.2 (2022-04-21) (comment included)
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


def _default_bias_init_fn(bias: torch.Tensor, weight: torch.Tensor) -> None:
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


def _process_init_fn(fn: Union[None, str, Callable]) -> Callable:
    if fn is None:
        return None
    elif isinstance(fn, str):
        if _is_init_fn_name(fn):
            return _get_init_fn(fn)
        else:
            raise ValueError(f"Cannot find '{fn}' in 'torch.nn.init'.")
    elif isinstance(fn, Callable):
        # User defined init function
        # TODO: We let this pass here but this needs to be checked elsewhere
        return fn
    else:
        raise TypeError(f"Expected string, callable or None. Got {fn}.")


def _process_init_fn_kwargs(kwargs: dict) -> dict:
    if not isinstance(kwargs, dict):
        raise TypeError(f"Expected dict. Got {kwargs}.")

    return kwargs
