# -*- coding: utf-8 -*-
from typing import Callable, Dict, Union

import torch


def _get_activation_dict() -> Dict[str, Callable]:
    dict_ = {"Identity": torch.nn.modules.Identity}

    module = torch.nn.modules.activation
    for k, v in module.__dict__.items():
        # Collect module members only
        if hasattr(v, "__module__") and v.__module__ == module.__name__:
            dict_[k] = v

    # HINT:  We let this loop collect the MultiheadAttention activation even
    # though we doubt that it is of use as a ReNDA activation.

    return dict_


def _is_activation_name(name: str) -> bool:
    if not isinstance(name, str):
        raise TypeError(f"Expected string. Got {name}.")

    return name in _get_activation_dict().keys()


def _is_activation_class(class_: Callable) -> bool:
    if not isinstance(class_, Callable):
        raise TypeError(f"Expected callable. Got {class_}.")

    return class_ in _get_activation_dict().values()


def _is_activation_object(obj: Callable) -> bool:
    if not isinstance(obj, Callable):
        raise TypeError(f"Expected callable. Got {obj}.")

    return hasattr(obj, "__class__") and _is_activation_class(obj.__class__)


def _get_activation_class(name: str) -> Callable:
    if not isinstance(name, str):
        raise TypeError(f"Expected string. Got {name}.")

    if _is_activation_name(name):
        return _get_activation_dict()[name]
    else:
        raise ValueError(
            f"Cannot find activation '{name}'. Expected one of these: "
            f"{', '.join(_get_activation_dict().keys())}"
        )


def _get_activation_name(class_or_obj: Callable) -> str:
    if not isinstance(class_or_obj, Callable):
        raise TypeError(f"Expected callable. Got {class_or_obj}.")

    if _is_activation_class(class_or_obj):
        return class_or_obj.__name__
    elif _is_activation_object(class_or_obj):
        return class_or_obj.__class__.__name__
    else:
        raise ValueError(
            f"Cannot find name of activation '{class_or_obj}'. This only "
            f"if the passed activation if one of these: "
            f"{', '.join(_get_activation_dict().keys())}"
        )


def _process_activation(activation: Union[str, Callable]) -> Callable:
    if isinstance(activation, str):
        if _is_activation_name(activation):
            activation_class = _get_activation_class(activation)
            return activation_class
        else:
            raise ValueError(
                f"Cannot find activation '{activation}'. Expected one of "
                f"these: {', '.join(_get_activation_dict().keys())}"
            )
    elif isinstance(activation, Callable):
        # Activation class or object, or user-defined activation
        # TODO (concerning user-defined activations): We let this pass here
        # but this needs to be checked elsewhere
        return activation
    else:
        raise TypeError(f"Expected string or callable. Got {activation}.")


def _process_activation_kwargs(kwargs: dict) -> dict:
    if not isinstance(kwargs, dict):
        raise TypeError(f"Expected dict. Got {kwargs}.")
    return kwargs
