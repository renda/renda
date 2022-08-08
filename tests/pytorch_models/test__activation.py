# -*- coding: utf-8 -*-
import pytest
import torch

from renda.pytorch_models._activation import (
    _get_activation_class,
    _get_activation_name,
    _is_activation_class,
    _is_activation_name,
    _is_activation_object,
    _process_activation,
    _process_activation_kwargs,
)

# Valid names according to PyTorch 1.8.1 documentation
# Future PyTorch versions may introduce further activation
_ACTIVATIONS = {
    "Identity": torch.nn.modules.linear.Identity,
    "Threshold": torch.nn.modules.activation.Threshold,
    "ReLU": torch.nn.modules.activation.ReLU,
    "RReLU": torch.nn.modules.activation.RReLU,
    "Hardtanh": torch.nn.modules.activation.Hardtanh,
    "ReLU6": torch.nn.modules.activation.ReLU6,
    "Sigmoid": torch.nn.modules.activation.Sigmoid,
    "Hardsigmoid": torch.nn.modules.activation.Hardsigmoid,
    "Tanh": torch.nn.modules.activation.Tanh,
    "SiLU": torch.nn.modules.activation.SiLU,
    "Hardswish": torch.nn.modules.activation.Hardswish,
    "ELU": torch.nn.modules.activation.ELU,
    "CELU": torch.nn.modules.activation.CELU,
    "SELU": torch.nn.modules.activation.SELU,
    "GLU": torch.nn.modules.activation.GLU,
    "GELU": torch.nn.modules.activation.GELU,
    "Hardshrink": torch.nn.modules.activation.Hardshrink,
    "LeakyReLU": torch.nn.modules.activation.LeakyReLU,
    "LogSigmoid": torch.nn.modules.activation.LogSigmoid,
    "Softplus": torch.nn.modules.activation.Softplus,
    "Softshrink": torch.nn.modules.activation.Softshrink,
    "MultiheadAttention": torch.nn.modules.activation.MultiheadAttention,
    "PReLU": torch.nn.modules.activation.PReLU,
    "Softsign": torch.nn.modules.activation.Softsign,
    "Tanhshrink": torch.nn.modules.activation.Tanhshrink,
    "Softmin": torch.nn.modules.activation.Softmin,
    "Softmax": torch.nn.modules.activation.Softmax,
    "Softmax2d": torch.nn.modules.activation.Softmax2d,
    "LogSoftmax": torch.nn.modules.activation.LogSoftmax,
}

# Valid kwargs according to PyTorch 1.8.1 documentation
# Interfaces may change in future PyTorch versions
_ACTIVATION_KWARGS = {
    "Threshold": {
        "threshold": 0.0,
        "value": 0.0,
    },
    "MultiheadAttention": {
        "embed_dim": 10,
        "num_heads": 5,
    },
}


def test__is_activation_name():
    for activation_name in _ACTIVATIONS.keys():
        assert _is_activation_name(activation_name)

    assert not _is_activation_name("_")  # Unlikely activation name

    with pytest.raises(TypeError):
        _is_activation_name(lambda: 0)
    with pytest.raises(TypeError):
        _is_activation_name(0)


def test__is_activation_class():
    for activation_class in _ACTIVATIONS.values():
        assert _is_activation_class(activation_class)

    assert not _is_activation_class(lambda: 0)
    assert not _is_activation_class(min)  # Python built-in
    assert not _is_activation_class(torch.squeeze)  # From another PyTorch module
    assert not _is_activation_class(torch.nn.Sigmoid())  # Object instead of class

    def local_fn():
        pass

    assert not _is_activation_class(local_fn)

    del local_fn

    # Invalid types
    with pytest.raises(TypeError):
        _is_activation_class("Sigmoid")
    with pytest.raises(TypeError):
        _is_activation_class(0)


def test__is_activation_object():
    for activation_name, activation_class in _ACTIVATIONS.items():
        kwargs = _ACTIVATION_KWARGS.get(activation_name, {})
        assert _is_activation_object(activation_class(**kwargs))

    assert not _is_activation_object(lambda: 0)
    assert not _is_activation_object(min)  # Python built-in
    assert not _is_activation_object(torch.squeeze)  # From another PyTorch module
    assert not _is_activation_object(torch.nn.Sigmoid)  # Class instead of object

    def local_fn():
        pass

    assert not _is_activation_object(local_fn)

    del local_fn

    # Invalid types
    with pytest.raises(TypeError):
        _is_activation_object("Sigmoid")
    with pytest.raises(TypeError):
        _is_activation_object(0)


def test__get_activation_class():
    for activation_name, activation_class in _ACTIVATIONS.items():
        assert _get_activation_class(activation_name) is activation_class

    # Invalid types
    with pytest.raises(TypeError):
        _get_activation_class(lambda: 0)  # Callable
    with pytest.raises(TypeError):
        _get_activation_class(0)

    # Invalid values
    with pytest.raises(ValueError):
        _get_activation_class("_")  # Unlikely activation name


def test__get_activation_name():
    for activation_name, activation_class in _ACTIVATIONS.items():
        # Pass activation class
        assert _get_activation_name(activation_class) == activation_name

        # Pass activation object
        kwargs = _ACTIVATION_KWARGS.get(activation_name, {})
        assert _get_activation_name(activation_class(**kwargs)) == activation_name

    # Invalid types
    with pytest.raises(TypeError):
        _get_activation_name("Sigmoid")
    with pytest.raises(TypeError):
        _get_activation_name(0)

    # Invalid values
    with pytest.raises(ValueError):
        _get_activation_name(lambda: 0)
    with pytest.raises(ValueError):
        _get_activation_name(min)  # Python built-in
    with pytest.raises(ValueError):
        _get_activation_name(torch.squeeze)  # From another PyTorch module
    with pytest.raises(ValueError):

        def local_fn():
            pass

        _get_activation_name(local_fn)

        del local_fn


def test__process_activation():
    for activation_name, activation_class in _ACTIVATIONS.items():
        # Pass activation name
        same_activation_class = _process_activation(activation_name)
        assert same_activation_class is activation_class

        # Pass activation class
        same_activation_class = _process_activation(activation_class)
        assert same_activation_class is activation_class

        # Pass activation object
        activation_kwargs = _ACTIVATION_KWARGS.get(activation_name, {})
        activation_object = activation_class(**activation_kwargs)
        same_activation_object = _process_activation(activation_object)
        assert same_activation_object is activation_object

    # User-defined activation
    # TODO: We let this pass here but this needs to be checked elsewhere
    def local_fn():
        pass

    _process_activation(local_fn)

    del local_fn

    with pytest.raises(TypeError):
        _process_activation(0)

    with pytest.raises(ValueError):
        _process_activation("_")  # Unlikely activation name


def test__process_activation_kwargs():
    kwargs = {}
    assert _process_activation_kwargs(kwargs) == kwargs
    kwargs = {"key": 0}
    assert _process_activation_kwargs(kwargs) == kwargs

    with pytest.raises(TypeError):
        _process_activation_kwargs([0, 1, 2])
    with pytest.raises(TypeError):
        _process_activation_kwargs(0)
