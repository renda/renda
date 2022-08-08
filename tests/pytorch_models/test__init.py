# -*- coding: utf-8 -*-
import random

import pytest
import torch
from torch.nn.init import calculate_gain

from renda.pytorch_models._init import (
    _default_bias_init_fn,
    _default_weight_init_fn,
    _get_init_fn,
    _get_init_fn_name,
    _is_init_fn,
    _is_init_fn_name,
    _process_init_fn,
    _process_init_fn_kwargs,
)
from renda.utils.seeding import max_seed, min_seed

# Valid names according to PyTorch 1.8.1 documentation
# Future PyTorch versions may introduce further init functions
_INIT_FUNCTIONS = {
    "uniform_": torch.nn.init.uniform_,
    "normal_": torch.nn.init.normal_,
    "constant_": torch.nn.init.constant_,
    "ones_": torch.nn.init.ones_,
    "zeros_": torch.nn.init.zeros_,
    "eye_": torch.nn.init.eye_,
    "dirac_": torch.nn.init.dirac_,
    "xavier_uniform_": torch.nn.init.xavier_uniform_,
    "xavier_normal_": torch.nn.init.xavier_normal_,
    "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
    "kaiming_normal_": torch.nn.init.kaiming_normal_,
    "orthogonal_": torch.nn.init.orthogonal_,
    "sparse_": torch.nn.init.sparse_,
}


def test__is_init_fn():
    for fn in _INIT_FUNCTIONS.values():
        assert _is_init_fn(fn)

    def local_fn():
        pass

    assert not _is_init_fn(local_fn)
    assert not _is_init_fn(lambda: 0)
    assert not _is_init_fn(min)  # Python built-in
    assert not _is_init_fn(torch.squeeze)  # From other module
    assert not _is_init_fn(calculate_gain)  # Although in init module

    del local_fn

    with pytest.raises(TypeError):
        _is_init_fn("uniform_")
    with pytest.raises(TypeError):
        _is_init_fn(0)


def test__is_init_fn_name():
    for name in _INIT_FUNCTIONS.keys():
        assert _is_init_fn_name(name)

    assert not _is_init_fn_name("_")  # Unlikely init function name
    assert not _is_init_fn_name("calculate_gain")  # Although in init module

    with pytest.raises(TypeError):
        _is_init_fn_name(torch.nn.init.uniform_)
    with pytest.raises(TypeError):
        _is_init_fn_name(0)


def test__get_init_fn():
    for fn_name, fn in _INIT_FUNCTIONS.items():
        assert _get_init_fn(fn_name) is fn

    with pytest.raises(TypeError):
        _get_init_fn(torch.nn.init.uniform_)
    with pytest.raises(TypeError):
        _get_init_fn(0)

    with pytest.raises(ValueError):
        _get_init_fn("_")  # Unlikely init function name
    with pytest.raises(ValueError):
        _get_init_fn("calculate_gain")  # Although in init module


def test__get_init_fn_name():
    for fn_name, fn in _INIT_FUNCTIONS.items():
        assert _get_init_fn_name(fn) == fn_name

    with pytest.raises(TypeError):
        _get_init_fn_name("uniform_")
    with pytest.raises(TypeError):
        _get_init_fn_name(0)

    with pytest.raises(ValueError):
        _get_init_fn_name(lambda: 0)
    with pytest.raises(ValueError):
        _get_init_fn_name(min)  # Python built-in
    with pytest.raises(ValueError):
        _get_init_fn_name(torch.squeeze)  # From other module
    with pytest.raises(ValueError):
        _get_init_fn_name(calculate_gain)  # Although in init module
    with pytest.raises(ValueError):

        def local_fn():
            pass

        _get_init_fn_name(local_fn)

        del local_fn


def test__default_init_fns():
    seed = random.randint(min_seed(), max_seed())

    torch.manual_seed(seed)
    linear = torch.nn.Linear(10, 10, bias=True)

    weight = torch.Tensor(10, 10)
    bias = torch.Tensor(10)

    torch.manual_seed(seed)
    _default_weight_init_fn(weight)
    _default_bias_init_fn(bias, weight)

    assert (linear.weight == weight).all()
    assert (linear.bias == bias).all()


def test__process_init_fn():
    assert _process_init_fn(None) is None

    for fn_name, fn in _INIT_FUNCTIONS.items():
        assert _process_init_fn(fn_name) is fn
        assert _process_init_fn(fn) is fn

    # User-defined init function
    # TODO: We let this pass here but this needs to be checked elsewhere
    def local_fn():
        pass

    assert _process_init_fn(local_fn) is local_fn

    del local_fn

    with pytest.raises(TypeError):
        _process_init_fn(0)

    with pytest.raises(ValueError):
        _process_init_fn("_")  # Unlikely init function name


def test__process_init_fn_kwargs():
    kwargs = {}
    assert _process_init_fn_kwargs(kwargs) == kwargs
    kwargs = {"key": 0}
    assert _process_init_fn_kwargs(kwargs) == kwargs

    with pytest.raises(TypeError):
        _process_init_fn_kwargs([0, 1, 2])
    with pytest.raises(TypeError):
        _process_init_fn_kwargs(0)
