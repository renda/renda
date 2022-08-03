# -*- coding: utf-8 -*-
import random

import pytest
import torch
from torch.nn.init import calculate_gain

from renda.utils.init import (
    get_name_of_torch_nn_init_fn,
    get_torch_nn_init_fn_by_name,
    is_torch_nn_init_fn,
    is_torch_nn_init_fn_name,
    process_init_fn,
    process_init_fn_kwargs,
    torch_default_bias_init_fn,
    torch_default_weight_init_fn,
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


def test_is_torch_nn_init_fn_name():
    for name in _INIT_FUNCTIONS.keys():
        assert is_torch_nn_init_fn_name(name)

    assert not is_torch_nn_init_fn_name("_")  # Unlikely init function name
    assert not is_torch_nn_init_fn_name("calculate_gain")  # Although in init module

    with pytest.raises(TypeError):
        is_torch_nn_init_fn_name(torch.nn.init.uniform_)
    with pytest.raises(TypeError):
        is_torch_nn_init_fn_name(0)


def test_is_torch_nn_init_fn():
    for fn in _INIT_FUNCTIONS.values():
        assert is_torch_nn_init_fn(fn)

    def local_fn():
        pass

    assert not is_torch_nn_init_fn(local_fn)
    assert not is_torch_nn_init_fn(lambda: 0)
    assert not is_torch_nn_init_fn(min)  # Python built-in
    assert not is_torch_nn_init_fn(torch.squeeze)  # From other module
    assert not is_torch_nn_init_fn(calculate_gain)  # Although in init module

    del local_fn

    with pytest.raises(TypeError):
        is_torch_nn_init_fn("uniform_")
    with pytest.raises(TypeError):
        is_torch_nn_init_fn(0)


def test_get_name_of_torch_nn_init_fn():
    for fn_name, fn in _INIT_FUNCTIONS.items():
        assert get_name_of_torch_nn_init_fn(fn) == fn_name

    with pytest.raises(TypeError):
        get_name_of_torch_nn_init_fn("uniform_")
    with pytest.raises(TypeError):
        get_name_of_torch_nn_init_fn(0)

    with pytest.raises(ValueError):
        get_name_of_torch_nn_init_fn(lambda: 0)
    with pytest.raises(ValueError):
        get_name_of_torch_nn_init_fn(min)  # Python built-in
    with pytest.raises(ValueError):
        get_name_of_torch_nn_init_fn(torch.squeeze)  # From other module
    with pytest.raises(ValueError):
        get_name_of_torch_nn_init_fn(calculate_gain)  # Although in init module
    with pytest.raises(ValueError):

        def local_fn():
            pass

        get_name_of_torch_nn_init_fn(local_fn)

        del local_fn


def test_get_torch_nn_init_fn_by_name():
    for fn_name, fn in _INIT_FUNCTIONS.items():
        assert get_torch_nn_init_fn_by_name(fn_name) is fn

    with pytest.raises(TypeError):
        get_torch_nn_init_fn_by_name(torch.nn.init.uniform_)
    with pytest.raises(TypeError):
        get_torch_nn_init_fn_by_name(0)

    with pytest.raises(ValueError):
        get_torch_nn_init_fn_by_name("_")  # Unlikely init function name
    with pytest.raises(ValueError):
        get_torch_nn_init_fn_by_name("calculate_gain")  # Although in init module


def test_torch_default_init_fns():
    seed = random.randint(min_seed(), max_seed())

    torch.manual_seed(seed)
    linear = torch.nn.Linear(10, 10, bias=True)

    torch.manual_seed(seed)
    weight = torch.Tensor(10, 10)
    bias = torch.Tensor(10)

    torch_default_weight_init_fn(weight)
    torch_default_bias_init_fn(bias, weight)

    assert (linear.weight == weight).all()
    assert (linear.bias == bias).all()


def test_process_init_fn():
    for fn_name, fn in _INIT_FUNCTIONS.items():
        assert process_init_fn(fn_name) is fn
        assert process_init_fn(fn) is fn

    def local_fn():
        pass

    assert process_init_fn(local_fn) is local_fn  # Assume user-defined init function
    assert process_init_fn(None) is None

    with pytest.raises(TypeError):
        process_init_fn(0)

    with pytest.raises(ValueError):
        process_init_fn("_")  # Unlikely init function name


def test_process_init_fn_kwargs():
    kwargs = {}
    assert process_init_fn_kwargs(kwargs) == kwargs
    kwargs = {"key": 0}
    assert process_init_fn_kwargs(kwargs) == kwargs

    with pytest.raises(TypeError):
        process_init_fn_kwargs([0, 1, 2])
    with pytest.raises(TypeError):
        process_init_fn_kwargs(0)
