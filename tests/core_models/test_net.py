# -*- coding: utf-8 -*-
import itertools
import random

import pytest
import torch

from renda.core_models.net import (
    DecoderNet,
    EncoderNet,
    Net,
    process_bias,
    process_topology,
)
from renda.utils.activation import get_name_of_activation
from renda.utils.seeding import max_seed, min_seed


def test_process_topology():
    topology = [2, 20, 10, 1]
    same_topology = process_topology(topology)
    assert same_topology is topology

    topology = [2, 20]
    same_topology = process_topology(topology)
    assert same_topology is topology

    with pytest.raises(ValueError):
        process_topology(2)
    with pytest.raises(ValueError):
        process_topology([2])
    with pytest.raises(ValueError):
        process_topology(["Se", "qu", "en", "ce"])
    with pytest.raises(ValueError):
        process_topology([0, 20])


def test_process_bias():
    assert process_bias(False) is False
    assert process_bias(True) is True

    with pytest.raises(TypeError):
        process_bias(0)


@pytest.mark.parametrize(
    "net_class,bias,last_bias",
    itertools.product([DecoderNet, EncoderNet, Net], [False, True], [False, True]),
)
def test_net_seeding_with_default_init(net_class, bias, last_bias):
    seed = random.randint(min_seed(), max_seed())

    a = net_class(topology=[2, 20, 10, 1], bias=bias, last_bias=last_bias, seed=seed)
    b = net_class(topology=[2, 20, 10, 1], bias=bias, last_bias=last_bias, seed=seed)
    c = net_class(
        topology=[2, 20, 10, 1], bias=bias, last_bias=last_bias, seed=seed + 1
    )

    for a_, b_, c_ in zip(a.layer, b.layer, c.layer):
        assert (a_.weight == b_.weight).all()
        assert not (b_.weight == c_.weight).all()
        if a_.bias is not None:
            assert (a_.bias == b_.bias).all()
            assert not (b_.bias == c_.bias).all()
        else:
            assert b_.bias is None
            assert c_.bias is None

    T = torch.rand(10, a.topology[0])
    assert (a(T) == b(T)).all()
    assert not (b(T) == c(T)).all()


@pytest.mark.parametrize(
    "net_class,bias,last_bias",
    itertools.product([DecoderNet, EncoderNet, Net], [False, True], [False, True]),
)
def test_net_seeding_with_init_fn_args(net_class, bias, last_bias):
    seed = random.randint(min_seed(), max_seed())

    # Set init functions using strings
    a = net_class(
        topology=[2, 20, 10, 1],
        bias=bias,
        last_bias=last_bias,
        seed=seed,
        weight_init_fn="normal_",
        weight_init_fn_kwargs={"mean": 0.0, "std": 1.0},
        bias_init_fn="uniform_",
        bias_init_fn_kwargs={"a": 2.0, "b": 3.0},
    )

    # Set THE SAME init functions using callables
    b = net_class(
        topology=[2, 20, 10, 1],
        bias=bias,
        last_bias=last_bias,
        seed=seed,
        weight_init_fn=torch.nn.init.normal_,
        weight_init_fn_kwargs={"mean": 0.0, "std": 1.0},
        bias_init_fn=torch.nn.init.uniform_,
        bias_init_fn_kwargs={"a": 2.0, "b": 3.0},
    )

    for a_, b_ in zip(a.layer, b.layer):
        assert (a_.weight == b_.weight).all()
        if a_.bias is not None:
            assert (a_.bias == b_.bias).all()
        else:
            assert b_.bias is None

    T = torch.rand(10, a.topology[0])
    assert (a(T) == b(T)).all()


@pytest.mark.parametrize(
    "net_class,activation,last_activation",
    itertools.product(
        [DecoderNet, EncoderNet, Net], ["Tanh", "LeakyReLU"], ["Sigmoid", "Threshold"]
    ),
)
def test_net_activation_args(net_class, activation, last_activation):
    activation_kwargs = {
        "Tanh": {},
        "LeakyReLU": {"negative_slope": 0.01},
    }[activation]

    last_activation_kwargs = {
        "Sigmoid": {},
        "Threshold": {"threshold": 0.0, "value": 0.0},
    }[last_activation]

    net = net_class(
        topology=[2, 20, 10, 1],
        activation=activation,
        activation_kwargs=activation_kwargs,
        last_activation=last_activation,
        last_activation_kwargs=last_activation_kwargs,
    )

    for ii, layer in enumerate(net.layer):
        if ii < net.num_layers - 1:
            current_activation = activation
            current_activation_kwargs = activation_kwargs
        else:
            current_activation = last_activation
            current_activation_kwargs = last_activation_kwargs

        assert get_name_of_activation(layer.activation) == current_activation

        for k, v in current_activation_kwargs.items():
            assert getattr(layer.activation, k) == v
