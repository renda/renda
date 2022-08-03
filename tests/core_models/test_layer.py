# -*- coding: utf-8 -*-
import itertools
import random
from collections import OrderedDict

import pytest
import torch

from renda.core_models.layer import DecoderLayer, EncoderLayer, Layer
from renda.utils.seeding import max_seed, min_seed


@pytest.mark.parametrize(
    "layer_class,bias",
    itertools.product([EncoderLayer, Layer], [False, True]),
)
def test_layer_seeding_with_default_init(layer_class, bias):
    seed = random.randint(min_seed(), max_seed())

    # -------
    # Create
    # -------
    torch.manual_seed(seed)  # Set seed before
    a = torch.nn.Linear(10, 50, bias=bias)
    b = layer_class(10, 50, bias=bias, seed=seed)  # Set seed via argument
    c = layer_class(10, 50, bias=bias, seed=seed)  # Double-check reproducibility
    d = layer_class(10, 50, bias=bias, seed=seed + 1)  # Unequally seeded

    assert (a.weight == b.weight).all()
    assert (b.weight == c.weight).all()
    assert not (c.weight == d.weight).all()
    if bias:
        assert (a.bias == b.bias).all()
        assert (b.bias == c.bias).all()
        assert not (c.bias == d.bias).all()
    else:
        assert a.bias is None and b.bias is None
        assert b.bias is None and c.bias is None
        assert c.bias is None and d.bias is None

    # ------
    # Reset
    # ------
    # Remember before reset
    b_weight = b.weight.clone()
    if bias:
        b_bias = b.bias.clone()
    else:
        b_bias = b.bias  # Should be None

    torch.manual_seed(seed)
    a.reset_parameters()
    b.reset_parameters()
    c.reset_parameters()

    # Check reset vs. reset
    assert (a.weight == b.weight).all()
    assert (b.weight == c.weight).all()
    assert not (c.weight == d.weight).all()
    if bias:
        assert (a.bias == b.bias).all()
        assert (b.bias == c.bias).all()
        assert not (c.bias == d.bias).all()
        assert (b.bias == b_bias).all()
    else:
        assert a.bias is None and b.bias is None
        assert b.bias is None and c.bias is None
        assert c.bias is None and d.bias is None
        assert b.bias is None and b_bias is None

    # Check reset vs. remembered
    assert (b.weight == b_weight).all()  # Since no parameters were updated
    if bias:
        assert (b.bias == b_bias).all()
    else:
        assert b.bias is None and b_bias is None


@pytest.mark.parametrize(
    "layer_class,bias",
    itertools.product([DecoderLayer, EncoderLayer, Layer], [False, True]),
)
def test_layer_seeding_with_init_fn_args(layer_class, bias):
    seed = random.randint(min_seed(), max_seed())

    # -------
    # Create
    # -------
    # Set init functions using strings
    a = layer_class(
        in_features=10,
        out_features=50,
        bias=bias,
        seed=seed,
        weight_init_fn="normal_",
        weight_init_fn_kwargs={"mean": 0.0, "std": 1.0},
        bias_init_fn="uniform_",
        bias_init_fn_kwargs={"a": 2.0, "b": 3.0},
    )

    # Set THE SAME init functions using callables
    b = layer_class(
        in_features=10,
        out_features=50,
        bias=bias,
        seed=seed,
        weight_init_fn=torch.nn.init.normal_,
        weight_init_fn_kwargs={"mean": 0.0, "std": 1.0},
        bias_init_fn=torch.nn.init.uniform_,
        bias_init_fn_kwargs={"a": 2.0, "b": 3.0},
    )

    assert (a.weight == b.weight).all()
    if bias:
        assert (a.bias == b.bias).all()
    else:
        assert a.bias is None and b.bias is None

    # ------
    # Reset
    # ------
    # Remember before reset
    b_weight = b.weight.clone()
    if bias:
        b_bias = b.bias.clone()
    else:
        b_bias = b.bias  # Should be None

    a.reset_parameters()
    b.reset_parameters()

    # Check reset vs. reset
    assert (a.weight == b.weight).all()
    if bias:
        assert (a.bias == b.bias).all()
    else:
        assert a.bias is None and b.bias is None

    # Check reset vs. remembered
    assert (b.weight == b_weight).all()  # Since no parameters were updated
    if bias:
        assert (b.bias == b_bias).all()
    else:
        assert b.bias is None and b_bias is None


@pytest.mark.parametrize(
    "bias,tied_weights,seed_offset",
    itertools.product([False, True], [False, True], [0, 1]),
)
def test_tied_and_independent_weights(bias, tied_weights, seed_offset):
    seed = random.randint(min_seed(), max_seed())

    # --------
    # Prepare
    # --------
    model = torch.nn.Sequential(
        OrderedDict(
            [
                ("encoder", EncoderLayer(10, 50, bias=bias, seed=seed)),
                ("decoder", DecoderLayer(50, 10, bias=bias, seed=seed + seed_offset)),
            ]
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()
    target = torch.ones(100, 10)

    if tied_weights:
        model.decoder.weight = model.encoder.weight

    if tied_weights:
        assert (model.decoder.weight == model.encoder.weight).all()
    else:
        # Independent weights
        if seed_offset == 0:
            # Same seed
            assert (model.decoder.weight == model.encoder.weight).all()
        else:
            # Different seed
            assert not (model.decoder.weight == model.encoder.weight).all()

    # -------------
    # Update model
    # -------------
    optimizer.zero_grad()
    output = model(target)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    if tied_weights:
        assert (model.decoder.weight == model.encoder.weight).all()
    else:
        # Independent weights
        assert not (model.decoder.weight == model.encoder.weight).all()


@pytest.mark.parametrize("layer_class", [DecoderLayer, EncoderLayer, Layer])
def test_layer_activation_args(layer_class):
    layer_class(
        in_features=10,
        out_features=50,
        activation="LeakyReLU",
        activation_kwargs={"negative_slope": 0.01},
    )
    layer_class(
        in_features=10,
        out_features=50,
        activation=torch.nn.LeakyReLU,
        activation_kwargs={"negative_slope": 0.01},
    )
    layer_class(
        in_features=10,
        out_features=50,
        activation=torch.nn.LeakyReLU(negative_slope=0.01),
    )

    with pytest.raises(TypeError):
        layer_class(in_features=10, out_features=50, activation=None)
    with pytest.raises(TypeError):
        layer_class(in_features=10, out_features=50, activation=0)
    with pytest.raises(TypeError):
        layer_class(in_features=10, out_features=50, activation_kwargs=[])
    with pytest.raises(TypeError):
        layer_class(in_features=10, out_features=50, activation_kwargs=0)

    with pytest.raises(NotImplementedError):

        def local_fn():
            pass

        layer_class(in_features=10, out_features=50, activation=local_fn)

        del local_fn
