# -*- coding: utf-8 -*-
import itertools
import random

import pytest
import torch

from renda.core_models.autoencoder import Autoencoder
from renda.utils.activation import get_name_of_activation
from renda.utils.seeding import max_seed, min_seed


@pytest.mark.parametrize(
    ",".join(
        [
            "bias",
            "last_bias",
            "last_decoder_bias",
            "tied_weights",
            "tied_biases",
            "same_init_but_independent_update",
        ]
    ),
    itertools.product([False, True], repeat=6),
)
def test_tied_and_independent_parameters(
    bias,
    last_bias,
    last_decoder_bias,
    tied_weights,
    tied_biases,
    same_init_but_independent_update,
):
    topology = [2, 20, 10, 1]

    if same_init_but_independent_update:
        # Collect seeds for encoder
        seed = []
        num_layers = len(topology) - 1
        for _ in range(num_layers):
            seed.append(random.randint(min_seed(), max_seed()))

        # Use inverse seed order for decoder
        seed += seed[::-1]
    else:
        seed = random.randint(min_seed(), max_seed())

    # --------
    # Prepare
    # --------
    model = Autoencoder(
        topology=[2, 20, 10, 1],
        bias=bias,
        last_bias=last_bias,
        last_decoder_bias=last_decoder_bias,
        tied_weights=tied_weights,
        tied_biases=tied_biases,
        seed=seed,
        bias_init_fn="constant_",
        bias_init_fn_kwargs={"val": 0.01},
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()
    targets = torch.ones(100, 2)

    for e, d in zip(model.encoder.layer, model.decoder.layer[::-1]):
        if tied_weights:
            assert (d.weight == e.weight).all()
        else:
            if same_init_but_independent_update:
                # Same init
                assert (d.weight == e.weight).all()
            else:
                # Different init
                assert not (d.weight == e.weight).all()

    for e, d in zip(model.encoder.layer[:-1:], model.decoder.layer[-2::-1]):
        if d.bias is not None and e.bias is not None:
            if tied_biases:
                assert (d.bias == e.bias).all()
            else:
                assert (d.bias == e.bias).all()

    # -------------
    # Update model
    # -------------
    optimizer.zero_grad()
    features, reconstructions = model(targets)
    loss = loss_fn(reconstructions, targets)
    loss.backward()
    optimizer.step()

    for e, d in zip(model.encoder.layer, model.decoder.layer[::-1]):
        if tied_weights:
            assert (d.weight == e.weight).all()
        else:
            # Independent weights
            assert not (d.weight == e.weight).all()

    for e, d in zip(model.encoder.layer[:-1:], model.decoder.layer[-2::-1]):
        if d.bias is not None and e.bias is not None:
            if tied_biases:
                assert (d.bias == e.bias).all()
            else:
                assert not (d.bias == e.bias).all()


@pytest.mark.parametrize(
    "activation,last_activation,last_decoder_activation",
    itertools.product(["Tanh", "LeakyReLU"], ["Sigmoid", "Threshold"], ["ReLU", "ELU"]),
)
def test_net_activation_args(activation, last_activation, last_decoder_activation):
    activation_kwargs = {
        "Tanh": {},
        "LeakyReLU": {"negative_slope": 0.01},
    }[activation]

    last_activation_kwargs = {
        "Sigmoid": {},
        "Threshold": {"threshold": 0.0, "value": 0.0},
    }[last_activation]

    last_decoder_activation_kwargs = {
        "ReLU": {},
        "ELU": {"alpha": 1.0},
    }[last_decoder_activation]

    model = Autoencoder(
        topology=[2, 20, 10, 1],
        activation=activation,
        activation_kwargs=activation_kwargs,
        last_activation=last_activation,
        last_activation_kwargs=last_activation_kwargs,
        last_decoder_activation=last_decoder_activation,
        last_decoder_activation_kwargs=last_decoder_activation_kwargs,
    )

    for ii, layer in enumerate(model.encoder.layer):
        if ii < model.num_layers - 1:
            current_activation = activation
            current_activation_kwargs = activation_kwargs
        else:
            current_activation = last_activation
            current_activation_kwargs = last_activation_kwargs

        assert get_name_of_activation(layer.activation) == current_activation

        for k, v in current_activation_kwargs.items():
            assert getattr(layer.activation, k) == v

    for ii, layer in enumerate(model.decoder.layer):
        if ii < model.num_layers - 1:
            current_activation = activation
            current_activation_kwargs = activation_kwargs
        else:
            current_activation = last_decoder_activation
            current_activation_kwargs = last_decoder_activation_kwargs

        assert get_name_of_activation(layer.activation) == current_activation

        for k, v in current_activation_kwargs.items():
            assert getattr(layer.activation, k) == v
