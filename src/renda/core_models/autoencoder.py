# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch

from renda.core_models.net import DecoderNet, EncoderNet, process_bias, process_topology
from renda.utils.activation import process_activation, process_activation_kwargs
from renda.utils.init import process_init_fn, process_init_fn_kwargs
from renda.utils.seeding import _process_seed


class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        # ----
        # Net
        # ----
        topology: Sequence[int] = None,
        bias: bool = True,
        activation: Any = "Sigmoid",
        activation_kwargs: Dict = {},
        last_bias: bool = False,
        last_activation: Any = "Identity",
        last_activation_kwargs: Dict = {},
        last_decoder_bias: bool = False,
        last_decoder_activation: Any = "Identity",
        last_decoder_activation_kwargs: Dict = {},
        tied_weights: bool = True,
        tied_biases: bool = True,
        # -----
        # Init
        # -----
        seed: int = 0,
        weight_init_fn: Optional[Union[str, Callable]] = None,
        weight_init_fn_kwargs: Dict[str, Any] = {},
        bias_init_fn: Optional[Union[str, Callable]] = None,
        bias_init_fn_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__()

        self.topology = process_topology(topology)
        self.num_layers = len(self.topology) - 1
        self.bias = process_bias(bias)
        self.activation_kwargs = process_activation_kwargs(activation_kwargs)
        self.activation = process_activation(activation)
        self.last_bias = process_bias(last_bias)
        self.last_activation_kwargs = process_activation_kwargs(last_activation_kwargs)
        self.last_activation = process_activation(last_activation)
        self.last_decoder_bias = process_bias(last_decoder_bias)
        self.last_decoder_activation_kwargs = process_activation_kwargs(
            last_decoder_activation_kwargs
        )
        self.last_decoder_activation = process_activation(last_decoder_activation)

        seed = _process_seed(seed, allow_sequence=True, length=2 * self.num_layers)
        if seed is None:  # pragma: no cover
            # No explicit seeding
            self.seed = [None, None]
            self._seed = self.seed.copy()
        elif isinstance(seed, int):
            # Default seeding scheme
            self.seed = [seed, seed + int(1e6)]
            self._seed = self.seed.copy()
        else:
            # Custom seeding scheme
            # There already is a seed per layer
            self.seed = seed.copy()
            self._seed = [
                seed.copy()[: len(seed) // 2 :],  # Encoder gets first half
                seed.copy()[len(seed) // 2 : :],  # Decoder gets second half
            ]

        self.weight_init_fn = process_init_fn(weight_init_fn)
        self.weight_init_fn_kwargs = process_init_fn_kwargs(weight_init_fn_kwargs)
        self.bias_init_fn = process_init_fn(bias_init_fn)
        self.bias_init_fn_kwargs = process_init_fn_kwargs(bias_init_fn_kwargs)

        self.encoder = EncoderNet(
            topology=self.topology,
            bias=self.bias,
            activation=self.activation,
            activation_kwargs=self.activation_kwargs,
            last_bias=self.last_bias,
            last_activation=self.last_activation,
            last_activation_kwargs=self.last_activation_kwargs,
            seed=self._seed[0],  # Use internal representation
            weight_init_fn=self.weight_init_fn,
            weight_init_fn_kwargs=self.weight_init_fn_kwargs,
            bias_init_fn=self.bias_init_fn,
            bias_init_fn_kwargs=self.bias_init_fn_kwargs,
        )

        self.decoder = DecoderNet(
            topology=self.topology[::-1],  # Inverse encoder topology
            bias=self.bias,
            activation=self.activation,
            activation_kwargs=self.activation_kwargs,
            last_bias=self.last_decoder_bias,
            last_activation=self.last_decoder_activation,
            last_activation_kwargs=self.last_decoder_activation_kwargs,
            seed=self._seed[1],  # Use internal representation
            weight_init_fn=self.weight_init_fn,
            weight_init_fn_kwargs=self.weight_init_fn_kwargs,
            bias_init_fn=self.bias_init_fn,
            bias_init_fn_kwargs=self.bias_init_fn_kwargs,
        )

        def _process_bool(bool_: bool) -> bool:  # pragma: no cover
            if not isinstance(bool_, bool):
                raise TypeError(f"Expected bool. Got {bool_}.")

            return bool_

        self.tied_weights = _process_bool(tied_weights)
        self.tied_biases = _process_bool(tied_biases)

        if self.tied_weights or self.tied_biases:
            encoder_layer = self.encoder.layer
            decoder_layer = self.decoder.layer[::-1]

        if self.tied_weights:
            for e, d in zip(encoder_layer, decoder_layer):
                d.weight = e.weight

        if self.tied_biases:
            for e, d in zip(encoder_layer[:-1:], decoder_layer[1::]):
                if e.bias is not None and d.bias is not None:
                    d.bias = e.bias

    def forward(self, X, *args, **kwargs):
        Z = self.encoder(X)
        R = self.decoder(Z)
        return Z, R
