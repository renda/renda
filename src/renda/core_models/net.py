# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch

from renda.core_models.layer import DecoderLayer, EncoderLayer, Layer, _Layer
from renda.utils.activation import process_activation, process_activation_kwargs
from renda.utils.init import process_init_fn, process_init_fn_kwargs
from renda.utils.seeding import _fix_seed, _process_seed


def process_topology(topology: Sequence[int]) -> Sequence[int]:
    if not (
        isinstance(topology, Sequence)
        and len(topology) >= 2
        and all(isinstance(d, int) for d in topology)
        and all(d > 0 for d in topology)
    ):
        raise ValueError(
            f"Expected sequence of at least two strictly positive integers. "
            f"Got {topology}."
        )

    return topology


def process_bias(bool_: bool) -> bool:
    if not isinstance(bool_, bool):
        raise TypeError(f"Expected bool. Got {bool_}.")

    return bool_


class _Net(torch.nn.Module):
    def __init__(
        self,
        layer_class: _Layer,
        # ----
        # Net
        # ----
        topology: Sequence[int],
        bias: bool = True,
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
        last_bias: bool = False,
        last_activation: Union[str, Callable] = "Identity",
        last_activation_kwargs: Dict[str, Any] = {},
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

        if not issubclass(layer_class, _Layer):  # pragma: no cover
            raise TypeError(
                f"Expected subclass of '_Layer'. Got {layer_class} with base "
                f"classes {', '.join(layer_class.__bases__)}."
            )

        # --------------
        # Net (hparams)
        # --------------
        self.topology = process_topology(topology)
        self.num_layers = len(self.topology) - 1
        self.bias = process_bias(bias)
        self.activation_kwargs = process_activation_kwargs(activation_kwargs)
        self.activation = process_activation(activation)
        self.last_bias = process_bias(last_bias)
        self.last_activation_kwargs = process_activation_kwargs(last_activation_kwargs)
        self.last_activation = process_activation(last_activation)

        # ---------------
        # Init (hparams)
        # ---------------
        # Ensure seed per layer
        seed = _process_seed(seed, allow_sequence=True, length=self.num_layers)
        if seed is None:  # pragma: no cover
            # No explicit seeding
            seed = [seed] * self.num_layers
        elif isinstance(seed, int):
            # Default seeding scheme
            self.seed = []
            for ii in range(self.num_layers):
                seed_ = seed + ii * 1000
                seed_ = _fix_seed(seed_)  # Since seed_ could be too large
                self.seed.append(seed_)
        else:
            # Custom seeding scheme
            # There already is a seed per layer
            self.seed = seed

        self.weight_init_fn = process_init_fn(weight_init_fn)
        self.weight_init_fn_kwargs = process_init_fn_kwargs(weight_init_fn_kwargs)
        self.bias_init_fn = process_init_fn(bias_init_fn)
        self.bias_init_fn_kwargs = process_init_fn_kwargs(bias_init_fn_kwargs)

        # --------------------------------
        # Net (construction from hparams)
        # --------------------------------
        self.layer = torch.nn.ModuleList()

        bias = (self.bias,) * (self.num_layers - 1)
        bias += (self.last_bias,)
        activation = (self.activation,) * (self.num_layers - 1)
        activation += (self.last_activation,)
        activation_kwargs = (self.activation_kwargs,) * (self.num_layers - 1)
        activation_kwargs += (self.last_activation_kwargs,)

        for ii in range(self.num_layers):
            self.layer.append(
                layer_class(
                    in_features=self.topology[ii],
                    out_features=self.topology[ii + 1],
                    bias=bias[ii],
                    activation=activation[ii],
                    activation_kwargs=activation_kwargs[ii],
                    seed=self.seed[ii],
                    weight_init_fn=self.weight_init_fn,
                    weight_init_fn_kwargs=self.weight_init_fn_kwargs,
                    bias_init_fn=self.bias_init_fn,
                    bias_init_fn_kwargs=self.bias_init_fn_kwargs,
                )
            )

    def forward(self, T: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for layer in self.layer:
            T = layer(T)
        return T


class Net(_Net):
    def __init__(
        self,
        # ----
        # Net
        # ----
        topology: Sequence[int],
        bias: bool = True,
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
        last_bias: bool = False,
        last_activation: Union[str, Callable] = "Identity",
        last_activation_kwargs: Dict[str, Any] = {},
        # -----
        # Init
        # -----
        seed: int = 0,
        weight_init_fn: Optional[Union[str, Callable]] = None,
        weight_init_fn_kwargs: Dict[str, Any] = {},
        bias_init_fn: Optional[Union[str, Callable]] = None,
        bias_init_fn_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            layer_class=Layer,
            # ----
            # Net
            # ----
            topology=topology,
            bias=bias,
            activation=activation,
            activation_kwargs=activation_kwargs,
            last_bias=last_bias,
            last_activation=last_activation,
            last_activation_kwargs=last_activation_kwargs,
            # -----
            # Init
            # -----
            seed=seed,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            bias_init_fn=bias_init_fn,
            bias_init_fn_kwargs=bias_init_fn_kwargs,
        )


class EncoderNet(_Net):
    def __init__(
        self,
        # ----
        # Net
        # ----
        topology: Sequence[int],
        bias: bool = True,
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
        last_bias: bool = False,
        last_activation: Union[str, Callable] = "Identity",
        last_activation_kwargs: Dict[str, Any] = {},
        # -----
        # Init
        # -----
        seed: int = 0,
        weight_init_fn: Optional[Union[str, Callable]] = None,
        weight_init_fn_kwargs: Dict[str, Any] = {},
        bias_init_fn: Optional[Union[str, Callable]] = None,
        bias_init_fn_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            layer_class=EncoderLayer,
            # ----
            # Net
            # ----
            topology=topology,
            bias=bias,
            activation=activation,
            activation_kwargs=activation_kwargs,
            last_bias=last_bias,
            last_activation=last_activation,
            last_activation_kwargs=last_activation_kwargs,
            # -----
            # Init
            # -----
            seed=seed,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            bias_init_fn=bias_init_fn,
            bias_init_fn_kwargs=bias_init_fn_kwargs,
        )


class DecoderNet(_Net):
    def __init__(
        self,
        # ----
        # Net
        # ----
        topology: Sequence[int],
        bias: bool = True,
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
        last_bias: bool = False,
        last_activation: Union[str, Callable] = "Identity",
        last_activation_kwargs: Dict[str, Any] = {},
        # -----
        # Init
        # -----
        seed: int = 0,
        weight_init_fn: Optional[Union[str, Callable]] = None,
        weight_init_fn_kwargs: Dict[str, Any] = {},
        bias_init_fn: Optional[Union[str, Callable]] = None,
        bias_init_fn_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            layer_class=DecoderLayer,
            # ----
            # Net
            # ----
            topology=topology,
            bias=bias,
            activation=activation,
            activation_kwargs=activation_kwargs,
            last_bias=last_bias,
            last_activation=last_activation,
            last_activation_kwargs=last_activation_kwargs,
            # -----
            # Init
            # -----
            seed=seed,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            bias_init_fn=bias_init_fn,
            bias_init_fn_kwargs=bias_init_fn_kwargs,
        )
