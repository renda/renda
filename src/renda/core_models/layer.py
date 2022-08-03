# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from renda.utils.activation import (
    is_activation_class,
    process_activation,
    process_activation_kwargs,
)
from renda.utils.init import (
    process_init_fn,
    process_init_fn_kwargs,
    torch_default_bias_init_fn,
    torch_default_weight_init_fn,
)
from renda.utils.seeding import _process_seed, temp_seed


class _Layer(torch.nn.Module):
    def __init__(
        self,
        decoder: bool,
        # -------
        # Linear
        # -------
        in_features: int,
        out_features: int,
        bias: bool = True,
        # -----------
        # Activation
        # -----------
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
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

        # -------
        # Linear
        # -------
        self.in_features = in_features
        self.out_features = out_features

        if not decoder:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        # -----------
        # Activation
        # -----------
        self.activation = process_activation(activation)
        self.activation_kwargs = process_activation_kwargs(activation_kwargs)
        if is_activation_class(self.activation):
            self.activation = self.activation(**self.activation_kwargs)

        # -----
        # Init
        # -----
        self.seed = _process_seed(seed)

        self.weight_init_fn = process_init_fn(weight_init_fn)
        self.weight_init_fn_kwargs = process_init_fn_kwargs(weight_init_fn_kwargs)
        self.bias_init_fn = process_init_fn(bias_init_fn)
        self.bias_init_fn_kwargs = process_init_fn_kwargs(bias_init_fn_kwargs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with temp_seed(self.seed):
            if self.weight_init_fn is None:
                torch_default_weight_init_fn(self.weight)
            else:
                self.weight_init_fn(self.weight, **self.weight_init_fn_kwargs)

            if self.bias is not None:
                if self.bias_init_fn is None:
                    torch_default_bias_init_fn(self.bias, self.weight)
                else:
                    self.bias_init_fn(self.bias, **self.bias_init_fn_kwargs)


class Layer(_Layer):
    def __init__(
        self,
        # -------
        # Linear
        # -------
        in_features: int,
        out_features: int,
        bias: bool = True,
        # -----------
        # Activation
        # -----------
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
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
            decoder=False,
            # -------
            # Linear
            # -------
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            # -----------
            # Activation
            # -----------
            activation=activation,
            activation_kwargs=activation_kwargs,
            # -----
            # Init
            # -----
            seed=seed,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            bias_init_fn=bias_init_fn,
            bias_init_fn_kwargs=bias_init_fn_kwargs,
        )

    def forward(self, T, *args, **kwargs):
        T = F.linear(T, self.weight, self.bias)
        T = self.activation(T)
        return T


class EncoderLayer(Layer):
    pass


class DecoderLayer(_Layer):
    def __init__(
        self,
        # -------
        # Linear
        # -------
        in_features: int,
        out_features: int,
        bias: bool = True,
        # -----------
        # Activation
        # -----------
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
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
            decoder=True,
            # -------
            # Linear
            # -------
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            # -----------
            # Activation
            # -----------
            activation=activation,
            activation_kwargs=activation_kwargs,
            # -----
            # Init
            # -----
            seed=seed,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            bias_init_fn=bias_init_fn,
            bias_init_fn_kwargs=bias_init_fn_kwargs,
        )

    def forward(self, T, *args, **kwargs):
        T = F.linear(T, self.weight.t(), self.bias)
        T = self.activation(T)
        return T
