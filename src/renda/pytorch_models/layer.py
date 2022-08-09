# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from renda.pytorch_models._activation import (
    _is_activation_class,
    _process_activation,
    _process_activation_kwargs,
)
from renda.pytorch_models._init import (
    _default_bias_init_fn,
    _default_weight_init_fn,
    _process_init_fn,
    _process_init_fn_kwargs,
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
        seed: Optional[int] = 0,
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

        if decoder:
            self.weight = Parameter(torch.Tensor(in_features, out_features))
        else:
            self.weight = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        # -----------
        # Activation
        # -----------
        self.activation = _process_activation(activation)
        self.activation_kwargs = _process_activation_kwargs(activation_kwargs)
        if _is_activation_class(self.activation):
            self.activation = self.activation(**self.activation_kwargs)

        # -----
        # Init
        # -----
        self.seed = _process_seed(seed)

        self.weight_init_fn = _process_init_fn(weight_init_fn)
        self.weight_init_fn_kwargs = _process_init_fn_kwargs(weight_init_fn_kwargs)
        self.bias_init_fn = _process_init_fn(bias_init_fn)
        self.bias_init_fn_kwargs = _process_init_fn_kwargs(bias_init_fn_kwargs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with temp_seed(self.seed):
            if self.weight_init_fn is None:
                _default_weight_init_fn(self.weight)
            else:
                self.weight_init_fn(self.weight, **self.weight_init_fn_kwargs)

            if self.bias is not None:
                if self.bias_init_fn is None:
                    _default_bias_init_fn(self.bias, self.weight)
                else:
                    self.bias_init_fn(self.bias, **self.bias_init_fn_kwargs)


class EncoderLayer(_Layer):
    r"""
    Encoder layer for building (deep) autoencoder networks.

    It consists a linear part that mimics the behavior of ``torch.nn.Linear``
    followed by an ``activation`` function.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: If set to ``False``, the layer will not learn an additive bias.
        activation: Can be the name of an activation class (e.g.,
            ``'Sigmoid'``), an activation class (e.g., ``torch.nn.Sigmoid``) or
            an instance of an activation class (e.g., ``torch.nn.Sigmoid()``).
        activation_kwargs: Used to construct the specified ``activation``.
            Ignored if the ``activation`` passed is already an instance of an
            activation class.
        seed: Used to ensure reproducibility when ``weight`` and/or ``bias``
            are initialized randomly. If ``None``, the random initialization
            depends on the state of PyTorch's global random number generator.
        weight_init_fn: Can be the name of an init function (e.g.,
            ``'normal_'``) or an init function (e.g., ``torch.nn.init.normal_``)
            from ``torch.nn.init``, or ``None``. If ``None``, weights are
            initialized as in ``torch.nn.Linear``.
        weight_init_fn_kwargs: If ``weight_init_fn`` is not ``None``, weights
            is initialized via ``weight_init_fn(weight, **weight_init_fn_kwargs)``.
        bias_init_fn: Can be the name of an init function (e.g.,
            ``'normal_'``) or an init function (e.g., ``torch.nn.init.normal_``)
            from ``torch.nn.init``, or ``None``. If ``None``, biases are
            initialized as in ``torch.nn.Linear``.
        bias_init_fn_kwargs: If ``bias_init_fn`` is not ``None``, biases
            is initialized via ``bias_init_fn(bias, **bias_init_fn_kwargs)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
        seed: Optional[int] = 0,
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

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        :meta private:
        """
        X = F.linear(X, self.weight, self.bias)
        X = self.activation(X)
        return X


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

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        :meta private:
        """
        X = F.linear(X, self.weight.t(), self.bias)
        X = self.activation(X)
        return X


class Layer(EncoderLayer):
    """
    Alias for :class:`EncoderLayer` to be used when building (deep) neural
    networks that are not autoencoders.
    """

    pass
