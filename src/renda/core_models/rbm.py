# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import Parameter

from renda.utils.init import (
    process_init_fn,
    process_init_fn_kwargs,
    torch_default_bias_init_fn,
    torch_default_weight_init_fn,
)
from renda.utils.seeding import _process_seed, temp_seed


class RBM(torch.nn.Module):
    def __init__(
        self,
        # -------
        # Linear
        # -------
        in_features: int,
        out_features: int,
        bias: bool = True,
        in_bias: bool = True,
        # ---
        # CD
        # ---
        num_cd_steps: int = 1,
        # ------------
        # Init params
        # ------------
        seed: int = 0,
        weight_init_fn: Union[str, Callable] = "normal_",
        weight_init_fn_kwargs: Dict[str, Any] = {"std": 0.1},
        bias_init_fn: Union[str, Callable] = "zeros_",
        bias_init_fn_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__()

        # -------
        # Linear
        # -------
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if in_bias:
            self.in_bias = Parameter(torch.empty(in_features))
        else:
            self.register_parameter("in_bias", None)

        # ---
        # CD
        # ---
        if not isinstance(num_cd_steps, int) and not num_cd_steps > 0:
            raise ValueError(f"Expected integer > 0. Got {num_cd_steps}.")
        self.num_cd_steps = num_cd_steps

        # -----
        # Init
        # -----
        self.seed = _process_seed(seed)

        self.weight_init_fn = process_init_fn(weight_init_fn)
        self.weight_init_fn_kwargs = process_init_fn_kwargs(weight_init_fn_kwargs)
        self.bias_init_fn = process_init_fn(bias_init_fn)
        self.bias_init_fn_kwargs = process_init_fn_kwargs(bias_init_fn_kwargs)

        self.reset_parameters()

        # -------------------------
        # Used during the training
        # -------------------------
        self._generator = None
        self._loss = None

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

            if self.in_bias is not None:
                if self.bias_init_fn is None:
                    # HINT: Passing weight.t() since it is used when the
                    # transformation direction is "from hidden to visible".
                    torch_default_bias_init_fn(self.in_bias, self.weight.t())
                else:
                    self.bias_init_fn(self.in_bias, **self.bias_init_fn_kwargs)

    def forward(self, X, *args, **kwargs):
        self._loss, Z, R = _CD.apply(
            X,
            self.weight,
            self.bias,
            self.in_bias,
            self.num_cd_steps,
            self._generator,
        )

        return Z.detach(), R.detach()

    def loss(self):
        return self._loss

    def model_based_setup(self, model: LightningModule):
        self._generator = torch.Generator(device=model.device)
        self._generator.manual_seed(self.seed)

    def model_based_teardown(self, model: LightningModule):
        self._generator = None


class _CD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, in_bias, num_cd_steps, generator):
        # ---------------
        # Positive phase
        # ---------------
        pos_vis_probs = X
        pos_hid_probs = _vis_to_hid(pos_vis_probs, weight, bias)
        pos_hid_states = _sample_hid(pos_hid_probs, generator)

        # ---------------
        # Negative phase
        # ---------------
        neg_hid_states = pos_hid_states
        for _ in range(num_cd_steps):
            neg_vis_probs = _hid_to_vis(neg_hid_states, weight, in_bias)
            neg_hid_probs = _vis_to_hid(neg_vis_probs, weight, bias)
            neg_hid_states = _sample_hid(neg_hid_probs, generator)

        # --------------------------------------------
        # For gradient calculation in backward method
        # --------------------------------------------
        ctx.save_for_backward(
            weight,
            bias,
            in_bias,
            pos_vis_probs,
            pos_hid_probs,
            neg_vis_probs,
            neg_hid_probs,
        )

        loss = (pos_vis_probs - neg_vis_probs).pow(2).mean().sqrt()
        Z = pos_hid_probs  # Features
        R = neg_vis_probs  # Reconstructions

        return loss, Z, R

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            weight,
            bias,
            in_bias,
            pos_vis_probs,
            pos_hid_probs,
            neg_vis_probs,
            neg_hid_probs,
        ) = ctx.saved_tensors

        num_cases = pos_vis_probs.shape[0]

        # Weight gradient
        pos_hid_vis = pos_hid_probs.t() @ pos_vis_probs
        neg_hid_vis = neg_hid_probs.t() @ neg_vis_probs
        d_weight = -((pos_hid_vis - neg_hid_vis) / num_cases - 0.0002 * weight)

        # Bias gradient
        if bias is None:
            d_bias = None
        else:
            pos_hid = pos_hid_probs.sum(dim=0)
            neg_hid = neg_hid_probs.sum(dim=0)
            d_bias = -(pos_hid - neg_hid) / num_cases

        # Input bias gradient
        if in_bias is None:
            d_in_bias = None
        else:
            pos_vis = pos_vis_probs.sum(dim=0)
            neg_vis = neg_vis_probs.sum(dim=0)
            d_in_bias = -(pos_vis - neg_vis) / num_cases

        return None, d_weight, d_bias, d_in_bias, None, None


def _vis_to_hid(vis, weight, bias):
    hid = F.linear(vis, weight, bias)
    hid = F.sigmoid(hid)
    return hid


def _hid_to_vis(hid, weight, bias):
    vis = F.linear(hid, weight.t(), bias)
    vis = F.sigmoid(vis)
    return vis


def _sample_hid(P, generator):
    return torch.bernoulli(P, generator=generator)
