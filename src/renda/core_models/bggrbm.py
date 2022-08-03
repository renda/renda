# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import Parameter, UninitializedParameter

from renda.core_models.rbm import RBM
from renda.utils.init import torch_default_bias_init_fn, torch_default_weight_init_fn
from renda.utils.seeding import temp_seed


class BGGRBM(RBM):
    def __init__(
        self,
        # -------
        # Linear
        # -------
        in_features: int,
        out_features: int,
        bias: bool = True,
        in_bias: bool = True,
        extra_bias: bool = True,
        # --------
        # Special
        # --------
        lambda_: Union[float, int] = 0.5,
        # ---
        # CD
        # ---
        num_cd_steps: int = 1,
        # -----
        # Init
        # -----
        seed: int = 0,
        weight_init_fn: Union[str, Callable] = "normal_",
        weight_init_fn_kwargs: Dict[str, Any] = {"std": 0.1},
        bias_init_fn: Union[str, Callable] = "zeros_",
        bias_init_fn_kwargs: Dict[str, Any] = {},
        # ------------------------------------------------------
        # Only used by PyTorch Lightning's load_from_checkpoint
        # ------------------------------------------------------
        _extra_features: Optional[int] = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            in_bias=in_bias,
            num_cd_steps=num_cd_steps,
            seed=seed,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            bias_init_fn=bias_init_fn,
            bias_init_fn_kwargs=bias_init_fn_kwargs,
        )

        # HINT concerning sigma:
        #
        # To make this work for PyTorch Lightning's load_from_checkpoint,
        # we need to initialize the buffer with a tensor that already has the
        # right shape. Only then, load_from_checkpoint can properly restore
        # the buffers value from the saved state_dict.
        #
        # It is worth noting that buffers initialized with None are not
        # included in a module's state_dict, but they will as soon as a tensor
        # is assigned. Using this mechanism is not suited for making
        # load_from_checkpoint work: The initial state_dict would NOT include
        # a key for the buffer. Thus, a value stored for this buffer cannot be
        # assigned via load_state_dict().
        #
        # sigma will be determined and set in model_based_setup(). -1.0 is
        # easy to recognize as a placeholder since sigma will always be set to
        # a non-negative value.

        self.register_buffer("sigma", torch.tensor(-1.0))

        if _extra_features is None:
            # ---------------------------------
            # Finalized in model_based_setup()
            # ---------------------------------
            self.extra_features = None
            self.extra_weight = UninitializedParameter()
            if extra_bias:
                self.extra_bias = UninitializedParameter()
            else:
                self.register_parameter("extra_bias", None)

            self.register_buffer("target_codes", None)
        else:
            # ------------------------------------------------------
            # Finalized by PyTorch Lightning's load_from_checkpoint
            # ------------------------------------------------------

            # HINT concerning extra parameters / target_codes buffer:
            #
            # As in the case of sigma, we need to initialize parameters and
            # buffers with tensors that already have the right shape. Only
            # then, load_from_checkpoint can properly restore parameters and
            # buffers value from the saved state_dict.
            #
            # Since all shapes set here depend on the data, we need to ensure
            # that the hyperparameter _extra_features is updated in
            # model_based_setup(). Then, THIS instantiation branch will be
            # entered when load_from_checkpoint is called and there should not
            # occur any shape mismatches when loading the state_dict.

            self.extra_features = _extra_features
            self.extra_weight = Parameter(
                torch.empty(self.out_features, _extra_features)
            )
            if extra_bias:
                self.extra_bias = Parameter(torch.empty(_extra_features))
            else:
                self.register_parameter("extra_bias", None)

            self.register_buffer(
                "target_codes", torch.empty(_extra_features, _extra_features)
            )

        if not (isinstance(lambda_, (float, int)) and 0.0 <= lambda_ <= 1.0):
            raise ValueError(f"Expected real number between 0 and 1. Got {lambda_}.")
        else:
            self.register_buffer("lambda_", torch.tensor(lambda_))

        self._supervised_loss = None
        self._unsupervised_loss = None

    def forward(self, X, Y, *args, **kwargs):
        T = self.target_codes[Y.long().squeeze(), :]
        (
            self._loss,
            self._supervised_loss,
            self._unsupervised_loss,
            Z,
            RX,
            RT,
        ) = _CD.apply(
            X,
            T,
            self.weight,
            self.extra_weight,
            self.bias,
            self.in_bias,
            self.extra_bias,
            self.num_cd_steps,
            self._generator,
            self.sigma,
            self.lambda_,
        )

        return Z.detach(), RX.detach(), T.detach(), RT.detach()

    def loss(self):
        return self._loss, self._supervised_loss, self._unsupervised_loss

    def model_based_setup(self, model: LightningModule):
        self._generator = torch.Generator(device=model.device)
        self._generator.manual_seed(self.seed)

        sequential_train_dataloader = model.trainer.get_sequential_train_dataloader()

        # ------------------
        # Gather labels (Y)
        # ------------------
        Y = []
        for batch in sequential_train_dataloader:
            if isinstance(batch, Sequence) and len(batch) >= 2:
                Y.append(batch[1])
            else:
                raise RuntimeError(
                    "Could not retrieve labels (Y). Please make sure that your"
                    "dataset class provides labels as its second return value."
                )
        Y = torch.cat(Y)

        # ---------------------
        # Compute target codes
        # ---------------------
        Y = Y.long().squeeze()
        Y_unique = Y.unique(sorted=True)
        Y_max = Y_unique.max()
        C = Y_unique.numel()
        N = Y.shape[0]

        # Compute diagonal matrix of target codes. Here is an example for
        # three classes where X is a placeholder for actual diagonal entries:
        #
        #   tensor([[ X , 0.0, 0.0],
        #           [0.0,  X , 0.0],
        #           [0.0, 0.0,  X ]])

        target_codes = N * torch.ones(C, device=model.device)
        for c, Yc in enumerate(Y_unique):
            Nc = (Y == Yc).float().sum().to(device=model.device)
            target_codes[c] /= Nc
            target_codes[c].sqrt_()
        target_codes = target_codes.diag()

        # Create a code book so that arbitrary non-negative vectors of labels
        # can be used to retrieve the corresponding target codes directly. If
        # the class labels in the above example were 1, 2 and 5, the code book
        # would look like this:
        #
        #   tensor([[nan, nan, nan],
        #           [ X , 0.0, 0.0],    <-- 1
        #           [0.0,  X , 0.0],    <-- 2
        #           [nan, nan, nan],
        #           [nan, nan, nan],
        #           [0.0, 0.0,  X ]])   <-- 5

        self.target_codes = torch.full((Y_max + 1, C), np.nan, device=model.device)
        self.target_codes[Y_unique] = target_codes

        # --------------
        # Compute sigma
        # --------------
        self.sigma = self.target_codes[Y].std(dim=0).mean()

        # --------------------------
        # Finalize extra parameters
        # --------------------------
        self.extra_features = C

        # HINT: There two possible views on the extra layer.
        #
        #   VIEW 1:
        #
        #       EXTRA
        #       |            extra_weight.shape: (C, self.out_features)
        #       OUT
        #       |
        #       IN
        #
        #   VIEW 2:
        #
        #       OUT
        #      /   \         extra_weight.shape: (self.out_features, C)
        #    IN     EXTRA
        #
        # We choose VIEW 2 because the hidden probabilities OUT are based on
        # the clamped visible states IN and EXTRA.

        self.extra_weight.materialize(shape=(self.out_features, C))
        if self.extra_bias is not None:
            self.extra_bias.materialize(shape=(C,))

        with temp_seed(self.seed + 1):
            if self.weight_init_fn is None:
                torch_default_weight_init_fn(self.extra_weight)
            else:
                self.weight_init_fn(self.extra_weight, **self.weight_init_fn_kwargs)

            if self.extra_bias is not None:
                if self.bias_init_fn is None:
                    # HINT: Since we choose option 2 above, we pass
                    # extra_weight.t(). It is used when the transformation
                    # direction is "from OUT (hidden) to EXTRA (visible)". This
                    # is handled similarly in renda.core.models.rbm.RBM where
                    # it concerns the in_bias parameter.
                    torch_default_bias_init_fn(self.extra_bias, self.extra_weight.t())
                else:
                    self.bias_init_fn(self.extra_bias, **self.bias_init_fn_kwargs)

        # ---------------------------------------------
        # For PyTorch Lightning's load_from_checkpoint
        # ---------------------------------------------

        # HINT concerning extra parameters / target_codes buffer:

        # This is crucial for load_from_checkpoint. Withough _extra_features
        # the saved state_dict cannot be loaded into a "freshly" instantiated
        # object of this core model.

        model.update_hyperparameters({"_extra_features": C})

    def model_based_teardown(self, model: LightningModule):
        self._generator = None


class _CD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        T,
        weight,
        extra_weight,
        bias,
        in_bias,
        extra_bias,
        num_cd_steps,
        generator,
        sigma,
        lambda_,
    ):
        # =====================================================================
        # Supervised forward
        # =====================================================================
        # ---------------
        # Positive phase
        # ---------------
        sup_pos_vis_probs = X
        sup_pos_tar_probs = T
        sup_pos_hid_probs = _vis_and_tar_to_hid(
            sup_pos_vis_probs,
            sup_pos_tar_probs,
            weight,
            extra_weight,
            bias,
            sigma,
        )
        sup_pos_hid_states = _sample_hid(sup_pos_hid_probs, generator)

        # ---------------
        # Negative phase
        # ---------------
        sup_neg_hid_states = sup_pos_hid_states
        sup_neg_vis_probs = X
        for _ in range(num_cd_steps):
            sup_neg_tar_probs = _hid_to_tar(
                sup_neg_hid_states,
                extra_weight,
                extra_bias,
                sigma,
            )
            sup_neg_hid_probs = _vis_and_tar_to_hid(
                sup_neg_vis_probs,
                sup_neg_tar_probs,
                weight,
                extra_weight,
                bias,
                sigma,
            )
            sup_neg_hid_states = _sample_hid(sup_neg_hid_probs, generator)

        sup_loss = (sup_pos_tar_probs - sup_neg_tar_probs).pow(2).mean().sqrt()

        # =====================================================================
        # Unsupervised forward
        # =====================================================================
        # ---------------
        # Positive phase
        # ---------------
        uns_pos_vis_probs = X
        uns_pos_hid_probs = _vis_to_hid(uns_pos_vis_probs, weight, bias)
        uns_pos_hid_states = _sample_hid(uns_pos_hid_probs, generator)

        # ---------------
        # Negative phase
        # ---------------
        uns_neg_hid_states = uns_pos_hid_states
        for _ in range(num_cd_steps):
            uns_neg_vis_probs = _hid_to_vis(uns_neg_hid_states, weight, in_bias)
            uns_neg_hid_probs = _vis_to_hid(uns_neg_vis_probs, weight, bias)
            uns_neg_hid_states = _sample_hid(uns_neg_hid_probs, generator)

        uns_loss = (uns_pos_vis_probs - uns_neg_vis_probs).pow(2).mean().sqrt()

        # =====================================================================
        # For gradient calculation in backward method
        # =====================================================================
        ctx.save_for_backward(
            # Parameters
            weight,
            extra_weight,
            bias,
            in_bias,
            extra_bias,
            # From supervised forward
            sup_pos_vis_probs,
            sup_pos_hid_probs,
            sup_pos_tar_probs,
            sup_neg_vis_probs,
            sup_neg_hid_probs,
            sup_neg_tar_probs,
            # From unsupervised forward
            uns_pos_vis_probs,
            uns_pos_hid_probs,
            uns_neg_vis_probs,
            uns_neg_hid_probs,
            # Buffers
            sigma,
            lambda_,
        )

        # =====================================================================
        # Combine results from forward and return
        # =====================================================================
        if lambda_ == 0.0:
            loss = sup_loss
        elif lambda_ == 1.0:
            loss = uns_loss
        else:
            loss = (1 - lambda_) * sup_loss + lambda_ * uns_loss

        Z = uns_pos_hid_probs  # Features
        RX = uns_neg_vis_probs  # Reconstructions of the data
        RT = sup_neg_tar_probs  # Reconstructions of the target codes

        return loss, sup_loss, uns_loss, Z, RX, RT

    @staticmethod
    def backward(ctx, *grad_outputs):
        # =====================================================================
        # From forward method
        # =====================================================================
        (
            # Parameters
            weight,
            extra_weight,
            bias,
            in_bias,
            extra_bias,
            # From supervised forward
            sup_pos_vis_probs,
            sup_pos_hid_probs,
            sup_pos_tar_probs,
            sup_neg_vis_probs,
            sup_neg_hid_probs,
            sup_neg_tar_probs,
            # From unsupervised forward
            uns_pos_vis_probs,
            uns_pos_hid_probs,
            uns_neg_vis_probs,
            uns_neg_hid_probs,
            # Buffers
            sigma,
            lambda_,
        ) = ctx.saved_tensors

        # =====================================================================
        # Supervised backward
        # =====================================================================
        num_cases = sup_pos_vis_probs.shape[0]

        # Supervised weight gradient
        if lambda_ == 1.0:
            sup_d_weight = torch.zeros_like(weight)
        else:
            sup_pos_hid_vis = sup_pos_hid_probs.t() @ sup_pos_vis_probs
            sup_neg_hid_vis = sup_neg_hid_probs.t() @ sup_neg_vis_probs
            sup_d_weight = -(
                (sup_pos_hid_vis - sup_neg_hid_vis) / num_cases - 0.0002 * weight
            )

        # Supervised extra_weight gradient
        if lambda_ == 1.0:
            sup_d_extra_weight = torch.zeros_like(extra_weight)
        else:
            sup_pos_hid_tar = sup_pos_hid_probs.t() @ sup_pos_tar_probs
            sup_neg_hid_tar = sup_neg_hid_probs.t() @ sup_neg_tar_probs
            sup_d_extra_weight = -(
                (sup_pos_hid_tar - sup_neg_hid_tar) / num_cases / sigma
                - 0.0002 * extra_weight  # !!! Weight decay is sigma-independent
            )

        # Supervised bias gradient
        if bias is None:
            sup_d_bias = None
        elif lambda_ == 1.0:
            sup_d_bias = torch.zeros_like(bias)
        else:
            sup_pos_hid = sup_pos_hid_probs.sum(dim=0)
            sup_neg_hid = sup_neg_hid_probs.sum(dim=0)
            sup_d_bias = -(sup_pos_hid - sup_neg_hid) / num_cases

        # Supervised in_bias gradient
        if in_bias is None:
            sup_d_in_bias = None
        else:
            sup_d_in_bias = torch.zeros_like(in_bias)

        # Supervised extra_bias gradient
        if extra_bias is None:
            sup_d_extra_bias = None
        elif lambda_ == 1.0:
            sup_d_extra_bias = torch.zeros_like(extra_bias)
        else:
            sup_pos_tar = sup_pos_tar_probs.sum(dim=0)
            sup_neg_tar = sup_neg_tar_probs.sum(dim=0)
            sup_d_extra_bias = -(sup_pos_tar - sup_neg_tar) / num_cases / (sigma**2)

        # =====================================================================
        # Unsupervised backward
        # =====================================================================
        num_cases = uns_pos_vis_probs.shape[0]

        # Unsupervised weight gradient
        if lambda_ == 0.0:
            uns_d_weight = torch.zeros_like(weight)
        else:
            uns_pos_hid_vis = uns_pos_hid_probs.t() @ uns_pos_vis_probs
            uns_neg_hid_vis = uns_neg_hid_probs.t() @ uns_neg_vis_probs
            uns_d_weight = -(
                (uns_pos_hid_vis - uns_neg_hid_vis) / num_cases - 0.0002 * weight
            )

        # Unsupervised extra_weight gradient
        uns_d_extra_weight = torch.zeros_like(extra_weight)

        # Unsupervised bias gradient
        if bias is None:
            uns_d_bias = None
        elif lambda_ == 0.0:
            uns_d_bias = torch.zeros_like(bias)
        else:
            uns_pos_hid = uns_pos_hid_probs.sum(dim=0)
            uns_neg_hid = uns_neg_hid_probs.sum(dim=0)
            uns_d_bias = -(uns_pos_hid - uns_neg_hid) / num_cases

        # Unsupervised in_bias gradient
        if in_bias is None:
            uns_d_in_bias = None
        elif lambda_ == 0.0:
            uns_d_in_bias = torch.zeros_like(in_bias)
        else:
            uns_pos_vis = uns_pos_vis_probs.sum(dim=0)
            uns_neg_vis = uns_neg_vis_probs.sum(dim=0)
            uns_d_in_bias = -(uns_pos_vis - uns_neg_vis) / num_cases

        # Unsupervised extra_bias gradient
        if extra_bias is None:
            uns_d_extra_bias = None
        else:
            uns_d_extra_bias = torch.zeros_like(extra_bias)

        # =====================================================================
        # Combine results from backward and return
        # =====================================================================
        if lambda_ == 0.0:
            d_weight = sup_d_weight
            d_extra_weight = sup_d_extra_weight

            if bias is None:
                d_bias = None
            else:
                d_bias = sup_d_bias

            if in_bias is None:
                d_in_bias = None
            else:
                d_in_bias = sup_d_in_bias

            if extra_bias is None:
                d_extra_bias = None
            else:
                d_extra_bias = sup_d_extra_bias

        elif lambda_ == 1.0:
            d_weight = uns_d_weight
            d_extra_weight = uns_d_extra_weight

            if bias is None:
                d_bias = None
            else:
                d_bias = uns_d_bias

            if in_bias is None:
                d_in_bias = None
            else:
                d_in_bias = uns_d_in_bias

            if extra_bias is None:
                d_extra_bias = None
            else:
                d_extra_bias = uns_d_extra_bias
        else:
            d_weight = (1.0 - lambda_) * sup_d_weight + lambda_ * uns_d_weight
            d_extra_weight = (
                1.0 - lambda_
            ) * sup_d_extra_weight + lambda_ * uns_d_extra_weight

            if bias is None:
                d_bias = None
            else:
                d_bias = (1.0 - lambda_) * sup_d_bias + lambda_ * uns_d_bias

            if in_bias is None:
                d_in_bias = None
            else:
                d_in_bias = (1.0 - lambda_) * sup_d_in_bias + lambda_ * uns_d_in_bias

            if extra_bias is None:
                d_extra_bias = None
            else:
                d_extra_bias = (
                    1.0 - lambda_
                ) * sup_d_extra_bias + lambda_ * uns_d_extra_bias

        return (
            None,
            None,
            d_weight,
            d_extra_weight,
            d_bias,
            d_in_bias,
            d_extra_bias,
            None,
            None,
            None,
            None,
        )


# HINT: Recall how we chose to view the extra layer. This is how it translates
# to the names used here:
#
#       OUT                 bias          hid
#      /   \         ==>    weight       /   \      extra_weight
#    IN     EXTRA           in_bias   vis     tar   extra_bias


def _vis_and_tar_to_hid(vis, tar, weight, extra_weight, bias, sigma):
    # One has to take the bias into account:
    hid = F.linear(vis, weight, bias)
    # The other must ignore it:
    hid = hid + F.linear(tar / sigma, extra_weight, bias=None)
    return hid


def _vis_to_hid(vis, weight, bias):
    hid = F.linear(vis, weight, bias)
    return hid


def _hid_to_vis(hid, weight, in_bias):
    vis = F.linear(hid, weight.t(), in_bias)
    vis = F.sigmoid(vis)
    return vis


def _hid_to_tar(hid, extra_weight, extra_bias, sigma):
    tar = F.linear(hid * sigma, extra_weight.t(), extra_bias)
    return tar


def _sample_hid(hid_probs, generator):
    return torch.normal(hid_probs, torch.ones_like(hid_probs), generator=generator)
