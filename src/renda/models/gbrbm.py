# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, Sequence, Union

import torch
from torchmetrics import MeanSquaredError

from renda.core_models.gbrbm import GBRBM as CoreGBRBM
from renda.models.base_model import BaseModel
from renda.optim.momentum_scheduler import LambdaMomentum


class GBRBM(BaseModel):
    LOG_TABLE_COLUMN_NAMES = ["RMSE"]
    LOG_TABLE_MIN_COLUMN_WIDTH = 10

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
        # -----
        # Init
        # -----
        seed: int = 0,
        weight_init_fn: Union[str, Callable] = "normal_",
        weight_init_fn_kwargs: Dict[str, Any] = {"std": 0.1},
        bias_init_fn: Union[str, Callable] = "zeros_",
        bias_init_fn_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=kwargs.keys())

        self.core_model = CoreGBRBM(
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

        self.optimizer = torch.optim.SGD(
            params=self.core_model.parameters(),
            lr=0.001,
            # This momentum value gets multiplied by the scheduler values (see
            # below). Thus, the initial momentum is 0.5 (epochs 0 to 4) and the
            # final momentum is 0.9 (epochs >= 5) as proposed by Hinton et al.
            momentum=1.0,
            # We implemented the weight decay in the core_model so that it only
            # effects the weights and never the biases.
            weight_decay=0,
        )

        self.scheduler = LambdaMomentum(
            optimizer=self.optimizer,
            momentum_lambda=lambda epoch: 0.5 if epoch < 5 else 0.9,
        )

        self._cumulative_loss = defaultdict(MeanSquaredError)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, Sequence):
            X = batch[0]
        else:
            X = batch

        self.core_model(X)
        loss = self.core_model.loss()
        self.print_losses(loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, Sequence):
            X = batch[0]
            Y = batch[1]
        else:
            X = batch

        Z, R = self.core_model(X)

        # Only relevant if trained on GPU: PyTorch Lightning's docs says we
        # shouldn't call .cpu(). However, if these tensors remained on GPU, we
        # would quickly run out of GPU memory because PyTorch Lightning
        # collects all validation step outputs internally.
        X = X.detach().cpu()
        Z = Z.detach().cpu()
        R = R.detach().cpu()

        # Also, our cumulative losses will always expect R and X to live on
        # the CPU. This is because making them dictionary entries prevents
        # their automatic registration as submodules, i.e., they are not
        # transfered to the same device as the model.
        self._cumulative_loss[dataloader_idx].update(R, X)

        if isinstance(batch, Sequence):
            Y = Y.detach().cpu()
            return X, Y, Z, R
        else:
            return X, Z, R

    def validation_loss_summary(self, *outputs, log_tool):
        losses = OrderedDict()

        for ii, dataloader_name in enumerate(log_tool.val_dataloader_names):
            loss = self._cumulative_loss[ii].compute().sqrt()
            self.print_losses(loss, comment=dataloader_name)

            # Our cumulative losses are not automatically reset. This is
            # because making them dictionary entries prevents their automatic
            # registration as submodules.
            self._cumulative_loss[ii].reset()

            # For "Pretty logging to TensorBoard" (see below)
            # And for saving results
            losses[dataloader_name] = loss

            # For PyTorch Lightning callbacks (ModelCheckpoint, etc.)
            # Unfortunately, this adds a section with one axes in the
            # TensorBoard's SCALARS tab that shows only the loss curve for the
            # current dataloader.
            self.log(f"{dataloader_name}_loss", loss)

        # ------------------------------
        # Pretty logging to TensorBoard
        # ------------------------------
        if log_tool.exp_phase.startswith("prefit"):
            # RBM is used for pretraining
            loss_name = f"{log_tool.exp_phase}_loss"
        else:
            # RBM is trained as a standalone model
            loss_name = "loss"

        # This makes it easier to monitor the model's generalization.
        log_tool.tensorboard.add_scalars(
            # -------------------------------------------------------------
            # Add 'losses' section in the tensorboard's SCALARS tab.
            # Add axes per 'loss_name' (prefit0_loss, prefit1_loss, etc.).
            # -------------------------------------------------------------
            main_tag=f"losses/{loss_name}",
            # --------------------------------------------------
            # Add loss curve per dataloader (train, val, etc.).
            # --------------------------------------------------
            tag_scalar_dict=losses,
            global_step=log_tool.current_epoch,
        )

        # -------
        # Saving
        # -------
        log_tool.save(losses, append=True)
