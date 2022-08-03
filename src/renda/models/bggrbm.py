# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, Optional, Union

import torch
from torchmetrics import MeanSquaredError

from renda.core_models.bggrbm import BGGRBM as CoreBGGRBM
from renda.models.base_model import BaseModel
from renda.optim.momentum_scheduler import LambdaMomentum


class BGGRBM(BaseModel):
    LOG_TABLE_COLUMN_NAMES = ["RMSE", "Sup. RMSE", "Uns. RMSE"]
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=kwargs.keys())

        self.core_model = CoreBGGRBM(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            in_bias=in_bias,
            extra_bias=extra_bias,
            lambda_=lambda_,
            num_cd_steps=num_cd_steps,
            seed=seed,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            bias_init_fn=bias_init_fn,
            bias_init_fn_kwargs=bias_init_fn_kwargs,
            _extra_features=_extra_features,
        )

        self.optimizer = torch.optim.SGD(
            params=self.core_model.parameters(),
            lr=0.01,
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

        self._cumulative_supervised_loss = defaultdict(MeanSquaredError)
        self._cumulative_unsupervised_loss = defaultdict(MeanSquaredError)

    def training_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1]

        self.core_model(X, Y)
        loss, supervised_loss, unsupervised_loss = self.core_model.loss()
        self.print_losses(loss, supervised_loss, unsupervised_loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch[0]
        Y = batch[1]

        Z, RX, T, RT = self.core_model(X, Y)

        # Only relevant if trained on GPU: PyTorch Lightning's docs says we
        # shouldn't call .cpu(). However, if these tensors remained on GPU, we
        # would quickly run out of GPU memory because PyTorch Lightning
        # collects all validation step outputs internally.
        X = X.detach().cpu()
        Y = Y.detach().cpu()
        Z = Z.detach().cpu()
        RX = RX.detach().cpu()
        T = T.detach().cpu()
        RT = RT.detach().cpu()

        # Also, our cumulative losses will always expect R, X, T_ and T to
        # live on the CPU. This is because making them dictionary entries
        # prevents their automatic registration as submodules, i.e., they are
        # not transfered to the same device as the model.
        self._cumulative_supervised_loss[dataloader_idx].update(RT, T)
        self._cumulative_unsupervised_loss[dataloader_idx].update(RX, X)

        return X, Y, Z, RX, T, RT

    def validation_loss_summary(self, *outputs, log_tool):
        X, Y, Z, RX, T, RT = outputs
        lambda_ = self.core_model.lambda_.cpu()

        losses = defaultdict(OrderedDict)

        for ii, dataloader_name in enumerate(log_tool.val_dataloader_names):
            supervised_loss = self._cumulative_supervised_loss[ii].compute().sqrt()
            unsupervised_loss = self._cumulative_unsupervised_loss[ii].compute().sqrt()
            loss = (1.0 - lambda_) * supervised_loss + lambda_ * unsupervised_loss
            self.print_losses(
                loss, supervised_loss, unsupervised_loss, comment=dataloader_name
            )

            # Our cumulative losses are not automatically reset. This is
            # because making them dictionary entries prevents their automatic
            # registration as submodules.
            self._cumulative_supervised_loss[ii].reset()
            self._cumulative_unsupervised_loss[ii].reset()

            # For "Pretty logging to TensorBoard" (see below)
            # And for saving results
            losses["loss"][dataloader_name] = loss
            losses["supervised_loss"][dataloader_name] = supervised_loss
            losses["unsupervised_loss"][dataloader_name] = unsupervised_loss

            # For PyTorch Lightning callbacks (ModelCheckpoint, etc.)
            # Unfortunately, this adds a section with one axes in the
            # TensorBoard's SCALARS tab that shows only the loss curve for the
            # current dataloader.
            self.log(f"{dataloader_name}_loss", loss)
            self.log(f"{dataloader_name}_supervised_loss", supervised_loss)
            self.log(f"{dataloader_name}_unsupervised_loss", unsupervised_loss)

        # ------------------------------
        # Pretty logging to TensorBoard
        # ------------------------------
        if log_tool.exp_phase.startswith("prefit"):
            # RBM is used for pretraining
            prefit = f"{log_tool.exp_phase}_"
        else:
            # RBM is trained as a standalone model
            prefit = ""

        # ------------------------------
        # Pretty logging to TensorBoard
        # ------------------------------
        # This makes it easier to monitor the model's generalization.
        for loss_name, loss_dict in losses.items():
            log_tool.tensorboard.add_scalars(
                # -------------------------------------------------------------
                # Add 'losses' section in the tensorboard's SCALARS tab.
                # Add axes per 'loss_name' (loss, supervised_loss,
                # unsupervised_loss).
                # -------------------------------------------------------------
                main_tag=f"losses/{prefit}{loss_name}",
                # --------------------------------------------------
                # Add loss curve per dataloader (train, val, etc.).
                # --------------------------------------------------
                tag_scalar_dict=loss_dict,
                global_step=self._log_tool.current_epoch,
            )

        # -------
        # Saving
        # -------
        log_tool.save(losses, append=True)
