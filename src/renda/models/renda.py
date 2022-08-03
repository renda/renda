# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch

from renda.core_models.autoencoder import Autoencoder
from renda.loss_functions.renda_loss import ReNDALoss
from renda.models.base_model import (
    BaseModel,
    _ModelTrainerKwargsMixin,
    _ModelValFunctionsMixin,
    _ParentMixin,
)
from renda.models.bggrbm import BGGRBM
from renda.models.gbrbm import GBRBM
from renda.models.rbm import RBM
from renda.utils.activation import get_name_of_activation


class ReNDA(BaseModel):
    LOG_TABLE_COLUMN_NAMES = ["Loss", "Enc. loss", "Dec. loss"]
    LOG_TABLE_MIN_COLUMN_WIDTH = 10

    def __init__(
        self,
        # ----
        # Net
        # ----
        topology: Sequence[int] = None,
        bias: bool = True,
        activation: Union[str, Callable] = "Sigmoid",
        activation_kwargs: Dict[str, Any] = {},
        last_bias: bool = False,
        last_activation: Union[str, Callable] = "Identity",
        last_activation_kwargs: Dict[str, Any] = {},
        last_decoder_bias: bool = False,
        last_decoder_activation: Union[str, Callable] = "Identity",
        last_decoder_activation_kwargs: Dict[str, Any] = {},
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
        # -----
        # Loss
        # -----
        lambda_: Union[float, Sequence[float]] = [0.5, 0.5],
        weighted: bool = True,
        _gerda_loss_version: int = -1,
        _nmse_loss_version: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=kwargs.keys())

        if isinstance(lambda_, float):
            self.lambda_ = (lambda_, lambda_)
        elif isinstance(lambda_, Sequence) and all(
            isinstance(e, float) for e in lambda_
        ):
            if len(lambda_) == 1:
                # HINT: ReNDA's command line interface parses lambda_ as list,
                # even if the user only passes one lambda_ value.
                self.lambda_ = (lambda_[0], lambda_[0])
            elif len(lambda_) == 2:
                self.lambda_ = tuple(lambda_)
        else:
            raise ValueError(
                f"Expected float, or a sequence of one or two floats. "
                f"Got {lambda_}."
            )

        self.core_model = Autoencoder(
            topology=topology,
            bias=bias,
            activation=activation,
            activation_kwargs=activation_kwargs,
            last_bias=last_bias,
            last_activation=last_activation,
            last_activation_kwargs=last_activation_kwargs,
            last_decoder_bias=last_decoder_bias,
            last_decoder_activation=last_decoder_activation,
            last_decoder_activation_kwargs=last_decoder_activation_kwargs,
            tied_weights=tied_weights,
            tied_biases=tied_biases,
            seed=seed,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            bias_init_fn=bias_init_fn,
            bias_init_fn_kwargs=bias_init_fn_kwargs,
        )

        self.loss = ReNDALoss(
            lambda_=self.lambda_[1],
            weighted=weighted,
            _gerda_loss_version=_gerda_loss_version,
            _nmse_loss_version=_nmse_loss_version,
        )

        # Lazy initialization via rbm_stack property
        self._rbm_stack = None

    @property
    def rbm_stack(self):
        if self._rbm_stack is None:
            self._rbm_stack = _RBMStack(self.core_model, self.lambda_[0])

        return self._rbm_stack

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            X, Y = batch
            Q = X
        elif len(batch) == 3:
            X, Y, Q = batch

        Z, R = self.core_model(X)
        loss, encoder_loss, decoder_loss = self.loss(Q, Y, Z, R)
        self.print_losses(loss, encoder_loss, decoder_loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if len(batch) == 2:
            X, Y = batch
        if len(batch) == 3:
            X, Y, Q = batch

        Z, R = self.core_model(X)

        X = X.detach().cpu()
        Y = Y.detach().cpu()
        Z = Z.detach().cpu()
        R = R.detach().cpu()

        if len(batch) == 2:
            return X, Y, Z, R
        elif len(batch) == 3:
            Q = Q.detach().cpu()
            return X, Y, Z, Q, R

    def validation_loss_summary(self, *outputs, log_tool):
        if len(outputs) == 4:
            X, Y, Z, R = outputs
            Q = X
        elif len(outputs) == 5:
            X, Y, Z, Q, R = outputs

        losses = defaultdict(OrderedDict)

        for dataloader_name in log_tool.val_dataloader_names:
            loss, encoder_loss, decoder_loss = self.loss(
                Q[dataloader_name],
                Y[dataloader_name],
                Z[dataloader_name],
                R[dataloader_name],
            )

            self.print_losses(loss, encoder_loss, decoder_loss, comment=dataloader_name)

            # For "Pretty logging to TensorBoard" (see below)
            # And for saving results
            losses["loss"][dataloader_name] = loss.cpu()
            losses["encoder_loss"][dataloader_name] = encoder_loss.cpu()
            losses["decoder_loss"][dataloader_name] = decoder_loss.cpu()

            # For PyTorch Lightning callbacks (ModelCheckpoint, etc.)
            # Unfortunately, this adds a section with one axes in the
            # TensorBoard's SCALARS tab that shows only the loss curve for the
            # current dataloader.
            self.log(f"{dataloader_name}_loss", loss)
            self.log(f"{dataloader_name}_encoder_loss", encoder_loss)
            self.log(f"{dataloader_name}_decoder_loss", decoder_loss)

        # ------------------------------
        # Pretty logging to TensorBoard
        # ------------------------------
        # This makes it easier to monitor the model's generalization.
        for loss_name, loss_dict in losses.items():
            log_tool.tensorboard.add_scalars(
                # -------------------------------------------------------------
                # Add 'losses' section in the tensorboard's SCALARS tab.
                # Add axes per 'loss_name' (loss, encoder_loss, decoder_loss).
                # -------------------------------------------------------------
                main_tag=f"losses/{loss_name}",
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


class _RBMStack(_ModelValFunctionsMixin, _ModelTrainerKwargsMixin, _ParentMixin):
    def __init__(self, core_model: Autoencoder, lambda_: float) -> None:
        _ParentMixin.__init__(self, _ModelValFunctionsMixin, _ModelTrainerKwargsMixin)
        _ModelTrainerKwargsMixin.__init__(self)
        _ModelValFunctionsMixin.__init__(self)

        self._core_model = core_model
        self._lambda_ = lambda_
        self._rbm_stack = []

        layer_pairs = zip(
            self._core_model.encoder.layer,
            self._core_model.decoder.layer[::-1],  # Reverse order
        )
        num_layers = self._core_model.num_layers

        for ii, (encoder_layer, decoder_layer) in enumerate(layer_pairs):
            if ii == 0:
                # ------------
                # First layer
                # ------------
                activation = decoder_layer.activation
                if not isinstance(activation, str):
                    activation = get_name_of_activation(activation)

                if activation == "Sigmoid":
                    rbm_class = RBM
                else:
                    rbm_class = GBRBM

            elif ii < num_layers - 1:
                # --------------------
                # Intermediate layers
                # --------------------
                rbm_class = RBM

            else:
                # -----------
                # Last layer
                # -----------
                activation = encoder_layer.activation
                if not isinstance(activation, str):
                    activation = get_name_of_activation(activation)

                if activation == "Sigmoid":
                    # HINT concerning the TODO: Because the sigmoid activation
                    # maps to [0, 1], Bernoulli units would provide a better
                    # modeling than Gaussian units in this case.

                    rbm_class = BGGRBM  # TODO: Implement BBGRBM and use it here.
                else:
                    rbm_class = BGGRBM

            rbm = rbm_class(
                in_features=encoder_layer.in_features,
                out_features=encoder_layer.out_features,
                bias=True,
                in_bias=True,
                extra_bias=True,
                lambda_=self._lambda_,
                seed=encoder_layer.seed,
            )

            rbm.parent = self

            # HINT: torch.nn.Module's __repr__ code only works for string keys
            self._rbm_stack.append(rbm)

    def copy_parameters_to_model(self):
        with torch.no_grad():
            layer_pairs = zip(
                self._core_model.encoder.layer,
                self._core_model.decoder.layer[::-1],  # Reverse order
            )

            for ii, (encoder_layer, decoder_layer) in enumerate(layer_pairs):
                core_rbm = self._rbm_stack[ii].core_model

                encoder_layer.weight.copy_(core_rbm.weight)
                if encoder_layer.bias is not None:
                    encoder_layer.bias.copy_(core_rbm.bias)

                if not self._core_model.tied_weights:
                    decoder_layer.weight.copy_(core_rbm.weight)

                if not self._core_model.tied_biases:
                    if decoder_layer.bias is not None:
                        decoder_layer.bias.copy_(core_rbm.in_bias)

    def __getitem__(self, index):
        return self._rbm_stack[index]

    def __len__(self):
        return len(self._rbm_stack)

    def __repr__(self):
        # ------------------------------------------------------------------
        # 2022-05-23 (M. Becker): Copied from PyTorch 1.8.2 (LTS),
        # torch\nn\modules\module.py, and modified in the following ways to
        # print self._rbm_stack:
        #   - self._modules.items() -> enumerate(self._rbm_stack), see (1)
        #   - key -> str(key), see (2)
        # ------------------------------------------------------------------
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        # for key, module in self._modules.items():
        for key, module in enumerate(self._rbm_stack):  # (1)
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + str(key) + "): " + mod_str)  # (2)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def extra_repr(self) -> str:
        return (
            "________________________________________________________________\n"
            " NOTE: This is no LightningModule / torch.nn.Module. It is only \n"
            " printed like a torch.nn.Module to make it easier to inspect on \n"
            " the fly. However, each of the RBMs listed is a LightningModule \n"
            " and may be accessed for further inspection / customization.    \n"
            "________________________________________________________________\n"
        )

    def _get_name(self) -> str:
        # ---------------------------------------------------------
        # 2022-05-23 (M. Becker): Copied from PyTorch 1.8.2 (LTS),
        # torch\nn\modules\module.py, because __repr__ calls it.
        # ---------------------------------------------------------
        return self.__class__.__name__


def _addindent(s_, numSpaces):
    # ---------------------------------------------------------
    # 2022-05-23 (M. Becker): Copied from PyTorch 1.8.2 (LTS),
    # torch\nn\modules\module.py, because __repr__ calls it.
    # ---------------------------------------------------------
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
