# -*- coding: utf-8 -*-
from typing import Union

import torch

from renda.loss_functions.gerda_loss import GerDALoss
from renda.loss_functions.nmse_loss import NMSELoss


class ReNDALoss(torch.nn.Module):
    def __init__(
        self,
        lambda_: Union[float, int] = 0.5,
        weighted: bool = True,
        _gerda_loss_version: int = -1,
        _nmse_loss_version: int = -1,
    ) -> None:
        super().__init__()

        self.lambda_ = self._process_lambda(lambda_)
        self.weighted = weighted
        self._gerda_loss_version = _gerda_loss_version
        self._nmse_loss_version = _nmse_loss_version

        self.encoder_loss_function = GerDALoss(
            weighted=weighted, _version=_gerda_loss_version
        )

        self.decoder_loss_function = NMSELoss(_version=_nmse_loss_version)

    def _process_lambda(self, lambda_):
        if (
            lambda_ is None
            or not isinstance(lambda_, (float, int))
            or not 0.0 <= lambda_ <= 1.0
        ):
            raise ValueError(
                f"Expected real number between 0.0 and 1.0. Got {lambda_}."
            )
        return lambda_

    def forward(self, X, Y, Z, R):
        encoder_loss = self.encoder_loss_function(Z, Y)
        decoder_loss = self.decoder_loss_function(X, R)

        if self.lambda_ == 0.0:
            loss = encoder_loss
        elif self.lambda_ == 1.0:
            loss = decoder_loss
        else:
            loss = (1.0 - self.lambda_) * encoder_loss
            loss += self.lambda_ * decoder_loss

        return loss, encoder_loss, decoder_loss

    def extra_repr(self):
        return f"lambda_={self.lambda_}, weighted={self.weighted}"
