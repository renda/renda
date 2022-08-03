# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from renda.logging.logger import Logger
from renda.utils.dict_ import transform_dicts
from renda.validation.plot_functions import plot_features, plot_image_grids


def torch_to_numpy_dtype(dtype):
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Expected torch.dtype, Got {dtype}.")

    dummy = torch.empty((), dtype=dtype)
    numpy_dtype = dummy.numpy().dtype
    del dummy

    return numpy_dtype


class LogTool:
    def __init__(self, lightning_module: pl.LightningModule) -> None:
        self._lightning_module = lightning_module
        self._saved_results = {}
        self._val_dataloader_names = None

    def log(self, *args, **kwargs) -> None:
        self._lightning_module.log(*args, **kwargs)

    def print(self, *args, **kwargs) -> None:
        self._lightning_module.print(*args, **kwargs)

    def save(
        self,
        dict_to_save: Dict[str, Any],
        append: bool = False,
        node: Optional[Dict[str, Any]] = None,
    ) -> None:
        if node is None:
            node = self._saved_results

        if not isinstance(dict_to_save, dict):
            raise TypeError(f"Expected a dict. Got {dict_to_save}.")

        def _process_leaf_object(leaf_object):
            leaf_object_as_passed = leaf_object

            try:
                leaf_object = np.expand_dims(leaf_object, axis=0)
                leaf_object = leaf_object.astype(np.float64)
            except Exception:
                raise TypeError(
                    f"Cannot convert leaf object {leaf_object_as_passed} of "
                    f"type {type(leaf_object_as_passed)} into a "
                    f"numpy.float64 array."
                )

            return leaf_object

        for k, v in dict_to_save.items():
            if isinstance(v, dict):
                # ----------
                # Recursion
                # ----------
                if k not in node.keys():
                    node[k] = {}
                self.save(v, append=append, node=node[k])
            else:
                # ------------
                # Leaf object
                # ------------
                if append:
                    # -------
                    # Append
                    # -------
                    if k not in node.keys():
                        node[k] = []
                    v = _process_leaf_object(v)
                    node[k].append(v)
                else:
                    # --------
                    # Replace
                    # --------
                    v = _process_leaf_object(v)
                    node[k] = [v]

    def get_saved_results(
        self,
        dtype: Union[torch.dtype, np.floating] = np.float64,
    ) -> Dict[str, Any]:
        """
        Returns a copy of the saved results.
        """
        self.save({"current_epoch": self.current_epoch}, append=False)

        if isinstance(dtype, torch.dtype):
            dtype = torch_to_numpy_dtype(dtype)
            transforms = [
                np.concatenate,
                lambda a: a.astype(dtype),
                np.squeeze,
                torch.from_numpy,
            ]
        elif isinstance(dtype, np.floating):
            transforms = [
                np.concatenate,
                dtype,
                np.squeeze,
            ]
        else:
            raise TypeError(
                f"Expected 'dtype' to be a torch.dtype or a np.floating type "
                f"(preferably, np.float64 or np.float32). Got {dtype}."
            )

        return transform_dicts(self._saved_results, transforms=transforms)

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def current_epoch(self) -> int:
        """
        Current epoch.

        It is 0 before the training and greater 1 during the training.
        """
        if self._lightning_module.global_step == 0:
            return self._lightning_module.current_epoch
        else:
            return self._lightning_module.current_epoch + 1

    @property
    def tensorboard(self) -> Logger:
        """
        The current tensorboard.
        """
        if not isinstance(self._lightning_module.logger, Logger):
            raise TypeError(
                f"Expected 'renda.core.logger.Logger'. Found "
                f"{self._lightning_module.logger}."
            )

        return self._lightning_module.logger.experiment

    @property
    def exp_group(self) -> str:
        """
        The current experiment group.
        """
        return self._lightning_module.logger.exp_group

    @property
    def exp_name(self) -> str:
        """
        The current experiment name.
        """
        return self._lightning_module.logger.exp_name

    @property
    def exp_phase(self) -> str:
        """
        The current experiment phase.
        """
        return self._lightning_module.logger.exp_phase

    @property
    def exp_version(self) -> int:
        """
        The current experiment version.
        """
        return self._lightning_module.logger.exp_version

    @property
    def exp_timestamp(self) -> str:
        """
        The current experiment timestamp.
        """
        return self._lightning_module.logger.exp_timestamp

    @property
    def exp_version_dir(self) -> str:
        """
        The current experiment version dir.
        """
        return self._lightning_module.logger.exp_version_dir

    @property
    def exp_path(self) -> str:
        """
        The current experiment path.
        """
        return self._lightning_module.logger.exp_path

    @property
    def exp_pid(self) -> int:
        return self._lightning_module.logger.exp_pid

    @property
    def exp_ppid(self) -> int:
        return self._lightning_module.logger.exp_ppid

    @property
    def val_dataloader_names(self) -> Sequence[str]:
        """
        List of names of all validation dataloaders.
        """
        if self._val_dataloader_names is None:
            trainer = self._lightning_module.trainer

            if trainer.datamodule is not None:
                val_dataloaders = trainer.datamodule.val_dataloader()
            else:
                val_dataloaders = trainer.val_dataloaders

            if not isinstance(val_dataloaders, List):
                raise RuntimeError

            self._val_dataloader_names = ["train"]
            for ii in range(1, len(val_dataloaders)):
                if ii == 1:
                    self._val_dataloader_names.append("val")
                else:
                    self._val_dataloader_names.append(f"val_{ii:02d}")

        return self._val_dataloader_names

    @property
    def lightning_module(self) -> pl.LightningModule:
        """
        The current LightningModule.
        """
        return self._lightning_module

    @property
    def trainer(self) -> pl.Trainer:
        """
        The current Trainer.
        """
        return self._lightning_module.trainer

    @property
    def train_dataloader(self) -> pl.Trainer:
        """
        The current train_dataloader.
        """
        return self._lightning_module.trainer.train_dataloader

    @property
    def val_dataloaders(self) -> pl.Trainer:
        """
        The current val_dataloaders.
        """
        return self._lightning_module.trainer.val_dataloaders

    @property
    def datamodule(self) -> pl.Trainer:
        """
        The current datamodule.
        """
        return self._lightning_module.trainer.datamodule

    # =========================================================================
    # Plot functions
    # =========================================================================
    def plot_features(
        self,
        X: Dict[str, Any],
        Y: Dict[str, Any],
        feature_names: Optional[Sequence[str]] = None,
        tag: str = "features",
        add_figures_to_tensorboard: bool = True,
        return_figures: Optional[bool] = None,
    ):
        figures = plot_features(X=X, Y=Y, feature_names=feature_names)

        if add_figures_to_tensorboard:
            for k, figure in figures.items():
                self.tensorboard.add_figure(
                    tag=f"{tag}/{k}",
                    figure=figure,
                    global_step=self.current_epoch,
                )

            if return_figures is None:
                # Since the user did not ask for the figures
                return_figures = False
        else:
            if return_figures is None:
                # Otherwise this function call would be pointless
                return_figures = True

        if return_figures:
            # It is the user's responsibility to close the figures
            return figures
        else:
            transform_dicts(figures, transforms=[lambda f: plt.close(f)])

    def plot_image_grids(
        self,
        X: Union[Dict[str, Any], Any],
        Y: Optional[Union[Dict[str, Any], Any]] = None,
        image_shape: Optional[Sequence[int]] = None,
        image_format: str = "CHW",
        num_rows: int = 8,
        num_columns: int = 8,
        tag: str = "image_grids",
        add_image_grids_to_tensorboard: bool = True,
        return_image_grids: Optional[bool] = None,
    ):
        image_grids = plot_image_grids(
            X=X,
            Y=Y,
            image_shape=image_shape,
            image_format=image_format,
            num_rows=num_rows,
            num_columns=num_columns,
        )

        if add_image_grids_to_tensorboard:
            for k, image_grid in image_grids.items():
                self.tensorboard.add_image(
                    tag=f"{tag}/{k}",
                    img_tensor=image_grid,
                    global_step=self.current_epoch,
                )

            if return_image_grids is None:
                # Since the user did not ask for the image grids
                return_image_grids = False
        else:
            if return_image_grids is None:
                # Otherwise this function call would be pointless
                return_image_grids = True

        if return_image_grids:
            # It is the user's responsibility to close the figures
            return image_grids
