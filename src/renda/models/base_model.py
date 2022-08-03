# -*- coding: utf-8 -*-
import copy
import io
import os
import textwrap
from collections import UserDict
from contextlib import redirect_stderr, redirect_stdout
from glob import glob
from pprint import pformat
from typing import Any, Dict

import numpy as np
import scipy.io as sio
import torch
from pytorch_lightning import LightningModule

import renda
from renda.logging.log_table import LogTable
from renda.logging.log_tool import LogTool
from renda.trainer import check_trainer_kwargs
from renda.utils.dict_ import OrderedDefaultDict, transform_dicts
from renda.utils.timing import Timer


class _ParentMixin:
    def __init__(self, *required_base_classes: Any) -> None:
        self._required_base_classes = required_base_classes
        self._parent = None

    @property
    def parent(self) -> Any:
        return self._parent

    @parent.setter
    def parent(self, parent: Any) -> None:
        if not isinstance(parent, self._required_base_classes):
            raise TypeError(
                f"Expected 'parent' to have the following base classes: "
                f"{', '.join(self._required_base_classes)}. Got {parent} with "
                f"these base classes: {', '.join(parent.__bases__)}."
            )

        self._parent = parent


class _ModelTrainerKwargs(UserDict):
    def __setitem__(self, key, value):
        if key == "overwrite_defaults":
            self._check_overwrite_defaults(value)

        check_trainer_kwargs(
            trainer_kwargs={**self, **{key: value}},
            overwrite_defaults=self.get("overwrite_defaults", False),
            context="trainer_kwarg(s)",
        )

        super().__setitem__(key, value)

    def _check_overwrite_defaults(self, value):
        value_ = self.get("overwrite_defaults", None)
        if value_ is not None:
            raise KeyError(
                f"The value of 'overwrite_defaults' can only be set once. "
                f"It has already been set to '{value_}'."
            )

        if not isinstance(value, bool):
            raise TypeError(f"Expected boolen. Got {value}.")


class _ModelTrainerKwargsMixin:
    def __init__(self):
        if not isinstance(self, _ParentMixin):
            raise TypeError(f"Missing base class: {_ParentMixin}.")

        self._trainer_kwargs = _ModelTrainerKwargs()

    @property
    def trainer_kwargs(self):
        return self._trainer_kwargs

    @property
    def all_trainer_kwargs(self):
        if self.parent is None:
            return copy.deepcopy(self._trainer_kwargs)
        else:
            return {
                **copy.deepcopy(self.parent.all_trainer_kwargs),
                **copy.deepcopy(self._trainer_kwargs),
            }


class _ModelValFunctionsMixin:
    def __init__(self):
        if not isinstance(self, _ParentMixin):
            raise TypeError(f"Missing base class: {_ParentMixin}.")

        self._val_functions = []
        self._val_function_outputs = []

    @property
    def val_functions(self):
        return self._val_functions

    @property
    def all_val_functions(self):
        if self.parent is None:
            return copy.deepcopy(self._val_functions)
        else:
            val_functions = copy.deepcopy(self.parent.all_val_functions)
            val_functions += copy.deepcopy(self._val_functions)
            return val_functions

    def run_val_functions(self, *args, **kwargs):
        for val_function in self.all_val_functions:
            with io.StringIO() as f:
                with redirect_stderr(f), redirect_stdout(f):
                    val_function(*args, **kwargs)

                val_function_output = f.getvalue()
                val_function_output = val_function_output.strip()
                self._val_function_outputs.append(val_function_output)

    def print_val_function_outputs(self) -> None:
        for ii, val_function_output in enumerate(self._val_function_outputs):
            self.print(f"Output of val function {ii}:", end="\n\n")
            if len(val_function_output) > 0:
                self.print(val_function_output, end="\n\n")
            else:
                self.print("No output.", end="\n\n")

        self._val_function_outputs.clear()


class BaseModel(
    _ModelValFunctionsMixin,
    _ModelTrainerKwargsMixin,
    _ParentMixin,
    LightningModule,
):
    LOG_TABLE_COLUMN_NAMES = []
    LOG_TABLE_MIN_COLUMN_WIDTH = 9

    def __init__(self) -> None:
        LightningModule.__init__(self)
        _ParentMixin.__init__(self, _ModelValFunctionsMixin, _ModelTrainerKwargsMixin)
        _ModelTrainerKwargsMixin.__init__(self)
        _ModelValFunctionsMixin.__init__(self)

        self.core_model = None
        self.loss = None
        self.optimizer = None
        self.scheduler = []

        self._log_table = LogTable(self)
        self._log_tool = LogTool(self)

        self._losses_to_print = []
        self._comments_to_print = []

        self._timer = Timer()
        self._training_timer = Timer()
        self._validation_timer = Timer()
        self._epoch_timer = Timer()
        self._step_timer = Timer()

        self.register_buffer("_num_runs_model_based_setup", torch.tensor(1))
        self.register_buffer("_num_runs_model_based_teardown", torch.tensor(1))

        self._hyperparameters_to_update = {}

    # =========================================================================
    # BaseModel methods
    # =========================================================================
    def validation_loss_summary(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def print_losses(self, *losses, comment: str = "") -> None:
        self._losses_to_print.append(losses)
        self._comments_to_print.append(comment)

    def update_hyperparameters(self, dict_: Dict[str, Any]):
        self._hyperparameters_to_update.update(dict_)

    # =========================================================================
    # LightningModule methods
    # =========================================================================
    def forward(self, *args, **kwargs) -> Any:
        return self.core_model.forward(*args, **kwargs)

    def validation_epoch_end(self, outputs):
        outputs = self._process_outputs(outputs)

        # This does not work for loss summaries when training of RBMs:
        # outputs = transform_dicts(*outputs, transforms=[torch.Tensor.cpu])
        # So for now, we just do it for the user-defined val functions

        self.validation_loss_summary(*outputs, log_tool=self._log_tool)

        outputs = transform_dicts(*outputs, transforms=[torch.Tensor.cpu])
        self.run_val_functions(*outputs, log_tool=self._log_tool)

    def _process_outputs(self, outputs):
        outputs_ = OrderedDefaultDict(lambda: OrderedDefaultDict(list))

        # -----------------------------
        # Visit outputs per dataloader
        # -----------------------------
        dataloader_names = self._log_tool.val_dataloader_names
        for dataloader_index, dataloader_name in enumerate(dataloader_names):

            if len(dataloader_names) > 1:
                dataloader_outputs = outputs[dataloader_index]
            else:
                dataloader_outputs = outputs

            # ---------------------------
            # Visit step (batch) outputs
            # ---------------------------
            for step_outputs in dataloader_outputs:

                if not isinstance(step_outputs, tuple):
                    step_outputs = (step_outputs,)

                for index, tensor in enumerate(step_outputs):

                    # The index reflects the order in which 'validation_step'
                    # returns these tensors. We swap dimensions to index,
                    # dataloader_name, step (batch) so that we can later
                    # concatenate the tensors per [index][dataloader_name]
                    # along the [step (batch)] dimension.

                    tensor = tensor.detach()
                    outputs_[index][dataloader_name].append(tensor)

        # -----------------------------------------
        # Concatenate along step (batch) dimension
        # -----------------------------------------
        for position in outputs_.keys():
            for dataloader_name in dataloader_names:
                outputs_[position][dataloader_name] = torch.cat(
                    outputs_[position][dataloader_name]
                )

        # ------------------
        # Return index-wise
        # ------------------
        return tuple(outputs_.values())

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = [torch.optim.Adam(self.core_model.parameters())]

        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]

        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]

        return self.optimizer, self.scheduler

    # =========================================================================
    # LightningModule hooks
    # =========================================================================
    def on_fit_start(self):
        self._print_renda_info()
        self._print_experiment_info()
        self._print_model_info()
        self._print_trainer_info()

        self._timer.reset()
        self._validation_timer.reset()

    def _print_renda_info(self) -> None:
        self._print_heading(f"RENDA (VERSION {renda.__version__})")

    def _print_heading(self, heading: str) -> None:
        heading = heading.strip().upper()
        self.print(f"    {'_' * (len(heading) + 2)}")
        self.print(f"___/ {heading} \\".ljust(80, "_"), end="\n\n")

    def _print_experiment_info(self) -> None:
        self._print_heading("EXPERIMENT")
        self.print("exp_group:", self._log_tool.exp_group)
        self.print("exp_name:", self._log_tool.exp_name)
        self.print("exp_version_dir:", self._log_tool.exp_version_dir)
        self.print("exp_phase:", self._log_tool.exp_phase)
        self.print("exp_path:", self._log_tool.exp_path)
        self.print("exp_pid:", self._log_tool.exp_pid)
        self.print("exp_ppid:", self._log_tool.exp_ppid, end="\n\n")

    def _print_model_info(self) -> None:
        self._print_heading("MODEL")
        self.print(f"{self.__class__.__name__} with hyperparameters:", end="\n\n")

        if len(self.hparams) == 0:
            HPARAMS = (
                "Couldn't find hyperparameters. Is this intended or did you "
                "forget to call 'self.save_hyperparameters()' at the end of "
                "the '__init__' method of your model class? Note that this "
                "might indicate that you'll not be able to load a model via "
                "ReNDA.load_from_checkpoint(...)."
            )
            HPARAMS = textwrap.fill(HPARAMS, width=70)
        else:
            HPARAMS = pformat(self.hparams, depth=4, sort_dicts=False)

        self.print(HPARAMS, end="\n\n")

    def _print_trainer_info(self) -> None:
        self._print_heading("TRAINER")
        self.print("PyTorch Lightning Trainer - using defaults except for:", end="\n\n")
        self.print(pformat(self.trainer.trainer_kwargs, depth=1), end="\n\n")

    def on_train_start(self):
        self._print_heading("TRAINING")

    def on_train_epoch_start(self):
        self._log_table.print_header()
        self._log_table.print_hline("=")

        self._training_timer.resume()  # Starts timer if it hasn't been started yet
        self._validation_timer.pause()
        self._epoch_timer.reset()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self.trainer.output_file.flush()
        self._step_timer.reset()

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self._losses_to_print != []:
            self._log_table.print_entry(
                self.current_epoch + 1,
                batch_idx + 1,
                self._step_timer.value,
                *self._losses_to_print[0],
                comment=self._comments_to_print[0],
            )

        self._losses_to_print.clear()
        self._comments_to_print.clear()

    def on_validation_start(self):
        # HACK: Lets the core_model look at to THIS model. (One core_model
        # where this is required is renda.core.models.gbrbm.GBRBM - it uses
        # this backdoor to get access to the data.) This seems to be the
        # latest possible hook before the core_model is used for the first
        # time. We can only hope that THIS model does currently provide the
        # information that the core_model requires. It is important to note
        # that the access to that information may be provided differently by
        # different versions of PyTorch Lightning. CONCLUSION: We should use
        # this backdoor only if there is no other way. If we use it, we should
        # try to realize the core_model's setup without accessing too much
        # information from the model.
        if self._num_runs_model_based_setup > 0:
            self._num_runs_model_based_setup -= 1
            if hasattr(self.core_model, "model_based_setup"):
                self.core_model.model_based_setup(self)

    def on_validation_epoch_start(self):
        self._epoch_timer.pause()
        self._training_timer.pause()
        self._validation_timer.resume()

    def on_validation_epoch_end(self):
        if self._losses_to_print != []:

            # TODO: Implement a buffer for these losses and comments so that
            # looping through them becomes easier, and consider moving this
            # buffer (and most of the following code) to the LogTable class.
            # Model classes should really only be triggering the LogTable
            # class to print its buffer content.

            zip_ = enumerate(zip(self._losses_to_print, self._comments_to_print))

            for ii, (losses, comment) in zip_:
                if self.global_step == 0:
                    # ----------------------------
                    # Entered during sanity check
                    # ----------------------------
                    if ii == 0:
                        self._print_heading("BEFORE TRAINING")
                        self._log_table.print_header()
                        self._log_table.print_hline("=")

                    self._log_table.print_entry(*losses, comment=comment)
                else:
                    # --------------------------------
                    # Entered during regular training
                    # --------------------------------
                    if ii == 0:
                        self._log_table.print_hline()
                        self._log_table.print_entry(
                            self.current_epoch + 1,
                            None,
                            self._epoch_timer.value,
                            *losses,
                            comment=comment,
                        )
                    else:
                        self._log_table.print_entry(
                            None,
                            None,
                            None,
                            *losses,
                            comment=comment,
                        )

            self.print(end="\n")
            self.print_val_function_outputs()

        self._losses_to_print.clear()
        self._comments_to_print.clear()

    def on_fit_end(self):
        self._print_heading("TRAINING COMPLETED")
        self.print(f"Training duration:     {self._training_timer.value}")
        self.print(f"Validation duration:   {self._validation_timer.value}")
        self.print(f"Total duration:        {self._timer.value}")

        if self.trainer.save_results_for_matlab:
            self._save_results_for_matlab()

        # HACK: Lets the core_model look at to THIS model one last time. We
        # can only hope that THIS model does still provide the information
        # that the core_model requires. It is important to note that the
        # access to that information may be provided differently by different
        # versions of PyTorch Lightning. CONCLUSION: We should use this
        # backdoor only if there is no other way. If we use it, we should try
        # to realize the core_model's teardown without accessing too much
        # information from the model.
        if self._num_runs_model_based_teardown > 0:
            self._num_runs_model_based_teardown -= 1
            if hasattr(self.core_model, "model_based_teardown"):
                self.core_model.model_based_teardown(self)

    def _save_results_for_matlab(self):
        exp_path = self._log_tool.exp_path
        checkpoints_path = os.path.join(exp_path, "checkpoints")
        ckpt_files = glob(os.path.join(checkpoints_path, "*.ckpt"))

        for ckpt_file in ckpt_files:
            checkpoint = torch.load(ckpt_file)
            mat_file = ckpt_file.replace(".ckpt", ".mat")
            saved_results = checkpoint["renda_dict"]["saved_results"]
            saved_results = transform_dicts(saved_results, transforms=[np.float64])
            sio.savemat(file_name=mat_file, mdict=saved_results)

    def on_save_checkpoint(self, checkpoint) -> None:
        saved_results = self._log_tool.get_saved_results(torch.float32)

        checkpoint.update(
            {
                "renda_dict": {
                    "renda_version": renda.__version__,
                    "model_class_module": self.__class__.__module__,
                    "model_class": self.__class__.__name__,
                    "saved_results": saved_results,
                }
            }
        )

        checkpoint["hyper_parameters"].update(self._hyperparameters_to_update)
