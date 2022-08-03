# -*- coding: utf-8 -*-
import copy
import os
import re
import shutil
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader

from renda.logging.logger import LoggerBuilder
from renda.utils.data import (
    convert_to_sequential_dataloader,
    transform_dataloaders,
    transform_datamodule,
)
from renda.utils.pytorch_lightning import pl_get_profiler_class_by_name, pl_no_output

_BLOCKED_TRAINER_KWARGS = {
    "enable_progress_bar": False,
    "logger": "'INFO: Will be set to renda.logging.logger.Logger internally.'",
    "num_sanity_val_steps": -1,
}

_DEFAULT_TRAINER_KWARGS = {
    "benchmark": False,
    "deterministic": True,
    "precision": 32,
    "weights_summary": None,
}

_TRAINER_KWARGS_PREFIX_PATTERNS = [
    re.compile("fit__"),
    re.compile("prefit__"),
    re.compile("prefit[0-9]+__"),
]


class Trainer:
    """ """

    def __init__(
        self,
        *,
        exp_group: str = "results",
        exp_name: str = "experiment",
        print_to_file: bool = False,
        save_results_for_matlab: bool = False,
        overwrite_defaults: bool = False,
        **trainer_kwargs,
    ) -> None:
        self._exp_group = exp_group
        self._exp_name = exp_name
        self._print_to_file = print_to_file
        self._save_results_for_matlab = save_results_for_matlab
        self._overwrite_defaults = overwrite_defaults

        self._trainer_kwargs_groups = self._group_trainer_kwargs(trainer_kwargs)

        # HINT pt. 1: We check each group of trainer_kwargs but we don't
        # create a PyTorch Lightning trainer yet. Instead, we create trainers
        # dynamically whenever prefit() or fit() is called.
        for group, trainer_kwargs in self._trainer_kwargs_groups.items():
            check_trainer_kwargs(
                trainer_kwargs=trainer_kwargs,
                overwrite_defaults=overwrite_defaults,
                context=group,
            )

        # HINT pt. 2: When prefit() or fit() is called, the current trainer
        # and trainer_kwargs are accessible through these attributes.
        self._trainer = None
        self._trainer_kwargs = None

        # HINT pt. 3: The dynamic creation of trainers lets us assign loggers
        # that all log into subfolders of the same common log folder. The
        # current logger and the current output_file are accessible through
        # these attributes. Accessing the latter only makes sense if
        # print_to_file is True.
        self._logger = None
        self._output_file = type("", (object,), {"flush": lambda: None})

        # HINT pt. 4: The logger_builder will be set only once in _fit(), when
        # either prefit() or fit() is called for the first time. On creation,
        # the logger_builder determines the path of the common log folder
        # mentioned in HINT pt. 3. It is then used to build loggers as
        # described (see _fit() and __fit()).
        self._logger_builder = None

    @staticmethod
    def _group_trainer_kwargs(trainer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        trainer_kwargs_groups = defaultdict(dict)

        for key, value in trainer_kwargs.items():
            for prefix_pattern in _TRAINER_KWARGS_PREFIX_PATTERNS:
                match = re.match(prefix_pattern, key)
                if match is not None:
                    group = key[: match.end() - 2]
                    key_ = key[match.end() :]
                    break
            else:
                group = "common"
                key_ = key

            trainer_kwargs_groups[group][key_] = value

        return trainer_kwargs_groups

    def fit(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> None:
        """ """
        self._fit(
            exp_phase="fit",
            model=model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
        )

    def _fit(
        self,
        exp_phase: str,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
        _local_trainer_kwargs: Dict[str, Any] = {},
    ) -> None:
        """
        Handle the trainer's print_to_file option.

        - Build the logger first so that the file to print to can be placed in
          the folder where everything else gets logged.
        - Set up redirection of stdout and stderr and wrap it around all
          further steps (see __fit method).

        """
        if self._logger_builder is None:
            self._logger_builder = LoggerBuilder(
                exp_group=self._exp_group,
                exp_name=self._exp_name,
            )

        self._logger = self._logger_builder.build(exp_phase)

        if self._print_to_file:
            output_file = os.path.join(self._logger.exp_path, "output.txt")
            with open(output_file, "w") as f:
                self._output_file = f
                with redirect_stdout(f), redirect_stderr(f):
                    self.__fit(
                        model=model,
                        train_dataloader=train_dataloader,
                        val_dataloaders=val_dataloaders,
                        datamodule=datamodule,
                        _local_trainer_kwargs=_local_trainer_kwargs,
                    )
        else:
            self.__fit(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                _local_trainer_kwargs=_local_trainer_kwargs,
            )

    def __fit(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
        _local_trainer_kwargs: Dict[str, Any] = {},
    ) -> None:
        """
        Launch the actual training.

        - Prepare data.
        - Finalize trainer_kwargs.
        - Create the Lightning trainer.
        - Call the Lightning trainer's fit method.

        """
        # =====================================================================
        # Prepare data
        # =====================================================================
        datamodule = _prepare_datamodule(datamodule)
        train_dataloader, val_dataloaders = _prepare_dataloaders(
            train_dataloader, val_dataloaders
        )

        # =====================================================================
        # Finalize trainer_kwargs
        # =====================================================================
        # ----------------
        # From __init__()
        # ----------------
        if self._logger.exp_phase.startswith("prefit"):
            initial_trainer_kwargs = {
                **self._trainer_kwargs_groups.get("common", {}),
                **self._trainer_kwargs_groups.get("prefit", {}),
                **self._trainer_kwargs_groups.get(self._logger.exp_phase, {}),
            }
        elif self._logger.exp_phase == "fit":
            initial_trainer_kwargs = {
                **self._trainer_kwargs_groups.get("common", {}),
                **self._trainer_kwargs_groups.get("fit", {}),
            }
        else:
            initial_trainer_kwargs = {}

        # -----------
        # From model
        # -----------
        model_trainer_kwargs = model.all_trainer_kwargs  # All means model + parents
        model_overwrite_defaults = model_trainer_kwargs.pop("overwrite_defaults", False)

        # ---------------------
        # Merge trainer_kwargs
        # ---------------------
        if self._overwrite_defaults or model_overwrite_defaults:
            self._trainer_kwargs = {
                **copy.deepcopy(_DEFAULT_TRAINER_KWARGS),
                **copy.deepcopy(initial_trainer_kwargs),
                **copy.deepcopy(model_trainer_kwargs),
                **copy.deepcopy(_BLOCKED_TRAINER_KWARGS),
                **copy.deepcopy(_local_trainer_kwargs),
            }
        else:
            self._trainer_kwargs = {
                **copy.deepcopy(initial_trainer_kwargs),
                **copy.deepcopy(model_trainer_kwargs),
                **copy.deepcopy(_DEFAULT_TRAINER_KWARGS),
                **copy.deepcopy(_BLOCKED_TRAINER_KWARGS),
                **copy.deepcopy(_local_trainer_kwargs),
            }

        # ---------------------
        # Tweak trainer_kwargs
        # ---------------------
        self._trainer_kwargs["logger"] = self._logger

        profiler = self._trainer_kwargs.get("profiler", None)
        if isinstance(profiler, str):
            # If specified as a string, we can increase usability by making
            # the profiler log to the same folder as the logger.
            profiler_class = pl_get_profiler_class_by_name(profiler)
            profiler = profiler_class(
                dirpath=self._logger.exp_path,
                filename="profiler",
            )
            self._trainer_kwargs["profiler"] = profiler
        else:
            pass  # Otherwise, we do not interfere.

        # =====================================================================
        # Create trainer
        # =====================================================================
        self._trainer = pl.Trainer(**self._trainer_kwargs)

        # Add user settings to trainer attributes
        setattr(self._trainer, "exp_group", self._exp_group)
        setattr(self._trainer, "exp_name", self._exp_name)
        setattr(self._trainer, "print_to_file", self._print_to_file)
        setattr(self._trainer, "save_results_for_matlab", self._save_results_for_matlab)
        setattr(self._trainer, "overwrite_defaults", self._overwrite_defaults)
        setattr(self._trainer, "trainer_kwargs", self._trainer_kwargs)

        # Add further functionalities
        setattr(self._trainer, "output_file", self._output_file)
        setattr(
            self._trainer,
            "get_sequential_train_dataloader",
            # HINT: Once the data is prepared, the val_dataloader at index 0
            # is a sequential version of the train_dataloader. If a datamodule
            # is used, this sequential train_dataloader is accessible after
            # datamodule.setup() has been called.
            lambda: self._get_val_dataloader(index=0),
        )

        # =====================================================================
        # Call fit
        # =====================================================================
        self._trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
        )

    def _get_val_dataloader(self, *, index: int = 0):
        ERROR_MESSAGE = (
            f"Could not retrieve val_dataloader at index {index} from "
            f"current trainer instance. The reasons for this can be as "
            f"follows: (1) WRONG ACCESS: The access to the val_dataloaders "
            f"may be provided differently by different versions of PyTorch "
            f"Lightning. (2) TOO EARLY: PyTorch Lightning has not connected "
            f"the passed datamodule or val_dataloders to the Trainer by "
            f"now.\n\n"
            f"HOW TO PROCEED: (1) Make sure you are using the PyTorch "
            f"Lightning version listed in our requirements.txt. (2) Contact "
            f"us if this error still occurs. (3) Also, please do not "
            f"hesitate to contact us if switching to the listed PyTorch "
            f"Lightning version is not an option for you."
        )

        try:
            datamodule = self._trainer.datamodule
            val_dataloaders = self._trainer.val_dataloaders

            if isinstance(datamodule, _DataModuleWrapper):
                dataloader = datamodule.val_dataloader()[index]
            elif isinstance(val_dataloaders, List):
                dataloader = val_dataloaders[index]
            else:
                raise RuntimeError(ERROR_MESSAGE)

            return dataloader

        except Exception as e:
            raise RuntimeError(
                f"{ERROR_MESSAGE}\n\nERROR CAUGHT: {e.__class__.__name__}: {e}"
            )

    def prefit(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> None:
        # These are required for our approch to saving the prefitted model
        train_dataloader_passed = train_dataloader
        val_dataloaders_passed = val_dataloaders
        datamodule_passed = datamodule

        # -------------
        # Prefit model
        # -------------
        rbm_stack = model.rbm_stack

        for ii, rbm in enumerate(rbm_stack):
            self._fit(
                model=rbm,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                exp_phase=f"prefit{ii}",
            )

            if ii < len(rbm_stack) - 1:
                train_dataloader = transform_dataloaders(rbm, train_dataloader)
                val_dataloaders = transform_dataloaders(rbm, val_dataloaders)
                datamodule = transform_datamodule(rbm, datamodule)

        rbm_stack.copy_parameters_to_model()

        # ---------------------
        # Save prefitted model
        # ---------------------
        # Run _fit with zero training / validation epochs and zero sanity
        # validation steps to create a new trainer (self._trainer) that is
        # connected to the prefitted model.
        with pl_no_output():
            self._fit(
                # Dummy exp_phase
                exp_phase="temp",
                # Connect prefitted model
                model=model,
                # Data is required but not used
                train_dataloader=train_dataloader_passed,
                val_dataloaders=val_dataloaders_passed,
                datamodule=datamodule_passed,
                # Turn off training and validation / sanity validation steps
                _local_trainer_kwargs={
                    "max_epochs": 0,
                    "num_sanity_val_steps": 0,
                },
            )

            # Remove log subfolder created for the dummy exp_phase
            shutil.rmtree(self._trainer.logger.exp_path)

            # Save checkpoint manually
            self._trainer.save_checkpoint(
                os.path.join(
                    self._trainer.logger.exp_group,
                    self._trainer.logger.exp_name,
                    self._trainer.logger.exp_version_dir,
                    "prefit.ckpt",
                )
            )

            # HINT: Should saving the checkpoint ever become dependent on what
            # is stored in the log subfolder created for the dummy exp_phase,
            # the above solution will cause an error. This is intended. We
            # prefer to learn about that change rather than letting them fly
            # under our radar.


def check_trainer_kwargs(
    trainer_kwargs: Dict[str, Any],
    overwrite_defaults: bool = False,
    context: Optional[str] = None,
) -> None:
    if context is None:
        context = ""
    else:
        context = f"Context: {context.strip()}. "

    intersection = set(_BLOCKED_TRAINER_KWARGS.keys()) & set(trainer_kwargs.keys())
    if len(intersection) > 0:
        raise KeyError(
            f"{context}Cannot set {', '.join(intersection)}. "
            f"The following trainer_kwarg(s) are blocked: "
            f"{', '.join(f'{k}={v}' for k, v in _BLOCKED_TRAINER_KWARGS.items())}."
        )

    model_overwrite_defaults = trainer_kwargs.pop("overwrite_defaults", False)
    if not (overwrite_defaults or model_overwrite_defaults):
        intersection = set(_DEFAULT_TRAINER_KWARGS.keys()) & set(trainer_kwargs.keys())
        if len(intersection) > 0:
            raise KeyError(
                f"{context}Cannot set: {', '.join(intersection)}. "
                f"The following trainer_kwarg(s) have default values: "
                f"{', '.join(f'{k}={v}' for k, v in _DEFAULT_TRAINER_KWARGS.items())}. "  # noqa: E501
                f"Set overwrite_defaults to True if you wish to overwrite "
                f"these default values."
            )

    try:
        with pl_no_output():
            trainer = pl.Trainer(**trainer_kwargs)
            del trainer

    except Exception as e:
        raise KeyError(
            f"{context}Cannot create Trainer from trainer_kwarg(s): "
            f"{', '.join(f'{k} = {v}' for k, v in trainer_kwargs.items())}. "
            f"In case you are trying to use prefixed keywords, make sure that "
            f"(1) they can actually be used in this context and (2) they "
            f"contain a double underscore, e.g., fit__max_epochs, "
            f"prefit__callbacks, prefit2__accelerator, etc. ERROR CAUGHT: "
            f"{e.__class__.__name__}: {e}"
        )


def _prepare_dataloaders(
    train_dataloader: Optional[DataLoader] = None,
    val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
) -> Tuple[Optional[DataLoader], Optional[List[DataLoader]]]:
    """
    Ensure that val_dataloaders is a list or None. If train_dataloader is not
    None, add a sequential version of it as the first val_dataloader.

    """
    if isinstance(val_dataloaders, DataLoader):
        val_dataloaders = [val_dataloaders]

    if isinstance(train_dataloader, DataLoader):
        train_dataloader_ = convert_to_sequential_dataloader(train_dataloader)
        if val_dataloaders is None:
            val_dataloaders = [train_dataloader_]
        elif isinstance(val_dataloaders, List):
            val_dataloaders = [train_dataloader_] + val_dataloaders

    return train_dataloader, val_dataloaders


def _prepare_datamodule(
    datamodule: Optional[LightningDataModule],
) -> Optional[LightningDataModule]:
    """
    Wrap datamodule if it's a LightningDataModule.

    """
    if isinstance(datamodule, LightningDataModule):
        datamodule = _DataModuleWrapper(datamodule)

    return datamodule


class _DataModuleWrapper(LightningDataModule):
    """
    Wrapper ensuring that val_dataloaders() returns a list or None. If
    train_dataloader() does not return None, a sequential version of the
    train_dataloader is returned as the first val_dataloader.

    All methods except val_dataloaders() simply delegate their calls to the
    respective methods of the wrapped datamodule.

    """

    def __init__(self, datamodule: LightningDataModule, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._datamodule = datamodule

        if hasattr(datamodule, "prepare_data_per_node"):
            attr = getattr(datamodule, "prepare_data_per_node")
            self.prepare_data_per_node = attr

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            try:
                return self._datamodule.__getattribute__(name)
            except AttributeError:
                raise AttributeError(
                    f"Datamodule '{self._datamodule.__class__.__name__}' has "
                    f"no attribute '{name}'."
                )

    def prepare_data(self, *args, **kwargs):
        return self._datamodule.prepare_data(*args, **kwargs)

    def setup(self, *args, **kwargs):
        return self._datamodule.setup(*args, **kwargs)

    def train_dataloader(self) -> Optional[DataLoader]:
        return self._datamodule.train_dataloader()

    def val_dataloader(self) -> Optional[List[DataLoader]]:
        train_dataloader, val_dataloaders = _prepare_dataloaders(
            self._datamodule.train_dataloader(),
            self._datamodule.val_dataloader(),
        )
        del train_dataloader
        return val_dataloaders

    def test_dataloader(self):
        return self._datamodule.test_dataloader()

    def predict_dataloader(self):
        return self._datamodule.predict_dataloader()

    def transfer_batch_to_device(self, *args, **kwargs):
        return self._datamodule.transfer_batch_to_device(*args, **kwargs)

    def on_before_batch_transfer(self, *args, **kwargs):
        return self._datamodule.on_before_batch_transfer(*args, **kwargs)

    def on_after_batch_transfer(self, *args, **kwargs):
        return self._datamodule.on_after_batch_transfer(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._datamodule.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._datamodule.state_dict(*args, **kwargs)

    def on_train_dataloader(self):
        return self._datamodule.on_train_dataloader()

    def on_val_dataloader(self):
        return self._datamodule.on_val_dataloader()

    def on_test_dataloader(self):
        return self._datamodule.on_test_dataloader()

    def on_predict_dataloader(self):
        return self._datamodule.on_predict_dataloader()

    def teardown(self, *args, **kwargs):
        return self._datamodule.teardown(*args, **kwargs)
