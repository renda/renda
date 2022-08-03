# -*- coding: utf-8 -*-
import os
from glob import glob
from typing import Optional

import pytorch_lightning as pl

from renda.utils.timing import get_timestamp


def _get_exp_version(exp_name, exp_group) -> int:
    version_pattern = os.path.join(exp_group, exp_name, "v*")
    version_paths = glob(version_pattern)
    version_dirs = [_get_version_dir(path) for path in version_paths]
    versions = [_get_version(dir_) for dir_ in version_dirs]
    return max(versions) + 1 if len(versions) > 0 else 0


def _get_version_dir(version_path):
    return version_path.split(os.path.sep)[-1]


def _get_version(version_dir):
    underscore_index = version_dir.find("_")
    version_str = version_dir[1:underscore_index]
    version_int = int(version_str)
    return version_int


class Logger(pl.loggers.TensorBoardLogger):
    def __init__(
        self,
        exp_group: str = "results",
        exp_name: str = "experiment",
        exp_phase: str = "",
        exp_version: Optional[int] = None,
        exp_timestamp: Optional[str] = None,
    ) -> None:
        self.exp_group = exp_group
        self.exp_name = exp_name
        self.exp_phase = exp_phase

        if exp_version is None:
            self.exp_version = _get_exp_version(self.exp_name, self.exp_group)
        else:
            self.exp_version = exp_version

        if exp_timestamp is None:
            self.exp_timestamp = get_timestamp()
        else:
            self.exp_timestamp = exp_timestamp

        self.exp_version_dir = f"v{self.exp_version:02d}_{self.exp_timestamp}"

        if self.exp_phase == "":
            self.exp_path = os.path.join(
                self.exp_group,
                self.exp_name,
                self.exp_version_dir,
            )

            os.makedirs(self.exp_path)

            super().__init__(
                save_dir=self.exp_group,
                name=self.exp_name,
                version=self.exp_version_dir,
                log_graph=False,
                default_hp_metric=False,
            )
        else:
            self.exp_path = os.path.join(
                self.exp_group,
                self.exp_name,
                self.exp_version_dir,
                self.exp_phase,
            )

            os.makedirs(self.exp_path)

            super().__init__(
                save_dir=self.exp_group,
                name=self.exp_name,
                version=os.path.join(self.exp_version_dir, self.exp_phase),
                log_graph=False,
                default_hp_metric=False,
            )

        self.exp_pid = os.getpid()
        self.exp_ppid = os.getppid()


class LoggerBuilder:
    def __init__(
        self,
        exp_group: str = "results",
        exp_name: str = "experiment",
    ) -> None:
        self.exp_group = exp_group
        self.exp_name = exp_name
        self.exp_version = _get_exp_version(exp_name, exp_group)
        self.exp_timestamp = get_timestamp()

    def build(self, exp_phase: str = ""):
        return Logger(
            exp_group=self.exp_group,
            exp_name=self.exp_name,
            exp_phase=exp_phase,
            exp_version=self.exp_version,
            exp_timestamp=self.exp_timestamp,
        )
