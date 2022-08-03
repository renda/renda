# -*- coding: utf-8 -*-
import logging
import os
import warnings
from contextlib import ExitStack, redirect_stdout
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl


class pl_no_output(ExitStack):
    def __init__(self) -> None:
        super().__init__()

        f = self.enter_context(open(os.devnull, "w"))
        self.enter_context(redirect_stdout(f))

        self.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")

        pl_logger = logging.getLogger("pytorch_lightning")
        pl_logger_level = pl_logger.level
        pl_logger.setLevel(np.iinfo(np.uint64).max)

        def pl_logger_reset_level(exc_type, exc_value, traceback):
            pl_logger.setLevel(pl_logger_level)

        self.push(pl_logger_reset_level)


def pl_get_profiler_class_by_name(name: str) -> Optional[Callable]:
    try:
        with pl_no_output():
            trainer = pl.Trainer(profiler=name)
            profiler_class = trainer.profiler.__class__
            del trainer
            return profiler_class

    except Exception:
        raise AttributeError(f"Unknown profiler name {name}.")
