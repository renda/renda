# -*- coding: utf-8 -*-
from typing import Any

from pytorch_lightning import LightningModule


class LogTable:
    def __init__(self, lightning_module: LightningModule) -> None:
        self._lightning_module = lightning_module
        self._print = lightning_module.print

        self._column_names = lightning_module.LOG_TABLE_COLUMN_NAMES
        self._min_column_width = lightning_module.LOG_TABLE_MIN_COLUMN_WIDTH

        self._num_columns = len(self._column_names)
        self._column_widths = []
        for name in self._column_names:
            width = max(len(name), self._min_column_width)
            self._column_widths.append(width)

        self._hline = {}
        self._hline["-"] = "|-------|-------|--------------"
        self._hline["="] = "|=======|=======|=============="
        self._table_header = "| Epoch | Batch | Duration     "
        self._marker = len(self._table_header)

        for name, width in zip(self._column_names, self._column_widths):
            self._hline["-"] = f"{self._hline['-']}|-{'-' * width}-"
            self._hline["="] = f"{self._hline['=']}|={'=' * width}="
            self._table_header = f"{self._table_header}| {name.ljust(width)} "

        self._print_header_called_once = False
        self._before_training = True

    def print_header(self) -> None:
        if self._print_header_called_once:
            self._before_training = False
        else:
            self._print_header_called_once = True

        if self._before_training:
            start_index = self._marker
        else:
            start_index = 0

        self._print(self._table_header[start_index:], end="|\n")

    def print_hline(self, style: str = "-") -> None:
        if self._before_training:
            start_index = self._marker
        else:
            start_index = 0

        self._print(self._hline[style][start_index:], end="|\n")

    def print_entry(self, *args: Any, comment: str = "") -> None:
        if self._before_training:
            losses = args
        else:
            losses = args[3:]
            epoch, batch, duration = args[0:3]

            epoch = " " * 5 if epoch is None else f"{epoch:5d}"
            batch = " " * 5 if batch is None else f"{batch:5d}"
            duration = " " * 12 if duration is None else f"{duration}"
            self._print(f"| {epoch} | {batch} | {duration} ", end="")

        for ii, loss in enumerate(losses):
            if ii < self._num_columns:
                column_width = self._column_widths[ii]
            else:
                column_width = self._min_column_width

            if loss is None:
                loss = self._min_column_width * " "
            else:
                loss = f"{loss:.6f}"

            loss = loss.rjust(column_width)
            self._print(f"| {loss} ", end="")

        self._print(f"| {comment}")
