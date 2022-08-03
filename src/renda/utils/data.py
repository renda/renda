# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Sequence, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.nn import Module
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)


def convert_to_sequential_dataloader(dataloader: DataLoader) -> DataLoader:
    """
    Get a copy of ``dataloader`` that loads the data sequentially.
    """
    return DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_sampler.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=False,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
    )


def transform_dataloaders(
    model: Module,
    *dataloaders: DataLoader,
) -> Union[DataLoader, Tuple[DataLoader, ...]]:
    dataloaders_ = []
    for dataloader in dataloaders:
        dataloader = _transform_dataloader(model, dataloader)
        dataloaders_.append(dataloader)

    if len(dataloaders_) == 1:
        return dataloaders_[0]
    else:
        return tuple(dataloaders_)


def _transform_dataloader(model: Module, dataloader: DataLoader) -> DataLoader:
    if dataloader is None:
        return None

    elif isinstance(dataloader, DataLoader):
        # Continued below, transform dataloader
        batch_sampler_builder, is_sequential = _get_batch_sampler_builder(
            dataloader.batch_sampler
        )

    else:
        raise TypeError(
            f"Expected None or torch.utils.data.DataLoader. Got {dataloader}."
        )

    # ------------------
    # Transform dataset
    # ------------------
    if not is_sequential:
        # To preserve the "natural" order of the internal dataset
        dataloader = convert_to_sequential_dataloader(dataloader)

    Z = []
    T_dict = defaultdict(list)

    for batch in dataloader:
        if isinstance(batch, Sequence):
            # Assume that the model interface satisfies the following two
            # conditions: (1) All tensors needed to compute the output appear
            # in the required order at the beginning of the sequence. (2) The
            # model accepts unused *args so that the remaining tensors of
            # the sequence are silently ignored.
            Z_ = model(*batch)
        else:
            Z_ = model(batch)

        # Assume that only the first tensor has been transformed, and that its
        # transformed version is also the first return value.
        if isinstance(Z_, Sequence):
            Z_ = Z_[0]
        Z.append(Z_)
        if isinstance(batch, Sequence):
            # Collect all other tensors provided. Enumerate to keep track of
            # the position of each tensor in the given sequence.
            for position, T_ in enumerate(batch[1:]):
                T_dict[position].append(T_)

    Z = torch.cat(Z)

    T_list = []
    for position in range(len(T_dict)):
        # Concatenate per position
        T_cat = torch.cat(T_dict[position])
        # The "position key" becomes the list position
        T_list.append(T_cat)

    dataset = TensorDataset(Z, *T_list)

    # ---------------------------------------------
    # Create and return the transformed dataloader
    # ---------------------------------------------
    batch_sampler = batch_sampler_builder(dataset)

    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
    )


def _get_batch_sampler_builder(batch_sampler):
    """
    Future place to handle different batch samplers.

    Currently, only torch.utils.data.BatchSampler is supported.
    """
    if isinstance(batch_sampler, BatchSampler):
        sampler_builder, is_sequential = _get_sampler_builder(batch_sampler.sampler)
        batch_size = batch_sampler.batch_size
        drop_last = batch_sampler.drop_last

        def batch_sampler_builder(dataset):
            sampler = sampler_builder(dataset)
            return BatchSampler(sampler, batch_size, drop_last)

    else:
        raise TypeError(
            f"Expected torch.utils.data.BatchSampler. Got {batch_sampler} "
            f"of class {batch_sampler.__class__}."
        )

    return batch_sampler_builder, is_sequential


def _get_sampler_builder(sampler):
    """
    Future place to handle different samplers.

    Currently, only torch.utils.data.SequentialSampler and
    torch.utils.data.RandomSampler are supported.
    """
    if isinstance(sampler, SequentialSampler):

        def sampler_builder(dataset):
            return SequentialSampler(dataset)

        is_sequential = True

    elif isinstance(sampler, RandomSampler):
        generator = sampler.generator
        if isinstance(generator, torch.Generator):
            generator.manual_seed(generator.initial_seed())

        def sampler_builder(dataset):
            return RandomSampler(dataset, generator=generator)

        is_sequential = False

    else:
        raise TypeError(
            f"Expected torch.utils.data.SequentialSampler or "
            f"torch.utils.data.RandomSampler. Got {sampler} of class "
            f"{sampler.__class__}."
        )

    return sampler_builder, is_sequential


def transform_datamodule(model, datamodule):
    if datamodule is None:
        return None

    elif isinstance(datamodule, LightningDataModule):
        train_dataloader = datamodule.train_dataloader()
        train_dataloader = transform_dataloaders(model, train_dataloader)
        val_dataloader = datamodule.val_dataloader()
        val_dataloader = transform_dataloaders(model, val_dataloader)

        # TODO: Extend this ad hoc datamodule to work in more general contexts.
        # Perhaps we adapt _DataModuleWrapper to this use case.

        datamodule = LightningDataModule()
        setattr(datamodule, "train_dataloader", lambda: train_dataloader)
        setattr(datamodule, "val_dataloader", lambda: val_dataloader)

    else:
        raise TypeError(
            f"Expected {LightningDataModule().__class__}. Got {datamodule} of "
            f"class {datamodule.__class__}."
        )

    return datamodule
