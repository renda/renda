# -*- coding: utf-8 -*-
from renda.data.galaxy import GalaxyDataModule, GalaxyDataset

# from renda.data.lines import LinesDataModule, LinesDataset
from renda.data.vision_datamodules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
    SVHNDataModule,
)

__all__ = [
    "GalaxyDataModule",
    "GalaxyDataset",
    "CIFAR10DataModule",
    "FashionMNISTDataModule",
    "MNISTDataModule",
    "SVHNDataModule",
]
