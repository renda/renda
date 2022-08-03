# -*- coding: utf-8 -*-
from typing import Callable, Optional

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST

from renda.utils.seeding import _process_seed


class _VisionDataModule(LightningDataModule):
    def __init__(
        self,
        # --------
        # Dataset
        # --------
        root: str,
        num_train_samples: int,
        num_val_samples: Optional[int] = None,
        seed: int = 0,
        download: bool = False,
        # ----------
        # Transform
        # ----------
        grayscale: bool = False,
        zero_mean: bool = False,
        flatten: bool = False,
        # -----------
        # Dataloader
        # -----------
        batch_size: Optional[int] = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__()

        if not hasattr(self, "DATASET_CLASS"):
            raise NotImplementedError(
                f"_VisionModule subclass {self.__class__.__name__} must have "
                f"a static attribute DATASET_CLASS."
            )

        if not hasattr(self, "FULL_TRAIN_DATASET_SIZE"):
            raise NotImplementedError(
                f"_VisionModule subclass {self.__class__.__name__} must have "
                f"a static attribute FULL_TRAIN_DATASET_SIZE."
            )

        # --------
        # Dataset
        # --------
        self.root = root

        r = self.FULL_TRAIN_DATASET_SIZE
        if isinstance(num_train_samples, int) and 0 < num_train_samples <= r:
            self.num_train_samples = num_train_samples
            r -= num_train_samples
        else:
            raise ValueError(
                f"Expected positive integer <= {r}. Got {num_train_samples}."
            )

        if num_val_samples is None:
            self.num_val_samples = r
            r = 0
        elif isinstance(num_val_samples, int) and 0 <= num_val_samples <= r:
            self.num_val_samples = num_val_samples
            r -= num_val_samples
        else:
            raise ValueError(
                f"Expected non-negative integer <= {r}. Got {num_val_samples}."
            )

        self.num_remaining_samples = r

        self.seed = _process_seed(seed)
        self.download = download

        # ----------
        # Transform
        # ----------
        self.grayscale = grayscale
        self.zero_mean = zero_mean
        self.mean = None  # Set during setup()
        self.flatten = flatten

        transform = []
        transform.append(T.ToTensor())
        if self.grayscale:
            transform.append(T.Grayscale())
        self._initial_transform = T.Compose(transform)

        self._final_transform = None  # Set during setup()

        # -----------
        # Dataloader
        # -----------
        self._drop_last = drop_last
        self._common_dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
        }

        self._train_dataset = None
        self._train_dataloader = None
        self._val_dataset = None
        self._val_dataloader = None
        self._test_dataset = None
        self._test_dataloader = None

    def prepare_data(self):
        self.DATASET_CLASS(root=self.root, train=True, download=self.download)
        self.DATASET_CLASS(root=self.root, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        dataset = self.DATASET_CLASS(
            root=self.root, train=True, transform=self._initial_transform
        )

        lengths = [
            self.num_train_samples,
            self.num_val_samples,
            self.num_remaining_samples,
        ]

        train_dataset, val_dataset, _ = random_split(dataset, lengths, generator)
        del _  # Delete dataset of remaining samples

        self._set_final_transform(train_dataset)

        if stage == "fit" or stage is None:
            # HINT: Since 'train_dataset' and 'val_dataset' are no torchvision
            # datasets, they do not have a 'transform' attribute. However, they
            # are wrappers of the 'dataset' and draw samples from it. So we can
            # assign the 'transform' to 'full_dataset' to make 'train_dataset'
            # and 'val_dataset' use them.
            dataset.transform = self._final_transform

            # For random data loading
            self._train_dataset = train_dataset
            self._train_dataloader = DataLoader(
                dataset=self._train_dataset,
                shuffle=True,
                drop_last=self._drop_last,
                generator=generator.manual_seed(self.seed),  # !!! Reset seed
                **self._common_dataloader_kwargs,
            )

            # For sequential data loading
            self._val_dataset = val_dataset
            self._val_dataloader = DataLoader(
                dataset=self._val_dataset,
                shuffle=False,
                drop_last=False,
                **self._common_dataloader_kwargs,
            )

        if stage == "test" or stage is None:
            # In plain test mode, these are not needed anymore
            del train_dataset
            del val_dataset
            del dataset

            # Load test dataset
            self._test_dataset = self.DATASET_CLASS(
                self.root,
                train=False,
                transform=self._final_transform,
            )

            # For sequential data loading
            self._test_dataloader = DataLoader(
                dataset=self._test_dataset,
                shuffle=False,
                drop_last=False,
                **self._common_dataloader_kwargs,
            )

    def _set_final_transform(self, train_dataset: Dataset) -> Callable:
        transform = []
        transform.append(T.ToTensor())

        if self.grayscale:
            transform.append(T.Grayscale())

        if self.zero_mean:
            dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
            X = next(iter(dataloader))[0]  # Load all input data at once
            self.mean = X.mean(dim=0)
            std = torch.tensor(1.0)
            transform.append(T.Normalize(self.mean, std))
        else:
            X = train_dataset[0][0]  # Load first samples only
            self.mean = torch.zeros_like(X)

        if self.flatten:
            transform.append(T.Lambda(lambda X: X.flatten()))
            self.mean = self.mean.flatten()

        self._final_transform = T.Compose(transform)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


class CIFAR10DataModule(_VisionDataModule):
    DATASET_CLASS: Callable = CIFAR10
    FULL_TRAIN_DATASET_SIZE: int = 50000

    def __init__(
        self,
        # --------
        # Dataset
        # --------
        root: str,
        num_train_samples: int,
        num_val_samples: Optional[int] = None,
        seed: int = 0,
        download: bool = False,
        # ----------
        # Transform
        # ----------
        grayscale: bool = False,
        zero_mean: bool = False,
        flatten: bool = False,
        # -----------
        # Dataloader
        # -----------
        batch_size: Optional[int] = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__(
            # --------
            # Dataset
            # --------
            root=root,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            seed=seed,
            download=download,
            # ----------
            # Transform
            # ----------
            grayscale=grayscale,
            zero_mean=zero_mean,
            flatten=flatten,
            # -----------
            # Dataloader
            # -----------
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )


class FashionMNISTDataModule(_VisionDataModule):
    DATASET_CLASS: Callable = FashionMNIST
    FULL_TRAIN_DATASET_SIZE: int = 60000

    def __init__(
        self,
        # --------
        # Dataset
        # --------
        root: str,
        num_train_samples: int,
        num_val_samples: Optional[int] = None,
        seed: int = 0,
        download: bool = False,
        # ----------
        # Transform
        # ----------
        grayscale: bool = False,
        zero_mean: bool = False,
        flatten: bool = False,
        # -----------
        # Dataloader
        # -----------
        batch_size: Optional[int] = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__(
            # --------
            # Dataset
            # --------
            root=root,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            seed=seed,
            download=download,
            # ----------
            # Transform
            # ----------
            grayscale=grayscale,
            zero_mean=zero_mean,
            flatten=flatten,
            # -----------
            # Dataloader
            # -----------
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )


class MNISTDataModule(_VisionDataModule):
    DATASET_CLASS: Callable = MNIST
    FULL_TRAIN_DATASET_SIZE: int = 60000

    def __init__(
        self,
        # --------
        # Dataset
        # --------
        root: str,
        num_train_samples: int,
        num_val_samples: Optional[int] = None,
        seed: int = 0,
        download: bool = False,
        # ----------
        # Transform
        # ----------
        grayscale: bool = False,
        zero_mean: bool = False,
        flatten: bool = False,
        # -----------
        # Dataloader
        # -----------
        batch_size: Optional[int] = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__(
            # --------
            # Dataset
            # --------
            root=root,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            seed=seed,
            download=download,
            # ----------
            # Transform
            # ----------
            grayscale=grayscale,
            zero_mean=zero_mean,
            flatten=flatten,
            # -----------
            # Dataloader
            # -----------
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )


class SVHNDataModule(_VisionDataModule):
    class SVHNDataset(SVHN):
        def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
        ):
            super().__init__(
                root=root,
                split="train" if train else "test",
                transform=transform,
                target_transform=target_transform,
                download=download,
            )

    DATASET_CLASS: Callable = SVHNDataset
    FULL_TRAIN_DATASET_SIZE: int = 73257

    def __init__(
        self,
        # --------
        # Dataset
        # --------
        root: str,
        num_train_samples: int,
        num_val_samples: Optional[int] = None,
        seed: int = 0,
        download: bool = False,
        # ----------
        # Transform
        # ----------
        grayscale: bool = False,
        zero_mean: bool = False,
        flatten: bool = False,
        # -----------
        # Dataloader
        # -----------
        batch_size: Optional[int] = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__(
            # --------
            # Dataset
            # --------
            root=root,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            seed=seed,
            download=download,
            # ----------
            # Transform
            # ----------
            grayscale=grayscale,
            zero_mean=zero_mean,
            flatten=flatten,
            # -----------
            # Dataloader
            # -----------
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
