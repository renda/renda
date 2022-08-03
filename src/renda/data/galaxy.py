# -*- coding: utf-8 -*-
import math
from typing import Callable, Optional, Sequence, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from renda.utils.seeding import _process_seed, temp_seed
from renda.validation.plot_functions import plot_features


def _process_num_classes(num_classes: int) -> int:
    if not (isinstance(num_classes, int) and num_classes >= 2):
        raise ValueError(f"Expected integer >= 2. Got {num_classes}.")

    return num_classes


def _process_num_samples_per_class(
    num_samples_per_class: Union[int, Sequence[int]],
    num_classes: int,
) -> Union[int, Sequence[int]]:
    if isinstance(num_samples_per_class, int) and num_samples_per_class > 0:
        return [num_samples_per_class] * num_classes
    elif (
        isinstance(num_samples_per_class, Sequence)
        and len(num_samples_per_class) == num_classes
        and all(isinstance(e, int) for e in num_samples_per_class)
        and all(e > 0 for e in num_samples_per_class)
    ):
        return num_samples_per_class
    else:
        raise ValueError(
            f"Expected positive integer or a sequence of {num_classes} "
            f"(num_classes) positive integers. Got {num_samples_per_class}."
        )


class GalaxyDataset(TensorDataset):
    def __init__(
        self,
        num_samples_per_class: Union[int, Sequence[int]],
        num_classes: int = 3,
        num_rotations: float = 1.0,
        density_peak: float = 0.3,
        noise_sigma: float = 1.0,
        seed: int = 0,
    ) -> None:
        """
        Galaxy

        Args:
            num_samples_per_class (Union[int, Sequence[int]]): DESCRIPTION.
            num_classes (int, optional): Number of classes where each class is
                a distinct arm of the galaxy dataset. Defaults to 3.
            num_rotations (float, optional): Number of full rotations of each
                arm of the galaxy dataset. Defaults to 1.0.
            density_peak (float, optional): DESCRIPTION. Defaults to 0.3.
            noise_sigma (float, optional): DESCRIPTION. Defaults to 1.0.
            seed (int, optional): DESCRIPTION. Defaults to 0.
             (TYPE): DESCRIPTION.

        Returns:
            None: DESCRIPTION.

        """
        self.num_classes = _process_num_classes(num_classes)
        self.num_samples_per_class = _process_num_samples_per_class(
            num_samples_per_class, self.num_classes
        )

        self.num_rotations = num_rotations
        self._max_rotation_angle = 2.0 * math.pi * self.num_rotations
        self.density_peak = density_peak
        self.noise_sigma = noise_sigma
        self.seed = _process_seed(seed)

        with temp_seed(self.seed):
            phi_per_class = torch.linspace(0, 2.0 * math.pi, self.num_classes + 1)

            X = []
            Y = []

            for class_ in range(self.num_classes):
                n = self.num_samples_per_class[class_]

                t = []
                t_counter = 0
                t_upper_bound = self._max_rotation_angle
                mu = self.density_peak * self._max_rotation_angle
                noise_sigma = self.density_peak * 0.5 * self._max_rotation_angle

                while t_counter < n:
                    t_ = mu + noise_sigma * torch.randn(2 * n)
                    t_ = t_[(0.0 <= t_).logical_and(t_ <= t_upper_bound)]
                    t_counter += t_.numel()
                    t.append(t_)

                t = torch.cat(t)
                t = t[0:n]

                noisy_phi = []
                noisy_phi_counter = 0
                noisy_phi_bound = math.pi / self.num_classes

                while noisy_phi_counter < self.num_samples_per_class[class_]:
                    noisy_phi_ = torch.randn(self.num_samples_per_class[class_] * 2)
                    noisy_phi_ *= self.noise_sigma * noisy_phi_bound
                    noisy_phi_ = noisy_phi_[noisy_phi_.abs() <= noisy_phi_bound]
                    noisy_phi_counter += noisy_phi_.numel()
                    noisy_phi.append(noisy_phi_)

                noisy_phi = torch.cat(noisy_phi)
                noisy_phi = noisy_phi[0 : self.num_samples_per_class[class_]]
                noisy_phi = noisy_phi + phi_per_class[class_]

                X_ = torch.Tensor(2, self.num_samples_per_class[class_])
                X_[0, :] = t * torch.cos(t + noisy_phi)
                X_[1, :] = t * torch.sin(t + noisy_phi)
                X_ = X_.t() / self._max_rotation_angle
                X.append(X_)

                Y_ = (class_ + 1) * torch.ones(self.num_samples_per_class[class_], 1)
                Y.append(Y_)

            self.X = torch.cat(X)
            self.Y = torch.cat(Y)

        super().__init__(self.X, self.Y)

    def plot(self, show_figure: bool = True) -> None:
        return plot_features(self.X, self.Y, show_figures=show_figure)


class GalaxyDataModule(LightningDataModule):
    def __init__(
        self,
        # ---------
        # Datasets
        # ---------
        num_train_samples_per_class: Union[int, Sequence[int]],
        num_val_samples_per_class: Union[int, Sequence[int]] = 0,
        num_classes: int = 3,
        num_rotations: Union[float, int] = 1.0,
        noise_sigma: Union[float, int] = 0.3,
        # -----
        # Seed
        # -----
        seed: int = 0,
        # -----------
        # Dataloader
        # -----------
        batch_size: Optional[int] = 1,
        drop_last: bool = True,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__()

        # ---------
        # Datasets
        # ---------
        self._num_classes = _process_num_classes(num_classes)
        self._num_train_samples_per_class = _process_num_samples_per_class(
            num_train_samples_per_class, self._num_classes
        )
        self._num_val_samples_per_class = _process_num_samples_per_class(
            num_val_samples_per_class, self._num_classes
        )
        self._num_rotations = num_rotations
        self._noise_sigma = noise_sigma

        # -----
        # Seed
        # -----
        self._seed = _process_seed(seed)

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

    def setup(self, stage: Optional[str] = None):
        num_train_samples = sum(self._num_train_samples_per_class)
        if num_train_samples > 0:
            self._train_dataset = GalaxyDataset(
                num_samples_per_class=self._num_train_samples_per_class,
                num_classes=self._num_classes,
                num_rotations=self._num_rotations,
                noise_sigma=self._noise_sigma,
                seed=self._seed,
            )

            # Independent RNG for the train_dataloader
            generator = torch.Generator()
            generator.manual_seed(self._seed)

            # For random data loading
            self._train_dataloader = DataLoader(
                dataset=self._train_dataset,
                shuffle=True,
                drop_last=self._drop_last,
                generator=generator,
                **self._common_dataloader_kwargs,
            )

        num_val_samples = sum(self._num_val_samples_per_class)
        if num_val_samples > 0:
            self._val_dataset = GalaxyDataset(
                num_samples_per_class=self._num_val_samples_per_class,
                num_classes=self._num_classes,
                num_rotations=self._num_rotations,
                noise_sigma=self._noise_sigma,
                seed=self._seed + 1,
            )

            # For sequential data loading
            self._val_dataloader = DataLoader(
                dataset=self._val_dataset,
                shuffle=False,
                drop_last=False,
                **self._common_dataloader_kwargs,
            )

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
