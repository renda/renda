# -*- coding: utf-8 -*-
import math
from typing import Callable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from renda.utils.seeding import _process_seed, temp_seed
from renda.validation.plot_functions import plot_image_grids


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


class LinesDataset(TensorDataset):
    """
    Transform all leaf objects of one or more dictionaries.

    Args:
        offset_variation: .
        transforms (Sequence[Callable], optional): Transforms to apply to the
        leaf objects of each dictionary passed. Defaults to [].

    Note:
        If you wish to obtain (deep) copies, you need to include the
        appropriate transforms. E.g., in the example below, the NumPy arrays
        stored in ``a`` and tensors stored in ``b`` still share the same
        memory.
    """

    def __init__(
        self,
        # --------
        # General
        # --------
        num_samples_per_class: int = 1000,
        num_classes: int = 6,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 0,
        # -----------------
        # Image properties
        # -----------------
        image_size: int = 16,
        gamma: float = 10.0,
        min_intensity: float = 0.0,
        max_intensity: float = 1.0,
        gaussian_noise: float = 0.1,
        salt_pepper_noise: float = 0.1,
        # ---------------------
        # Variation properties
        # ---------------------
        offset_variation: float = 0.9,
        angle_variation: float = 0.9,
        gamma_variation: float = 1.0,
        intensity_variation: float = 0.9,
        gaussian_noise_variation: float = 1.0,
    ):
        # --------
        # General
        # --------
        self.num_classes = _process_num_classes(num_classes)
        self.num_samples_per_class = _process_num_samples_per_class(
            num_samples_per_class, self.num_classes
        )

        self.num_samples = self.num_classes * self.num_samples_per_class

        self.transform = transform
        self.target_transform = target_transform

        self.seed = _process_seed(seed)

        # -----------------
        # Image properties
        # -----------------
        if isinstance(image_size, int) and image_size >= 4:
            self.image_size = image_size
        else:
            raise ValueError(
                f"Expected image_size to be an integer >= 4. Got {image_size}."
            )

        if isinstance(gamma, float) and gamma >= 1.0:
            self.gamma = gamma
        else:
            raise ValueError(f"Expected gamma >= 1.0. Got {gamma}")

        if (
            isinstance(min_intensity, float)
            and isinstance(max_intensity, float)
            and 0.0 <= min_intensity < 1.0
            and 0.0 < max_intensity <= 1.0
            and min_intensity < max_intensity
        ):
            self.min_intensity = min_intensity
            self.max_intensity = max_intensity
        else:
            raise ValueError(
                f"Expected 0.0 <= min_intensity < 1.0 and 0.0 < max_intensity "
                f"<= 1.0 such that min_intensity < max_intensity. Got "
                f"min_intensity {min_intensity} and max_intensity "
                f"{max_intensity}"
            )

        if isinstance(gaussian_noise, float) and gaussian_noise >= 0.0:
            self.gaussian_noise = gaussian_noise
        else:
            raise ValueError(f"Expected gaussian_noise >= 0.0. Got {gaussian_noise}.")

        if isinstance(salt_pepper_noise, float) and 0.0 <= salt_pepper_noise <= 1.0:
            self.salt_pepper_noise = salt_pepper_noise
        else:
            raise ValueError(
                f"Expected 0.0 <= salt_pepper_noise <= 1.0. Got {salt_pepper_noise}."
            )

        # ---------------------
        # Variation properties
        # ---------------------
        if isinstance(offset_variation, float) and 0.0 <= offset_variation <= 1.0:
            self.offset_variation = offset_variation
            self._offset_variation = offset_variation * 0.5 * image_size
        else:
            raise ValueError(
                f"Expected 0.0 <= offset_variation <= 1.0. Got {offset_variation}."
            )

        angle_per_class = torch.linspace(0, math.pi, self.num_classes + 1)

        if isinstance(angle_variation, float) and 0.0 <= angle_variation <= 1.0:
            self.angle_variation = angle_variation
            self._angle_variation = angle_variation * 0.5 * angle_per_class[1]
        else:
            raise ValueError(
                f"Expected 0.0 <= angle_variation <= 1.0. Got {angle_variation}."
            )

        if isinstance(gamma_variation, float) and 0.0 <= gamma_variation <= 1.0:
            self.gamma_variation = gamma_variation
            self._gamma_variation = gamma_variation * max(0.0, self.gamma - 1.0)
        else:
            raise ValueError(
                f"Expected 0.0 <= gamma_variation <= 1.0. Got {gamma_variation}."
            )

        if isinstance(intensity_variation, float) and 0.0 <= intensity_variation <= 1.0:
            self.intensity_variation = intensity_variation
            delta_intensity = self.max_intensity - self.min_intensity
            self._intensity_variation = intensity_variation * 0.5 * delta_intensity
        else:
            raise ValueError(
                f"Expected 0.0 <= intensity_variation <= 1.0. "
                f"Got {intensity_variation}."
            )

        if (
            isinstance(gaussian_noise_variation, float)
            and 0.0 <= gaussian_noise_variation <= 1.0
        ):
            self.gaussian_noise_variation = gaussian_noise_variation
            self._gaussian_noise_variation = gaussian_noise_variation * gaussian_noise
        else:
            ValueError(
                f"Expected 0.0 <= gaussian_noise_variation <= 1.0. "
                f"Got {gaussian_noise_variation}."
            )

        with temp_seed(self.seed):
            values = torch.arange(image_size, dtype=torch.float32) - 0.5 * image_size
            x, y = torch.meshgrid(values, values)
            xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)

            X = []
            Y = []

            angle = torch.empty(1)
            offset = torch.empty(1)
            gamma = torch.empty(1)
            min_ = torch.empty(1)
            max_ = torch.empty(1)
            gaussian_noise = torch.empty(1)

            for class_ in range(self.num_classes):
                for i in range(self.num_samples_per_class[class_]):

                    offset.uniform_(-self._offset_variation, self._offset_variation)

                    angle.uniform_(-self._angle_variation, self._angle_variation)
                    angle.add_(angle_per_class[class_])

                    gamma.uniform_(-self._gamma_variation, 0.0)
                    gamma.add_(self.gamma)

                    min_.uniform_(0.0, self._intensity_variation)
                    min_.add_(self.min_intensity)
                    max_.uniform_(-self._intensity_variation, 0.0)
                    max_.add_(self.max_intensity)

                    gaussian_noise.uniform_(-self._gaussian_noise_variation, 0.0)
                    gaussian_noise.add_(self.gaussian_noise)

                    n = torch.cat([-torch.sin(angle), torch.cos(angle)])

                    X_ = ((xy + offset * n) @ n).abs()
                    X_ = 1.0 - X_ / X_.max()
                    X_ = X_.pow(gamma)
                    X_ = X_ * (max_ - min_) + min_

                    X_ = X_ + gaussian_noise * torch.randn_like(X_)
                    X_ = X_.clamp(0.0, 1.0)

                    if self.salt_pepper_noise > 0.0:
                        salt = torch.rand_like(X_) < 0.5 * self.salt_pepper_noise
                        pepper = torch.rand_like(X_) < 0.5 * self.salt_pepper_noise
                        X_[salt] = 0.0
                        X_[pepper] = 1.0

                    X_ = X_.permute([1, 0]).view(1, 1, image_size, image_size)
                    Y_ = torch.tensor([class_])

                    X.append(X_)
                    Y.append(Y_)

            self.X = torch.cat(X)
            self.Y = torch.cat(Y)

        super().__init__(self.X, self.Y)

    def plot(self) -> None:
        image_grid = plot_image_grids(self.X, self.Y, num_columns=16)
        plt.figure()
        plt.imshow(image_grid.permute([1, 2, 0]))


class LinesDataModule(LightningDataModule):
    def __init__(
        self,
        # ---------
        # Datasets
        # ---------
        num_train_samples_per_class: Union[int, Sequence[int]],
        num_val_samples_per_class: Union[int, Sequence[int]] = 0,
        num_classes: int = 6,
        # ------------------
        # Images properties
        # ------------------
        image_size: int = 16,
        gamma: float = 10.0,
        min_intensity: float = 0.0,
        max_intensity: float = 1.0,
        gaussian_noise: float = 0.1,
        salt_pepper_noise: float = 0.05,
        flatten: bool = False,
        # ----------------------------
        # Images variation properties
        # ----------------------------
        offset_variation: float = 0.9,
        angle_variation: float = 0.9,
        gamma_variation: float = 1.0,
        intensity_variation: float = 0.9,
        gaussian_noise_variation: float = 1.0,
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

        # ------------------
        # Images properties
        # ------------------
        self.image_size = image_size
        self.gamma = gamma
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.gaussian_noise = gaussian_noise
        self.salt_pepper_noise = salt_pepper_noise
        self.flatten = flatten

        # ----------------------------
        # Images variation properties
        # ----------------------------
        self.offset_variation = offset_variation
        self.angle_variation = angle_variation
        self.gamma_variation = gamma_variation
        self.intensity_variation = intensity_variation
        self.gaussian_noise_variation = gaussian_noise_variation

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
            self._train_dataset = LinesDataset(
                num_samples_per_class=self._num_train_samples_per_class,
                num_classes=self._num_classes,
                image_size=self.image_size,
                gamma=self.gamma,
                min_intensity=self.min_intensity,
                max_intensity=self.max_intensity,
                gaussian_noise=self.gaussian_noise,
                salt_pepper_noise=self.salt_pepper_noise,
                flatten=self.flatten,
                offset_variation=self.offset_variation,
                angle_variation=self.angle_variation,
                gamma_variation=self.gamma_variation,
                intensity_variation=self.intensity_variation,
                gaussian_noise_variation=self.gaussian_noise_variation,
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
            self._val_dataset = LinesDataset(
                num_samples_per_class=self._num_val_samples_per_class,
                num_classes=self._num_classes,
                image_size=self.image_size,
                gamma=self.gamma,
                min_intensity=self.min_intensity,
                max_intensity=self.max_intensity,
                gaussian_noise=self.gaussian_noise,
                salt_pepper_noise=self.salt_pepper_noise,
                flatten=self.flatten,
                offset_variation=self.offset_variation,
                angle_variation=self.angle_variation,
                gamma_variation=self.gamma_variation,
                intensity_variation=self.intensity_variation,
                gaussian_noise_variation=self.gaussian_noise_variation,
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


dataset = LinesDataset()
dataset.plot()
