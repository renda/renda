# -*- coding: utf-8 -*-
import random
from typing import Optional, Sequence, Union

import numpy as np
import torch


def get_min_seed() -> int:
    """
    Get smallest integer that may be used for seeding.

    Returns:
        int: Smallest integer that may be used for seeding.
    """
    return np.iinfo(np.uint32).min


def get_max_seed() -> int:
    """
    Get largest integer that may be used for seeding.

    Returns:
        int: Largest integer that may be used for seeding.
    """
    return np.iinfo(np.uint32).max


def is_seed(seed: int) -> bool:
    return (
        isinstance(seed, int)
        and not isinstance(seed, bool)
        and get_min_seed() <= seed <= get_max_seed()
    )


def process_seed(
    seed: Union[Optional[int], Sequence[int]],
    allow_sequence: bool = False,
    length: Optional[int] = None,
) -> Union[Optional[int], Sequence[int]]:
    if seed is None or is_seed(seed):
        return seed

    if not allow_sequence:
        # ------------------------------------------
        # We did not return although we should have
        # ------------------------------------------
        raise ValueError(
            f"Expected None or an int between {get_min_seed()} and "
            f"{get_max_seed()}. Got {seed}."
        )
    else:
        # ------------------------
        # Check sequence of seeds
        # ------------------------
        if (
            isinstance(seed, Sequence)  # Sequence ok?
            and (length is None or len(seed) == length)  # Length ok?
            and all(is_seed(s) for s in seed)  # Elements ok?
        ):
            return seed
        else:
            if length is None:
                of_length = ""
            else:
                of_length = f" of length {length}"

            raise ValueError(
                f"Expected one of these: (1) None; (2) an integer between "
                f"{get_min_seed()} and {get_max_seed()}; (3) a "
                f"sequence{of_length} where each element is an integer "
                f"between {get_min_seed()} and {get_max_seed()}. Got {seed}."
            )


def fix_seed(seed: int):
    return (seed - get_min_seed()) % (get_max_seed() - get_min_seed() + 1)


class temp_seed:
    """
    Context manager that temporarily seeds the random number generators from
    ``random``, ``numpy.random``, ``torch`` and ``torch.cuda``.

    Args:
        seed (Optional[int]): Integer between np.iinfo(np.uint32).min
        to np.iinfo(np.uint32).max or None. If None, `temp_seed` has no
        effect on any random number generator.

    Note:
        Use it *before training a model* to ensure reproduciblility, e.g.,
        when creating the model or the data set. Do not use it *during the
        training of a model*, e.g., in training, validation or testing loops.
    """

    def __init__(self, seed: Optional[int]) -> None:
        if seed is not None and not is_seed(seed):
            raise ValueError(
                f"Expected None or an int between {get_min_seed()} and "
                f"{get_max_seed()}. Got {seed}."
            )

        self._seed = seed

        self._random_state = None
        self._np_random_state = None
        self._cpu_rng_state = None
        self._gpu_rng_state_all = None

    def __enter__(self) -> None:  # pragma: no cover
        if self._seed is not None:
            # Remember current random states
            self._random_state = random.getstate()
            self._np_random_state = np.random.get_state()
            self._cpu_rng_state = torch.get_rng_state()
            self._gpu_rng_state_all = torch.cuda.get_rng_state_all()

            # Seed everything
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        if self._seed is not None:
            # Restore random states
            random.setstate(self._random_state)
            np.random.set_state(self._np_random_state)
            torch.set_rng_state(self._cpu_rng_state)
            torch.cuda.set_rng_state_all(self._gpu_rng_state_all)
