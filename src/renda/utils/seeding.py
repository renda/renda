# -*- coding: utf-8 -*-
"""
Utilities to be used to
"""

import random
from typing import Optional, Sequence, Union

import numpy as np
import torch


def min_seed() -> int:
    """
    Returns the smallest integer that may be used as a seed.
    """
    return np.iinfo(np.uint32).min


def max_seed() -> int:
    """
    Returns the largest integer that may be used as a seed.
    """
    return np.iinfo(np.uint32).max


def is_seed(seed: int) -> bool:
    """
    Returns a boolean indicating whether the passed integer is a valid seed.

    Seeds must lie between :func:`min_seed() <min_seed>` and
    :func:`max_seed() <max_seed>`.
    """
    return (
        isinstance(seed, int)
        and not isinstance(seed, bool)
        and min_seed() <= seed <= max_seed()
    )


class temp_seed:
    """
    Context manager that temporarily seeds the random number generators of
    Python's ``random``, ``numpy.random``, ``torch`` and ``torch.cuda``.

    Args:
        seed (Optional[int]): Integer between :func:`min_seed()
            <min_seed>` and :func:`max_seed() <max_seed>` or None.
            If None, :func:`temp_seed(seed) <temp_seed>` has no effect on any
            of random number generator menioned above. Defaults to ``None``.

    Example:

    .. code-block:: python

        import torch

        from renda.utils.seeding import temp_seed

        a = torch.rand(1)
        with temp_seed(0):
            x = torch.rand(1)

        b = torch.rand(1)
        with temp_seed(0):
            y = torch.rand(1)

        print("a == b yields", a == b)
        print("x == y yields", x == y)

    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None and not is_seed(seed):
            raise ValueError(
                f"Expected None or an int between {min_seed()} and "
                f"{max_seed()}. Got {seed}."
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


def _fix_seed(seed: int):
    return (seed - min_seed()) % (max_seed() - min_seed() + 1)


def _process_seed(
    seed: Optional[Union[int, Sequence[int]]],
    allow_sequence: bool = False,
    length: Optional[int] = None,
) -> Optional[Union[int, Sequence[int]]]:
    if seed is None or is_seed(seed):
        return seed

    if not allow_sequence:
        # ------------------------------------------
        # We did not return although we should have
        # ------------------------------------------
        raise ValueError(
            f"Expected None or an int between {min_seed()} and "
            f"{max_seed()}. Got {seed}."
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
                f"{min_seed()} and {max_seed()}; (3) a "
                f"sequence{of_length} where each element is an integer "
                f"between {min_seed()} and {max_seed()}. Got {seed}."
            )
