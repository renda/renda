# -*- coding: utf-8 -*-
import random

import numpy as np
import pytest
import torch

from renda.utils.seeding import (
    _fix_seed,
    _process_seed,
    is_seed,
    max_seed,
    min_seed,
    temp_seed,
)


def test_min_seed():
    assert min_seed() == np.iinfo(np.uint32).min


def test_max_seed():
    assert max_seed() == np.iinfo(np.uint32).max


def test_is_seed():
    assert is_seed(min_seed())
    assert is_seed(max_seed())

    # Too small or too large?
    assert not is_seed(min_seed() - 1)
    assert not is_seed(max_seed() + 1)

    # Simply returns False for non-integer types
    assert not is_seed(None)
    assert not is_seed("0")
    assert not is_seed("string")
    assert not is_seed([])

    # Tricky cases
    assert not is_seed(False)
    assert not is_seed(True)


def test_process_seed():
    assert _process_seed(min_seed()) == min_seed()
    assert _process_seed(max_seed()) == max_seed()

    seed = [0, 1, 2]

    assert _process_seed(seed, allow_sequence=True) == seed
    assert _process_seed(seed, allow_sequence=True, length=len(seed)) == seed

    with pytest.raises(ValueError):
        # Default: allow_sequence=False
        _process_seed(seed)
    with pytest.raises(ValueError):
        # Incorrect length
        _process_seed(seed, allow_sequence=True, length=len(seed) - 1)
    with pytest.raises(ValueError):
        # Sequence with invalid seed (None)
        _process_seed([0, None, 2], allow_sequence=True)


def test_fix_seed():
    seeds = [min_seed(), max_seed()]
    for seed in seeds:
        assert is_seed(seed)
        seed = _fix_seed(seed)
        assert is_seed(seed)

    seeds = [min_seed() - 1, max_seed() + 1]
    for seed in seeds:
        assert not is_seed(seed)
        seed = _fix_seed(seed)
        assert is_seed(seed)


def test_temp_seed_with_invalid_seed():
    seeds = [
        min_seed() - 1,
        max_seed() + 1,
        2 * max_seed(),
        "0",
        [0, 1, 2],
        False,
        True,
    ]

    for seed in seeds:
        with pytest.raises(ValueError):
            with temp_seed(seed):
                pass


def test_temp_seed_for_python_random():
    with temp_seed(0):
        a = random.random()
    with temp_seed(0):
        b = random.random()
    with temp_seed(1):
        c = random.random()
    with temp_seed(None):
        d = random.random()
    with temp_seed(None):
        e = random.random()

    random.seed(0)
    random.random()
    f = random.random()

    random.seed(0)
    random.random()
    with temp_seed(0):
        random.random()
    g = random.random()

    assert a == b
    assert a != c
    assert d != e
    assert f == g


def test_temp_seed_for_numpy_arrays():
    with temp_seed(0):
        a = np.random.rand(3, 3)
    with temp_seed(0):
        b = np.random.rand(3, 3)
    with temp_seed(1):
        c = np.random.rand(3, 3)
    with temp_seed(None):
        d = np.random.rand(3, 3)
    with temp_seed(None):
        e = np.random.rand(3, 3)

    np.random.seed(0)
    np.random.rand(3, 3)
    f = np.random.rand(3, 3)

    np.random.seed(0)
    np.random.rand(3, 3)
    with temp_seed(0):
        np.random.rand(3, 3)
    g = np.random.rand(3, 3)

    assert (a == b).all()
    assert not (a == c).all()
    assert not (d == e).all()
    assert (f == g).all()


def test_temp_seed_for_torch_tensors():
    with temp_seed(0):
        a = torch.rand(3, 3)
    with temp_seed(0):
        b = torch.rand(3, 3)
    with temp_seed(1):
        c = torch.rand(3, 3)
    with temp_seed(None):
        d = torch.rand(3, 3)
    with temp_seed(None):
        e = torch.rand(3, 3)

    torch.manual_seed(0)
    torch.rand(3, 3)
    f = torch.rand(3, 3)

    torch.manual_seed(0)
    torch.rand(3, 3)
    with temp_seed(0):
        torch.rand(3, 3)
    g = torch.rand(3, 3)

    assert (a == b).all()
    assert not (a == c).all()
    assert not (d == e).all()
    assert (f == g).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_temp_seed_for_torch_cuda_tensors():
    with temp_seed(0):
        a = torch.rand(3, 3, device="cuda")
    with temp_seed(0):
        b = torch.rand(3, 3, device="cuda")
    with temp_seed(1):
        c = torch.rand(3, 3, device="cuda")
    with temp_seed(None):
        d = torch.rand(3, 3, device="cuda")
    with temp_seed(None):
        e = torch.rand(3, 3, device="cuda")

    torch.cuda.manual_seed_all(0)
    torch.rand(3, 3, device="cuda")
    f = torch.rand(3, 3, device="cuda")

    torch.cuda.manual_seed_all(0)
    torch.rand(3, 3, device="cuda")
    with temp_seed(0):
        torch.rand(3, 3, device="cuda")
    g = torch.rand(3, 3, device="cuda")

    assert (a == b).all()
    assert not (a == c).all()
    assert not (d == e).all()
    assert (f == g).all()
