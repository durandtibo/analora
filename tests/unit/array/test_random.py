from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from analora.array import rand_replace

##################################
#     Tests for rand_replace     #
##################################


def test_rand_replace_prob_0() -> None:
    assert objects_are_equal(rand_replace(np.arange(10), value=-1, prob=0.0), np.arange(10))


def test_rand_replace_empty() -> None:
    assert objects_are_equal(rand_replace(np.array([]), value=-1, prob=0.4), np.array([]))


def test_rand_replace_same_seed() -> None:
    assert objects_are_equal(
        rand_replace(np.arange(100), value=-1, prob=0.4, rng=np.random.default_rng(1)),
        rand_replace(np.arange(100), value=-1, prob=0.4, rng=np.random.default_rng(1)),
    )


def test_rand_replace_different_seeds() -> None:
    assert not objects_are_equal(
        rand_replace(np.arange(100), value=-1, prob=0.4, rng=np.random.default_rng(1)),
        rand_replace(np.arange(100), value=-1, prob=0.4, rng=np.random.default_rng(2)),
    )
