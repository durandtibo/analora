r"""Contain random utility functions for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["rand_replace"]

from typing import Any

import numpy as np


def rand_replace(
    arr: np.ndarray, value: Any, prob: float = 0.5, rng: np.random.Generator | None = None
) -> np.ndarray:
    r"""Return an array that contains the same values as the input array,
    excepts some values are replaced by ``value``.

    Args:
        arr: The array with the original values.
        value: The value used to replace existing values.
        prob: The probability of value replacement.
            If the value is ``0.2``, it means each value as 20% chance
            to be replaced.
        rng: The random number generator used to decide which values
            are replaced or not. If ``None``, the default random
            number generator is used.

    Returns:
        The generated array.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from analora.array import rand_replace
    >>> rng = np.random.default_rng(42)
    >>> out = rand_replace(np.arange(10), value=-1, prob=0.4, rng=rng)

    ```
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.choice([True, False], size=arr.shape, p=[prob, 1.0 - prob])
    return np.where(mask, value, arr)
