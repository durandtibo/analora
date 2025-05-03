r"""Implement filtering utility functions for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["filter_range", "nonnan"]


import numpy as np


def filter_range(array: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    r"""Filter in the values in a given range.

    Args:
        array: The input array.
        xmin: The lower bound of the range.
        xmax: The upper bound of the range.

    Returns:
        A 1-d array with only the values in the given range.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from analora.array import filter_range
    >>> out = filter_range(np.arange(10), xmin=-1, xmax=5)
    >>> out
    array([0, 1, 2, 3, 4, 5])

    ```
    """
    return np.extract(np.logical_and(xmin <= array, array <= xmax), array)


def nonnan(array: np.ndarray) -> np.ndarray:
    r"""Return the non-NaN values of an array.

    Args:
        array: The input array.

    Returns:
        A 1d array with the non-NaN values of the input array.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from analora.array import nonnan
    >>> nonnan(np.asarray([1, 2, float("nan"), 5, 6]))
    array([1., 2., 5., 6.])
    >>> nonnan(np.asarray([[1, 2, float("nan")], [4, 5, 6]]))
    array([1., 2., 4., 5., 6.])

    ```
    """
    mask = np.isnan(array)
    return array[~mask]
