r"""Contain functions to check ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["check_same_shape", "check_square_matrix", "multi_isnan"]


from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def check_same_shape(arrays: Iterable[np.ndarray]) -> None:
    r"""Check if arrays have the same shape.

    Args:
        arrays: The arrays to check.

    Raises:
        RuntimeError: if the arrays have different shapes.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from analora.array import check_same_shape
    >>> check_same_shape([np.array([1, 0, 0, 1]), np.array([0, 1, 0, 1])])

    ```
    """
    shapes = {arr.shape for arr in arrays}
    if len(shapes) > 1:
        msg = f"arrays have different shapes: {shapes}"
        raise RuntimeError(msg)


def check_square_matrix(name: str, array: np.ndarray) -> None:
    r"""Check if the input array is a square matrix.

    Args:
        name: The name of the variable.
        array: The array to check.

    Raises:
        ValueError: if the array is not a square matrix.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from analora.array import check_square_matrix
    >>> check_square_matrix("var", np.ones((3, 3)))

    ```
    """
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = (
            f"Incorrect {name!r}. The array must be a square matrix but received an array of "
            f"shape {array.shape}"
        )
        raise ValueError(msg)


def multi_isnan(arrays: Sequence[np.ndarray]) -> np.ndarray:
    r"""Test element-wise for NaN for all input arrays and return result
    as a boolean array.

    Args:
        arrays: The input arrays to test. All the arrays must have the
            same shape.

    Returns:
        A boolean array. ``True`` where any array is NaN,
            ``False`` otherwise.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from analora.array import multi_isnan
    >>> mask = multi_isnan(
    ...     [np.array([1, 0, 0, 1, float("nan")]), np.array([1, float("nan"), 0, 1, 1])]
    ... )
    >>> mask
    array([False,  True, False, False,  True])

    ```
    """
    if len(arrays) == 0:
        msg = "'arrays' cannot be empty"
        raise RuntimeError(msg)
    mask = np.isnan(arrays[0])
    for arr in arrays[1:]:
        mask = np.logical_or(mask, np.isnan(arr))
    return mask
