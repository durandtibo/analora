r"""Contain functions to check ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["check_square_matrix"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


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
