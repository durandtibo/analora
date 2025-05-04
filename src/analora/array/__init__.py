r"""Contain functions for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = [
    "check_same_shape",
    "check_square_matrix",
    "filter_range",
    "find_range",
    "multi_isnan",
    "nonnan",
    "rand_replace",
]

from analora.array.checking import check_same_shape, check_square_matrix, multi_isnan
from analora.array.filtering import filter_range, nonnan
from analora.array.random import rand_replace
from analora.array.range import find_range
