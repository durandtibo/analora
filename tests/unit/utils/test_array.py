from __future__ import annotations

import numpy as np
import pytest

from analora.utils.array import filter_range, nonnan

##################################
#     Tests for filter_range     #
##################################


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_filter_range(dtype: np.dtype) -> None:
    assert np.array_equal(
        filter_range(np.arange(10, dtype=dtype), xmin=1, xmax=5),
        np.array([1, 2, 3, 4, 5], dtype=dtype),
    )


def test_filter_range_inf() -> None:
    assert np.array_equal(
        filter_range(np.arange(10), xmin=float("-inf"), xmax=float("inf")),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )


def test_filter_range_empty() -> None:
    assert np.array_equal(filter_range(np.array([]), xmin=1, xmax=5), np.array([]))


############################
#     Tests for nonnan     #
############################


def test_nonnan_empty() -> None:
    assert np.array_equal(nonnan(np.array([])), np.array([]))


def test_nonnan_1d() -> None:
    assert np.array_equal(
        nonnan(np.array([1, 2, float("nan"), 5, 6])), np.array([1.0, 2.0, 5.0, 6.0])
    )


def test_nonnan_2d() -> None:
    assert np.array_equal(
        nonnan(np.array([[1, 2, float("nan")], [4, 5, 6]])), np.array([1.0, 2.0, 4.0, 5.0, 6.0])
    )
