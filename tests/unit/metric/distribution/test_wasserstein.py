from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from coola import objects_are_allclose

from analora.metric import wasserstein_distance
from analora.testing import scipy_available

##########################################
#     Tests for wasserstein_distance     #
##########################################


@scipy_available
def test_wasserstein_distance_same() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
        ),
        {"count": 5, "wasserstein_distance": 0.0},
    )


@scipy_available
def test_wasserstein_distance_perfect_positive_correlation_2d() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([[1, 2, 3], [4, 5, 6]]), v_values=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {"count": 6, "wasserstein_distance": 0.0},
    )


@scipy_available
def test_wasserstein_distance_different() -> None:
    assert objects_are_allclose(
        wasserstein_distance(u_values=np.array([0, 1, 3]), v_values=np.array([5, 6, 8])),
        {"count": 3, "wasserstein_distance": 5.0},
    )


@scipy_available
def test_wasserstein_distance_empty() -> None:
    assert objects_are_allclose(
        wasserstein_distance(u_values=np.array([]), v_values=np.array([])),
        {"count": 0, "wasserstein_distance": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_wasserstein_distance_prefix_suffix() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([1, 2, 3, 4, 5]),
            v_values=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_wasserstein_distance_suffix": 0.0},
    )


@scipy_available
def test_wasserstein_distance_nan_omit() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
            v_values=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
            nan_policy="omit",
        ),
        {"count": 4, "wasserstein_distance": 0.0},
    )


@scipy_available
def test_wasserstein_distance_omit_u_values() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([1, 2, 3, 4, 5, float("nan")]),
            v_values=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="omit",
        ),
        {"count": 5, "wasserstein_distance": 0.0},
    )


@scipy_available
def test_wasserstein_distance_omit_v_values() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([1, 2, 3, 4, 5, 0]),
            v_values=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="omit",
        ),
        {"count": 5, "wasserstein_distance": 0.0},
    )


@scipy_available
def test_wasserstein_distance_nan_propagate() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
            v_values=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
        ),
        {"count": 7, "wasserstein_distance": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_wasserstein_distance_nan_propagate_u_values() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([float("nan"), 2, 3, 4, 5, 6]),
            v_values=np.array([1, 2, 3, 4, 5, 6]),
        ),
        {"count": 6, "wasserstein_distance": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_wasserstein_distance_nan_propagate_v_values() -> None:
    assert objects_are_allclose(
        wasserstein_distance(
            u_values=np.array([1, 2, 3, 4, 5, 6]),
            v_values=np.array([1, 2, 3, 4, 5, float("nan")]),
        ),
        {"count": 6, "wasserstein_distance": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_wasserstein_distance_nan_raise() -> None:
    with pytest.raises(ValueError, match="'u_values' contains at least one NaN value"):
        wasserstein_distance(
            u_values=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            v_values=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="raise",
        )


@scipy_available
def test_wasserstein_distance_nan_raise_u_values() -> None:
    with pytest.raises(ValueError, match="'u_values' contains at least one NaN value"):
        wasserstein_distance(
            u_values=np.array([float("nan"), 2, 3, 4, 5]),
            v_values=np.array([1, 2, 3, 4, 5]),
            nan_policy="raise",
        )


@scipy_available
def test_wasserstein_distance_nan_raise_v_values() -> None:
    with pytest.raises(ValueError, match="'v_values' contains at least one NaN value"):
        wasserstein_distance(
            u_values=np.array([1, 2, 3, 4, 5]),
            v_values=np.array([1, 2, 3, 4, float("nan")]),
            nan_policy="raise",
        )


@patch("analora.utils.imports.is_scipy_available", lambda: False)
def test_wasserstein_distance_no_scipy() -> None:
    with pytest.raises(RuntimeError, match="'scipy' package is required but not installed."):
        wasserstein_distance(u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5]))
