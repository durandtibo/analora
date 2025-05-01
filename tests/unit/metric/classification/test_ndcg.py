from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from coola import objects_are_allclose

from analora.metric import ndcg
from analora.testing import scipy_available, sklearn_available

##########################
#     Tests for ndcg     #
##########################


@sklearn_available
@scipy_available
def test_ndcg_correct() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, 1]]),
            y_score=np.array([[2.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
        ),
        {"count": 4, "ndcg": 1.0},
    )


@sklearn_available
@scipy_available
def test_ndcg_different() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[10, 0, 0, 1, 5], [10, 0, 0, 1, 5]]),
            y_score=np.array([[0.1, 0.2, 0.3, 4, 70], [0.05, 1.1, 1.0, 0.5, 0.0]]),
        ),
        {"count": 2, "ndcg": 0.5946871178793418},
    )


@sklearn_available
@scipy_available
def test_ndcg_empty() -> None:
    assert objects_are_allclose(
        ndcg(y_true=np.ones((0, 0)), y_score=np.ones((0, 0))),
        {"count": 0, "ndcg": float("nan")},
        equal_nan=True,
    )


@sklearn_available
@scipy_available
def test_ndcg_prefix_suffix() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, 1]]),
            y_score=np.array([[2.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 4, "prefix_ndcg_suffix": 1.0},
    )


@sklearn_available
@scipy_available
def test_ndcg_nan_omit() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
            ),
            nan_policy="omit",
        ),
        {"count": 2, "ndcg": 1.0},
    )


@sklearn_available
@scipy_available
def test_ndcg_omit_y_true() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, float("nan")]]),
            y_score=np.array([[2.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
            nan_policy="omit",
        ),
        {"count": 3, "ndcg": 1.0},
    )


@sklearn_available
@scipy_available
def test_ndcg_omit_y_score() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, 1]]),
            y_score=np.array(
                [[float("nan"), 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
            ),
            nan_policy="omit",
        ),
        {"count": 3, "ndcg": 1.0},
    )


@sklearn_available
@scipy_available
def test_ndcg_nan_propagate() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
            ),
        ),
        {"count": 4, "ndcg": float("nan")},
        equal_nan=True,
    )


@sklearn_available
@scipy_available
def test_ndcg_nan_propagate_y_true() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, float("nan")]]),
            y_score=np.array([[2.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
        ),
        {"count": 4, "ndcg": float("nan")},
        equal_nan=True,
    )


@sklearn_available
@scipy_available
def test_ndcg_nan_propagate_y_score() -> None:
    assert objects_are_allclose(
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, 1]]),
            y_score=np.array(
                [[float("nan"), 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
            ),
        ),
        {"count": 4, "ndcg": float("nan")},
        equal_nan=True,
    )


@sklearn_available
@scipy_available
def test_ndcg_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
            ),
            nan_policy="raise",
        )


@sklearn_available
@scipy_available
def test_ndcg_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, float("nan")]]),
            y_score=np.array([[2.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
            nan_policy="raise",
        )


@sklearn_available
@scipy_available
def test_ndcg_nan_raise_y_score() -> None:
    with pytest.raises(ValueError, match="'y_score' contains at least one NaN value"):
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, 1]]),
            y_score=np.array(
                [[float("nan"), 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
            ),
            nan_policy="raise",
        )


@patch("analora.utils.imports.is_sklearn_available", lambda: False)
def test_ndcg_no_sklearn() -> None:
    with pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."):
        ndcg(
            y_true=np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2], [0, 0, 1]]),
            y_score=np.array([[2.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
        )
