from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from coola import objects_are_equal

from analora.metric import r2_score
from analora.testing import sklearn_available

##############################
#     Tests for r2_score     #
##############################


@sklearn_available
def test_r2_score_correct() -> None:
    assert objects_are_equal(
        r2_score(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "r2_score": 1.0},
    )


@sklearn_available
def test_r2_score_correct_2d() -> None:
    assert objects_are_equal(
        r2_score(y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])),
        {"count": 6, "r2_score": 1.0},
    )


@sklearn_available
def test_r2_score_incorrect() -> None:
    assert objects_are_equal(
        r2_score(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])),
        {"count": 4, "r2_score": -3.0},
    )


@sklearn_available
def test_r2_score_empty() -> None:
    assert objects_are_equal(
        r2_score(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "r2_score": float("nan")},
        equal_nan=True,
    )


@sklearn_available
def test_r2_score_prefix_suffix() -> None:
    assert objects_are_equal(
        r2_score(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_r2_score_suffix": 1.0},
    )


@sklearn_available
def test_r2_score_nan_omit() -> None:
    assert objects_are_equal(
        r2_score(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="omit",
        ),
        {"count": 3, "r2_score": 1.0},
    )


@sklearn_available
def test_r2_score_nan_omit_y_true() -> None:
    assert objects_are_equal(
        r2_score(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="omit",
        ),
        {"count": 5, "r2_score": 1.0},
    )


@sklearn_available
def test_r2_score_nan_omit_y_pred() -> None:
    assert objects_are_equal(
        r2_score(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="omit",
        ),
        {"count": 5, "r2_score": 1.0},
    )


@sklearn_available
def test_r2_score_nan_propagate() -> None:
    assert objects_are_equal(
        r2_score(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "r2_score": float("nan")},
        equal_nan=True,
    )


@sklearn_available
def test_r2_score_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        r2_score(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="propagate",
        ),
        {"count": 6, "r2_score": float("nan")},
        equal_nan=True,
    )


@sklearn_available
def test_r2_score_nan_propagate_y_pred() -> None:
    assert objects_are_equal(
        r2_score(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "r2_score": float("nan")},
        equal_nan=True,
    )


@sklearn_available
def test_r2_score_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        r2_score(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="raise",
        )


@sklearn_available
def test_r2_score_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        r2_score(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="raise",
        )


@sklearn_available
def test_r2_score_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        r2_score(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="raise",
        )


@patch("analora.utils.imports.is_sklearn_available", lambda: False)
def test_r2_score_no_sklearn() -> None:
    with pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."):
        r2_score(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5]))
