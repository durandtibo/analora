from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

from analora.plot import binary_roc_curve
from analora.testing import sklearn_available

######################################
#     Tests for binary_roc_curve     #
######################################


@sklearn_available
def test_binary_roc_curve() -> None:
    _fig, ax = plt.subplots()
    binary_roc_curve(ax, y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))


@sklearn_available
def test_binary_roc_curve_empty() -> None:
    _fig, ax = plt.subplots()
    binary_roc_curve(ax, y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))


@sklearn_available
def test_binary_roc_curve_nan() -> None:
    _fig, ax = plt.subplots()
    binary_roc_curve(
        ax,
        y_true=np.array([1, 0, 0, 1, 1, float("nan"), float("nan"), 1]),
        y_score=np.array([2, -1, 0, 3, 1, float("nan"), 1, float("nan")]),
    )


@patch("analora.utils.imports.is_sklearn_available", lambda: False)
def test_binary_roc_curve_no_sklearn() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."):
        binary_roc_curve(
            ax,
            y_true=np.array([1, 0, 0, 1, 1, float("nan"), float("nan"), 1]),
            y_score=np.array([2, -1, 0, 3, 1, float("nan"), 1, float("nan")]),
        )
