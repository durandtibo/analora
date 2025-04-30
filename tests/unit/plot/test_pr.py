from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

from analora.plot import binary_precision_recall_curve
from analora.testing import sklearn_available

###################################################
#     Tests for binary_precision_recall_curve     #
###################################################


@sklearn_available
def test_binary_precision_recall_curve() -> None:
    _fig, ax = plt.subplots()
    binary_precision_recall_curve(
        ax, y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )


@sklearn_available
def test_binary_precision_recall_curve_empty() -> None:
    _fig, ax = plt.subplots()
    binary_precision_recall_curve(
        ax, y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )


@sklearn_available
def test_binary_precision_recall_curve_nan() -> None:
    _fig, ax = plt.subplots()
    binary_precision_recall_curve(
        ax,
        y_true=np.array([1, 0, 0, 1, 1, float("nan"), float("nan"), 1]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan"), 1, float("nan")]),
    )


@patch("analora.utils.imports.is_sklearn_available", lambda: False)
def test_binary_precision_recall_curve_no_sklearn() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."):
        binary_precision_recall_curve(
            ax, y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        )
