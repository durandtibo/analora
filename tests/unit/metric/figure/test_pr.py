from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

from analora.metric.figure import binary_precision_recall_curve
from analora.testing import sklearn_available

###################################################
#     Tests for binary_precision_recall_curve     #
###################################################


@sklearn_available
def test_binary_precision_recall_curve() -> None:
    assert isinstance(
        binary_precision_recall_curve(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        ),
        plt.Figure,
    )


@sklearn_available
def test_binary_precision_recall_curve_empty() -> None:
    assert binary_precision_recall_curve(y_true=np.array([]), y_pred=np.array([])) is None


@sklearn_available
def test_binary_precision_recall_curve_nan() -> None:
    assert isinstance(
        binary_precision_recall_curve(
            y_true=np.array([1, 0, 0, 1, 1, float("nan"), float("nan"), 1]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan"), 1, float("nan")]),
        ),
        plt.Figure,
    )


@patch("analora.utils.imports.is_sklearn_available", lambda: False)
def test_binary_precision_recall_curve_no_sklearn() -> None:
    with pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."):
        binary_precision_recall_curve(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        )
