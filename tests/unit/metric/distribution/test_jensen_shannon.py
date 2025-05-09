from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from coola import objects_are_allclose

from analora.metric import jensen_shannon_divergence
from analora.testing import scipy_available

###############################################
#     Tests for jensen_shannon_divergence     #
###############################################


@scipy_available
def test_jensen_shannon_divergence_same() -> None:
    assert objects_are_allclose(
        jensen_shannon_divergence(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
        ),
        {"size": 4, "jensen_shannon_divergence": 0.0},
    )


@scipy_available
def test_jensen_shannon_divergence_different() -> None:
    assert objects_are_allclose(
        jensen_shannon_divergence(p=np.array([0.10, 0.40, 0.50]), q=np.array([0.80, 0.15, 0.05])),
        {"size": 3, "jensen_shannon_divergence": 0.29126084062606405},
    )


@scipy_available
def test_jensen_shannon_divergence_empty() -> None:
    assert objects_are_allclose(
        jensen_shannon_divergence(p=np.array([]), q=np.array([])),
        {"size": 0, "jensen_shannon_divergence": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_jensen_shannon_divergence_prefix_suffix() -> None:
    assert objects_are_allclose(
        jensen_shannon_divergence(
            p=np.array([0.1, 0.6, 0.1, 0.2]),
            q=np.array([0.1, 0.6, 0.1, 0.2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_size_suffix": 4, "prefix_jensen_shannon_divergence_suffix": 0.0},
    )


@patch("analora.utils.imports.is_scipy_available", lambda: False)
def test_jensen_shannon_divergence_no_scipy() -> None:
    with pytest.raises(RuntimeError, match="'scipy' package is required but not installed."):
        jensen_shannon_divergence(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
        )
