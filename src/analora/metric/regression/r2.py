r"""Implement the R^2 (coefficient of determination) regression score
metrics."""

from __future__ import annotations

__all__ = ["r2_score"]


from typing import TYPE_CHECKING

from analora.metric.utils import contains_nan, preprocess_pred
from analora.utils.imports import check_sklearn, is_sklearn_available

if is_sklearn_available():  # pragma: no cover
    from sklearn import metrics

if TYPE_CHECKING:
    import numpy as np


def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    nan_policy: str = "propagate",
) -> dict[str, float]:
    r"""Return the R^2 (coefficient of determination) regression score
    metrics.

    Args:
        y_true: The ground truth target values.
        y_pred: The predicted values.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from analora.metric import r2_score
    >>> r2_score(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5]))
    {'count': 5, 'r2_score': 1.0}

    ```
    """
    check_sklearn()
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), drop_nan=nan_policy == "omit"
    )
    y_true_nan = contains_nan(arr=y_true, nan_policy=nan_policy, name="'y_true'")
    y_pred_nan = contains_nan(arr=y_pred, nan_policy=nan_policy, name="'y_pred'")

    count = y_true.size
    error = float("nan")
    if count > 0 and not y_true_nan and not y_pred_nan:
        error = float(metrics.r2_score(y_true=y_true, y_pred=y_pred))
    return {
        f"{prefix}count{suffix}": count,
        f"{prefix}r2_score{suffix}": error,
    }
