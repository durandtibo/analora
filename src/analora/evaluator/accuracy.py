r"""Implement the accuracy evaluator."""

from __future__ import annotations

__all__ = ["AccuracyEvaluator"]


from analora.evaluator.state import BaseStateEvaluator
from analora.metric import accuracy
from analora.state.accuracy import AccuracyState
from analora.utils.imports import check_sklearn


class AccuracyEvaluator(BaseStateEvaluator[AccuracyState]):
    r"""Implement the accuracy evaluator.

    Args:
        state: The state containing the ground truth and predicted
            labels.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from analora.evaluator import AccuracyEvaluator
    >>> from analora.state import AccuracyState
    >>> evaluator = AccuracyEvaluator(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> evaluator
    AccuracyEvaluator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
    )
    >>> evaluator.evaluate()
    {'accuracy': 1.0, 'count_correct': 5, 'count_incorrect': 0, 'count': 5, 'error': 0.0}

    ```
    """

    def __init__(self, state: AccuracyState) -> None:
        super().__init__(state)
        check_sklearn()

    def _evaluate(self) -> dict[str, float]:
        return accuracy(
            y_true=self._state.y_true,
            y_pred=self._state.y_pred,
            nan_policy=self._state.nan_policy,
        )
