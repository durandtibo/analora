r"""Contain the accuracy analyzer."""

from __future__ import annotations

__all__ = ["AccuracyAnalyzer"]

import logging
from typing import TYPE_CHECKING

from coola.utils.array import to_array
from coola.utils.format import repr_mapping_line

from analora.analyzer.bases import BaseTruePredAnalyzer
from analora.content import ContentGenerator
from analora.evaluator import AccuracyEvaluator
from analora.output import Output
from analora.state.accuracy import AccuracyState
from analora.utils.imports import check_sklearn

if TYPE_CHECKING:
    from collections.abc import Mapping


logger = logging.getLogger(__name__)


class AccuracyAnalyzer(BaseTruePredAnalyzer):
    r"""Implement the accuracy analyzer.

    Args:
        y_true: The key of the ground truth target
            labels.
        y_pred: The key of the predicted labels.
        missing_policy: The policy on how to handle missing keys.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one key is missing.
            If ``'warn'``, a warning is raised if at least one key
            is missing and the missing keys are ignored.
            If ``'ignore'``, the missing keys are ignored and
            no warning message appears.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

    Example usage:

    ```pycon

    >>> from analora.analyzer import AccuracyAnalyzer
    >>> analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred")
    >>> analyzer
    AccuracyAnalyzer(y_true='target', y_pred='pred', missing_policy='raise', nan_policy='propagate')
    >>> data = {"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]}
    >>> output = analyzer.analyze(data)
    >>> output
    Output(
      (content): ContentGenerator()
      (evaluator): AccuracyEvaluator(
          (state): AccuracyState(y_true=(6,), y_pred=(6,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
        )
    )

    ```
    """

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        missing_policy: str = "raise",
        nan_policy: str = "propagate",
    ) -> None:
        super().__init__(
            y_true=y_true, y_pred=y_pred, nan_policy=nan_policy, missing_policy=missing_policy
        )
        check_sklearn()

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "y_true": self._y_true,
                "y_pred": self._y_pred,
                "missing_policy": self._missing_policy,
                "nan_policy": self._nan_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def _analyze(self, data: Mapping) -> Output:
        logger.info(
            f"Evaluating the accuracy | y_true={self._y_true!r} | y_pred={self._y_pred!r} | "
            f"nan_policy={self._nan_policy!r}"
        )
        state = AccuracyState(
            y_true=to_array(data[self._y_true]).ravel(),
            y_pred=to_array(data[self._y_pred]).ravel(),
            y_true_name=self._y_true,
            y_pred_name=self._y_pred,
            nan_policy=self._nan_policy,
        )
        return Output(evaluator=AccuracyEvaluator(state), content=ContentGenerator())
