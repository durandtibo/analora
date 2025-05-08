r"""Define some template classes to implement some analyzers."""

from __future__ import annotations

__all__ = ["BaseTruePredAnalyzer"]

import logging
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

from analora.analyzer.lazy import BaseLazyAnalyzer
from analora.metric.utils import check_nan_policy
from analora.output.empty import EmptyOutput
from analora.utils.mapping import check_missing_key
from analora.utils.policy import check_missing_policy

if TYPE_CHECKING:
    from analora.output.base import BaseOutput

T = TypeVar("T", bound=Mapping)

logger = logging.getLogger(__name__)


class BaseTruePredAnalyzer(BaseLazyAnalyzer[T]):
    r"""Define a base class to implement a data analyzer that takes two
    input keys: ``y_true`` and ``y_pred``.

    Args:
        y_true: The key name of the ground truth target
            labels.
        y_pred: The key name of the predicted labels.
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
    """

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        missing_policy: str,
        nan_policy: str,
    ) -> None:
        self._y_true = y_true
        self._y_pred = y_pred

        check_missing_policy(missing_policy)
        self._missing_policy = missing_policy

        check_nan_policy(nan_policy)
        self._nan_policy = nan_policy

    def analyze(self, data: T, lazy: bool = True) -> BaseOutput:
        self._check_data(data)
        for key in [self._y_true, self._y_pred]:
            if key not in data:
                logger.info(
                    f"Skipping '{self.__class__.__qualname__}.analyze' "
                    f"because the input key {key!r} is missing"
                )
                return EmptyOutput()
        return super().analyze(data=data, lazy=lazy)

    def _check_data(self, data: T) -> None:
        r"""Check if an input key is missing.

        Args:
            data: The input data to check.
        """
        check_missing_key(data, key=self._y_true, missing_policy=self._missing_policy)
        check_missing_key(data, key=self._y_pred, missing_policy=self._missing_policy)

    @abstractmethod
    def _analyze(self, data: T) -> BaseOutput:
        r"""Analyze the data.

        Args:
            data: The data to analyze.

        Returns:
            The generated output.
        """
