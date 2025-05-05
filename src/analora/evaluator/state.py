r"""Define the base class to implement evaluators that compute the
metrics from a state object."""

from __future__ import annotations

__all__ = ["BaseStateEvaluator"]

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from analora.evaluator.base import BaseEvaluator
from analora.evaluator.vanilla import Evaluator
from analora.state.base import BaseState

T = TypeVar("T", bound=BaseState)


class BaseStateEvaluator(BaseEvaluator, Generic[T]):
    r"""Define the base class to implement evaluators that compute the
    metrics from a state object.

    Args:
        state: The state with the data.
    """

    def __init__(self, state: T) -> None:
        super().__init__()
        self._state = state

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def state(self) -> T:
        return self._state

    def compute(self) -> Evaluator:
        return Evaluator(metrics=self.evaluate())

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._state.equal(other._state, equal_nan=equal_nan)

    def evaluate(self, prefix: str = "", suffix: str = "") -> dict:
        metrics = self._evaluate()
        return {f"{prefix}{col}{suffix}": val for col, val in metrics.items()}

    @abstractmethod
    def _evaluate(self) -> dict:
        r"""Evaluate the metrics.

        Returns:
            The metrics.
        """
