r"""Define a base class to implement lazy analyzers."""

from __future__ import annotations

__all__ = ["BaseLazyAnalyzer"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

from analora.analyzer.base import BaseAnalyzer

if TYPE_CHECKING:
    from analora.output import BaseOutput

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseLazyAnalyzer(BaseAnalyzer):
    r"""Define a base class to implement a lazy analyzer.

    Example usage:

    ```pycon

    # >>> import polars as pl
    # >>> from analora.analyzer import SummaryAnalyzer
    # >>> analyzer = SummaryAnalyzer()
    # >>> analyzer
    # SummaryAnalyzer(columns=None, exclude_columns=(), missing_policy='raise', top=5)
    # >>> frame = pl.DataFrame(
    # ...     {
    # ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    # ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    # ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    # ...     },
    # ...     schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64},
    # ... )
    # >>> output = analyzer.analyze(frame)
    # >>> output
    # SummaryOutput(
    #   (state): DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig(), top=5)
    # )

    ```
    """

    def analyze(self, data: T, lazy: bool = True) -> BaseOutput:
        output = self._analyze(data)
        if not lazy:
            output = output.compute()
        return output

    @abstractmethod
    def _analyze(self, data: T) -> BaseOutput:
        r"""Analyze the data.

        Args:
            data: The data to analyze.

        Returns:
            The generated output.
        """
