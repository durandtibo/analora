r"""Contain an analyzer that generates an output with the given custom
content."""

from __future__ import annotations

__all__ = ["ContentAnalyzer"]

import logging
from typing import Any

from analora.analyzer.lazy import BaseLazyAnalyzer
from analora.content import ContentGenerator
from analora.evaluator import Evaluator
from analora.output import Output

logger = logging.getLogger(__name__)


class ContentAnalyzer(BaseLazyAnalyzer[Any]):
    r"""Implement an analyzer that generates an output with the given
    custom content.

    Args:
        content: The content to use in the HTML code.

    Example usage:

    ```pycon

    >>> from analora.analyzer import ContentAnalyzer
    >>> analyzer = ContentAnalyzer(content="meow")
    >>> analyzer
    ContentAnalyzer()
    >>> data = {"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]}
    >>> output = analyzer.analyze(data)
    >>> output
    Output(
      (content): ContentGenerator()
      (evaluator): Evaluator(count=0)
    )

    ```
    """

    def __init__(self, content: str) -> None:
        self._content = content

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _analyze(self, data: Any) -> Output:  # noqa: ARG002
        return Output(evaluator=Evaluator({}), content=ContentGenerator(self._content))
