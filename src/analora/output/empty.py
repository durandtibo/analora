r"""Implement an empty output."""

from __future__ import annotations

__all__ = ["EmptyOutput"]


from analora.content.vanilla import ContentGenerator
from analora.evaluator.vanilla import Evaluator
from analora.output.vanilla import Output


class EmptyOutput(Output):
    r"""Implement the accuracy output.

    Example usage:

    ```pycon

    >>> from analora.output import EmptyOutput
    >>> output = EmptyOutput()
    >>> output
    EmptyOutput()
    >>> output.get_content_generator()
    ContentGenerator()
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    def __init__(self) -> None:
        super().__init__(content=ContentGenerator(), evaluator=Evaluator())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"
