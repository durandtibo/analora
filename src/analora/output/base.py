r"""Contain the base class to implement an output."""

from __future__ import annotations

__all__ = ["BaseOutput"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.equality.comparators import BaseEqualityComparator
from coola.equality.handlers import EqualNanHandler, SameObjectHandler, SameTypeHandler
from coola.equality.testers import EqualityTester

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

    from analora.content.base import BaseContentGenerator
    from analora.evaluator.base import BaseEvaluator


class BaseOutput(ABC):
    r"""Define the base class to implement an output.

    Example usage:

    ```pycon

    >>> from analora.output import Output
    >>> from analora.content import ContentGenerator
    >>> from analora.evaluator import Evaluator
    >>> output = Output(content=ContentGenerator("meow"), evaluator=Evaluator())
    >>> output
    Output(
      (content): ContentGenerator()
      (evaluator): Evaluator(count=0)
    )
    >>> output.get_content_generator()
    ContentGenerator()
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    @abstractmethod
    def compute(self) -> BaseOutput:
        r"""Compute the results and return a new ouptut.

        Returns:
            A new ouptut with the computed results.

        Example usage:

        ```pycon

        >>> from analora.output import Output
        >>> from analora.content import ContentGenerator
        >>> from analora.evaluator import Evaluator
        >>> output = Output(
        ...     content=ContentGenerator("meow"), evaluator=Evaluator({"accuracy": 0.42})
        ... )
        >>> out = output.compute()
        >>> out
        Output(
          (content): ContentGenerator()
          (evaluator): Evaluator(count=1)
        )

        ```
        """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two outputs are equal or not.

        Args:
            other: The other output to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two outputs are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> from analora.output import Output
        >>> from analora.content import ContentGenerator
        >>> from analora.evaluator import Evaluator
        >>> output1 = Output(content=ContentGenerator("meow"), evaluator=Evaluator())
        >>> output2 = Output(content=ContentGenerator("meow"), evaluator=Evaluator())
        >>> output3 = Output(
        ...     content=ContentGenerator("hello"), evaluator=Evaluator({"accuracy": 0.42})
        ... )
        >>> output1.equal(output2)
        True
        >>> output1.equal(output3)
        False

        ```
        """

    @abstractmethod
    def get_content_generator(self, lazy: bool = True) -> BaseContentGenerator:
        r"""Get the HTML content generator associated to the output.

        Args:
            lazy: If ``True``, it forces the computation of the
                content, otherwise it returns a content generator
                object that contains the logic to generate the content.

        Returns:
            The HTML content generator.

        Example usage:

        ```pycon

        >>> from analora.output import Output
        >>> from analora.content import ContentGenerator
        >>> from analora.evaluator import Evaluator
        >>> output = Output(content=ContentGenerator("meow"), evaluator=Evaluator())
        >>> output.get_content_generator()
        ContentGenerator()

        ```
        """

    @abstractmethod
    def get_evaluator(self, lazy: bool = True) -> BaseEvaluator:
        r"""Get the evaluator associated to the output.

        Args:
            lazy: If ``True``, it forces the computation of the
                metrics, otherwise it returns an evaluator object
                that contains the logic to evaluate the metrics.

        Returns:
            The evaluator.

        Example usage:

        ```pycon

        >>> from analora.output import Output
        >>> from analora.content import ContentGenerator
        >>> from analora.evaluator import Evaluator
        >>> output = Output(content=ContentGenerator("meow"), evaluator=Evaluator())
        >>> output.get_evaluator()
        Evaluator(count=0)
        """


class OutputEqualityComparator(BaseEqualityComparator[BaseOutput]):
    r"""Implement an equality comparator for ``BaseOutput`` objects."""

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(EqualNanHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> OutputEqualityComparator:
        return self.__class__()

    def equal(self, actual: BaseOutput, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


if not EqualityTester.has_comparator(BaseOutput):  # pragma: no cover
    EqualityTester.add_comparator(BaseOutput, OutputEqualityComparator())
