r"""Contain the base class to implement a HTML Content Generator."""

from __future__ import annotations

__all__ = ["BaseContentGenerator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.equality.comparators import BaseEqualityComparator
from coola.equality.handlers import EqualNanHandler, SameObjectHandler, SameTypeHandler
from coola.equality.testers import EqualityTester

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.equality import EqualityConfig


class BaseContentGenerator(ABC):
    r"""Define the base class to implement a HTML Content Generator.

    Example usage:

    ```pycon

    >>> from analora.content import ContentGenerator
    >>> content = ContentGenerator("meow")
    >>> content
    ContentGenerator()

    ```
    """

    @abstractmethod
    def compute(self) -> BaseContentGenerator:
        r"""Compute the content and return a new content generator.

        Returns:
            A new content generator with the computed content.

        Example usage:

        ```pycon

        >>> from analora.content import ContentGenerator
        >>> content = ContentGenerator("meow")
        >>> content
        ContentGenerator()
        >>> content2 = content.compute()
        >>> content2
        ContentGenerator()

        ```
        """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two content generators are equal or not.

        Args:
            other: The other content generator to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two content generators are equal,
                otherwise ``False``.

        Example usage:

        ```pycon

        >>> from analora.content import ContentGenerator
        >>> content1 = ContentGenerator("meow")
        >>> content2 = ContentGenerator("meow")
        >>> content3 = ContentGenerator("hello")
        >>> content1.equal(content2)
        True
        >>> content1.equal(content3)
        False

        ```
        """

    @abstractmethod
    def generate_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        r"""Return the HTML body associated to the content.

        Args:
            number: The section number, if any.
            tags: The tags associated to the content section, if any.
            depth: The depth in the content section, if any.

        Returns:
            The HTML body associated to the content section.

        Example usage:

        ```pycon

        >>> from analora.content import ContentGenerator
        >>> content = ContentGenerator("meow")
        >>> content.generate_body()

        ```
        """

    @abstractmethod
    def generate_toc(
        self, number: str = "", tags: Sequence[str] = (), depth: int = 0, max_depth: int = 1
    ) -> str:
        r"""Return the HTML table of content (TOC) associated to the
        section.

        Args:
            number: The section number associated to the
                section, if any.
            tags: The tags associated to the section, if any.
            depth: The depth in the report, if any.
            max_depth: The maximum depth to generate in the TOC.

        Returns:
            The HTML table of content associated to the section.

        Example usage:

        ```pycon

        >>> from analora.content import ContentGenerator
        >>> content = ContentGenerator("meow")
        >>> content.generate_toc()

        ```
        """


class ContentGeneratorEqualityComparator(BaseEqualityComparator[BaseContentGenerator]):
    r"""Implement an equality comparator for ``BaseContentGenerator``
    objects."""

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(EqualNanHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> ContentGeneratorEqualityComparator:
        return self.__class__()

    def equal(self, actual: BaseContentGenerator, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


if not EqualityTester.has_comparator(BaseContentGenerator):  # pragma: no cover
    EqualityTester.add_comparator(BaseContentGenerator, ContentGeneratorEqualityComparator())
