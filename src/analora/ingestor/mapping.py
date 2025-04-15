r"""Contain a data ingestor that ingests data from a mapping of
ingestors."""

from __future__ import annotations

__all__ = ["MappingIngestor"]

from typing import TYPE_CHECKING, Any, TypeVar

from coola import objects_are_equal
from coola.utils.format import repr_indent, repr_mapping

from analora.ingestor import setup_ingestor
from analora.ingestor.base import BaseIngestor

if TYPE_CHECKING:
    from collections.abc import Mapping

T = TypeVar("T")


class MappingIngestor(BaseIngestor[dict[str, T]]):
    r"""Implement a simple data ingestor.

    Args:
        ingestors: The mapping of ingestors or their configuration.

    Example usage:

    ```pycon

    >>> from analora.ingestor import Ingestor, MappingIngestor
    >>> ingestor = MappingIngestor(
    ...     {"key1": Ingestor(data=[1, 2, 3, 4, 5]), "key2": Ingestor(data="meow")}
    ... )
    >>> ingestor
    MappingIngestor(
      (key1): Ingestor()
      (key2): Ingestor()
    )
    >>> data = ingestor.ingest()
    >>> data
    {'key1': [1, 2, 3, 4, 5], 'key2': 'meow'}

    ```
    """

    def __init__(self, ingestors: Mapping[str, BaseIngestor[T] | dict]) -> None:
        self._ingestors = {key: setup_ingestor(ingestor) for key, ingestor in ingestors.items()}

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._ingestors))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._ingestors, other._ingestors, equal_nan=equal_nan)

    def ingest(self) -> Any:
        return {key: ingestor.ingest() for key, ingestor in self._ingestors.items()}
