r"""Contain the implementation of a simple data ingestor."""

from __future__ import annotations

__all__ = ["Ingestor"]


from typing import Any

from coola import objects_are_equal

from analora.ingestor.base import BaseIngestor


class Ingestor(BaseIngestor[Any]):
    r"""Implement a simple data ingestor.

    Args:
        data: The data to ingest.

    Example usage:

    ```pycon

    >>> from analora.ingestor import Ingestor
    >>> ingestor = Ingestor(data=[1, 2, 3, 4, 5])
    >>> ingestor
    Ingestor()
    >>> data = ingestor.ingest()

    ```
    """

    def __init__(self, data: Any) -> None:
        self._data = data

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._data, other._data, equal_nan=equal_nan)

    def ingest(self) -> Any:
        return self._data
