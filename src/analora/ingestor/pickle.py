r"""Contain the implementation of a pickle ingestor."""

from __future__ import annotations

__all__ = ["PickleIngestor"]

import logging
from typing import TYPE_CHECKING, Any

from iden.io import load_pickle

from analora.ingestor.base import BaseIngestor
from analora.utils.path import sanitize_path
from analora.utils.timing import timeblock

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


class PickleIngestor(BaseIngestor[Any]):
    r"""Implement a pickle file ingestor.

    Args:
        path: The path to the pickle file containing the data to ingest.

    Example usage:

    ```pycon

    >>> from analora.ingestor import PickleIngestor
    >>> ingestor = PickleIngestor(path="/path/to/data.pickle")
    >>> ingestor
    PickleIngestor(path=/path/to/data.pickle)
    >>> data = ingestor.ingest()  # doctest: +SKIP

    ```
    """

    def __init__(self, path: Path | str) -> None:
        self._path = sanitize_path(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(path={self._path})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        if not isinstance(other, self.__class__):
            return False
        return self._path == other._path

    def ingest(self) -> Any:
        logger.info(f"Ingesting data from {self._path}...")
        with timeblock("Ingestion time: {time}"):
            return load_pickle(self._path)
