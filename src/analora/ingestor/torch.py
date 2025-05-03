r"""Contain the implementation of a torch ingestor."""

from __future__ import annotations

__all__ = ["TorchIngestor"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_mapping_line
from coola.utils.imports import check_torch
from iden.io import load_torch

from analora.ingestor.base import BaseIngestor
from analora.utils.path import sanitize_path
from analora.utils.timing import timeblock

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


class TorchIngestor(BaseIngestor[Any]):
    r"""Implement a torch file ingestor.

    Args:
        path: The path to the torch file containing the data to ingest.
        **kwargs: Additional arguments passed to ``torch.load``.

    Example usage:

    ```pycon

    >>> from analora.ingestor import TorchIngestor
    >>> ingestor = TorchIngestor(path="/path/to/data.pt")
    >>> ingestor
    TorchIngestor(path=/path/to/data.pt)
    >>> data = ingestor.ingest()  # doctest: +SKIP

    ```
    """

    def __init__(self, path: Path | str, **kwargs: Any) -> None:
        check_torch()
        self._path = sanitize_path(path)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(path={self._path}{repr_mapping_line(self._kwargs)})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        if not isinstance(other, self.__class__):
            return False
        return self._path == other._path

    def ingest(self) -> Any:
        logger.info(f"Ingesting data from {self._path}...")
        with timeblock("Ingestion time: {time}"):
            return load_torch(self._path, **self._kwargs)
