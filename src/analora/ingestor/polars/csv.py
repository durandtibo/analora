r"""Contain the implementation of CSV ingestors."""

from __future__ import annotations

__all__ = ["CsvIngestor"]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola import objects_are_equal

from analora.ingestor.base import BaseIngestor
from analora.utils.format import str_kwargs
from analora.utils.imports import check_polars, is_polars_available

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    pl = Mock()

if TYPE_CHECKING:
    from analora.ingestor.polars.utils import FileSource


logger = logging.getLogger(__name__)


class CsvIngestor(BaseIngestor[pl.DataFrame]):
    r"""Implement a CSV ingestor.

    Args:
        source: The source to the CSV data to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.scan_csv``.

    Example usage:

    ```pycon

    >>> from analora.ingestor.polars import CsvIngestor
    >>> ingestor = CsvIngestor(source="/path/to/frame.csv")
    >>> ingestor
    CsvIngestor(source=/path/to/frame.csv)
    >>> frame = ingestor.ingest()  # doctest: +SKIP

    ```
    """

    def __init__(self, source: FileSource, **kwargs: Any) -> None:
        check_polars()
        self._source = source
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(source={self._source}{str_kwargs(self._kwargs)})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._source == other._source and objects_are_equal(
            self._kwargs, other._kwargs, equal_nan=equal_nan
        )

    def ingest(self) -> pl.DataFrame:
        logger.info(f"Ingesting CSV data from {self._source}...")
        frame = pl.read_csv(self._source, **self._kwargs)
        logger.info(f"DataFrame ingested | schema={frame.collect_schema()}")
        return frame
