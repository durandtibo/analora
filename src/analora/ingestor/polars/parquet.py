r"""Contain the implementation of parquet ingestors."""

from __future__ import annotations

__all__ = ["ParquetIngestor"]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola import objects_are_equal
from iden.utils.time import timeblock

from analora.ingestor.base import BaseIngestor
from analora.utils.format import human_byte, str_kwargs
from analora.utils.imports import check_polars, is_polars_available

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    pl = Mock()

if TYPE_CHECKING:
    from analora.ingestor.polars.utils import FileSource


logger = logging.getLogger(__name__)


class ParquetIngestor(BaseIngestor[pl.DataFrame]):
    r"""Implement a parquet ingestor.

    Args:
        source: The source to the parquet data to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.read_parquet``.

    Example usage:

    ```pycon

    >>> from analora.ingestor.polars import ParquetIngestor
    >>> ingestor = ParquetIngestor(source="/path/to/frame.parquet")
    >>> ingestor
    ParquetIngestor(source=/path/to/frame.parquet)
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
        logger.info(f"Ingesting parquet data from {self._source}...")
        with timeblock("DataFrame ingestion time: {time}"):
            frame = pl.read_parquet(self._source, **self._kwargs)
            logger.info(
                f"DataFrame ingested | shape={frame.shape}  "
                f"estimated size={human_byte(frame.estimated_size())}"
            )
        return frame
