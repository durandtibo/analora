r"""Contain polars DataFrame ingestors."""

from __future__ import annotations

__all__ = ["CsvIngestor", "ParquetIngestor"]

from analora.ingestor.polars.csv import CsvIngestor
from analora.ingestor.polars.parquet import ParquetIngestor
