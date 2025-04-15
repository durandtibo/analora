r"""Contain utility functions for polars data ingestors."""

from __future__ import annotations

__all__ = ["FileSource"]

from pathlib import Path
from typing import IO, Union

FileSource = Union[  # noqa: UP007
    str,
    Path,
    IO[bytes],
    bytes,
    list[str],
    list[Path],
    list[IO[bytes]],
    list[bytes],
]
