r"""Contain data ingestors."""

from __future__ import annotations

__all__ = ["BaseIngestor", "Ingestor", "MappingIngestor", "is_ingestor_config", "setup_ingestor"]

from analora.ingestor.base import BaseIngestor, is_ingestor_config, setup_ingestor
from analora.ingestor.mapping import MappingIngestor
from analora.ingestor.vanilla import Ingestor
