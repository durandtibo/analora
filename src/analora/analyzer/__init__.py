r"""Contain analyzers."""

from __future__ import annotations

__all__ = ["BaseAnalyzer", "BaseLazyAnalyzer", "is_analyzer_config", "setup_analyzer"]

from analora.analyzer.base import BaseAnalyzer, is_analyzer_config, setup_analyzer
from analora.analyzer.lazy import BaseLazyAnalyzer
