r"""Contain analyzers."""

from __future__ import annotations

__all__ = [
    "AccuracyAnalyzer",
    "BaseAnalyzer",
    "BaseLazyAnalyzer",
    "BaseTruePredAnalyzer",
    "ContentAnalyzer",
    "is_analyzer_config",
    "setup_analyzer",
]

from analora.analyzer.accuracy import AccuracyAnalyzer
from analora.analyzer.base import BaseAnalyzer, is_analyzer_config, setup_analyzer
from analora.analyzer.bases import BaseTruePredAnalyzer
from analora.analyzer.content import ContentAnalyzer
from analora.analyzer.lazy import BaseLazyAnalyzer
