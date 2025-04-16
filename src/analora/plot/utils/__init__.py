r"""Contain utility functions to generate plots."""

from __future__ import annotations

__all__ = [
    "auto_yscale_continuous",
    "auto_yscale_discrete",
    "axvline_median",
    "axvline_quantile",
    "readable_xticklabels",
    "readable_yticklabels",
]

from analora.plot.utils.line import axvline_median, axvline_quantile
from analora.plot.utils.scale import auto_yscale_continuous, auto_yscale_discrete
from analora.plot.utils.tick import readable_xticklabels, readable_yticklabels
