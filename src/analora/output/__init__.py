r"""Contain outputs."""

from __future__ import annotations

__all__ = ["BaseLazyOutput", "BaseOutput", "EmptyOutput", "Output"]

from analora.output.base import BaseOutput
from analora.output.empty import EmptyOutput
from analora.output.lazy import BaseLazyOutput
from analora.output.vanilla import Output
