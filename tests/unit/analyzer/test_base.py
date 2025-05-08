from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from analora.analyzer import (
    ContentAnalyzer,
    is_analyzer_config,
    setup_analyzer,
)

if TYPE_CHECKING:
    import pytest

########################################
#     Tests for is_analyzer_config     #
########################################


def test_is_analyzer_config_true() -> None:
    assert is_analyzer_config({OBJECT_TARGET: "analora.analyzer.ContentAnalyzer"})


def test_is_analyzer_config_false() -> None:
    assert not is_analyzer_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_analyzer     #
####################################


def test_setup_analyzer_object() -> None:
    analyzer = ContentAnalyzer("meow")
    assert setup_analyzer(analyzer) is analyzer


def test_setup_analyzer_dict() -> None:
    assert isinstance(
        setup_analyzer({OBJECT_TARGET: "analora.analyzer.ContentAnalyzer", "content": "meow"}),
        ContentAnalyzer,
    )


def test_setup_analyzer_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_analyzer({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
