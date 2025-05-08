from __future__ import annotations

from analora.analyzer import ContentAnalyzer
from analora.content import ContentGenerator
from analora.evaluator import Evaluator
from analora.output import Output

#####################################
#     Tests for ContentAnalyzer     #
#####################################


def test_content_analyzer_repr() -> None:
    assert repr(ContentAnalyzer(content="meow")).startswith("ContentAnalyzer(")


def test_content_analyzer_str() -> None:
    assert str(ContentAnalyzer(content="meow")).startswith("ContentAnalyzer(")


def test_content_analyzer_analyze() -> None:
    assert (
        ContentAnalyzer(content="meow")
        .analyze({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]})
        .equal(Output(content=ContentGenerator("meow"), evaluator=Evaluator()))
    )


def test_content_analyzer_analyze_lazy_false() -> None:
    assert (
        ContentAnalyzer(content="meow")
        .analyze({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}, lazy=False)
        .equal(Output(content=ContentGenerator("meow"), evaluator=Evaluator()))
    )
