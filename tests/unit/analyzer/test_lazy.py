from __future__ import annotations

from analora.analyzer import BaseLazyAnalyzer
from analora.content import ContentGenerator
from analora.evaluator import Evaluator
from analora.output import Output

######################################
#     Tests for BaseLazyAnalyzer     #
######################################


class MyLazyAnalyzer(BaseLazyAnalyzer):
    def _analyze(self, data: dict) -> Output:
        return Output(content=ContentGenerator("meow"), evaluator=Evaluator(data))


def test_base_lazy_analyzer_analyze() -> None:
    assert (
        MyLazyAnalyzer()
        .analyze({"accuracy": 0.42})
        .equal(Output(content=ContentGenerator("meow"), evaluator=Evaluator({"accuracy": 0.42})))
    )


def test_base_lazy_analyzer_analyze_lazy_false() -> None:
    assert (
        MyLazyAnalyzer()
        .analyze({"accuracy": 0.42}, lazy=False)
        .equal(Output(content=ContentGenerator("meow"), evaluator=Evaluator({"accuracy": 0.42})))
    )
