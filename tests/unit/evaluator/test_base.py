from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from analora.evaluator import Evaluator
from analora.evaluator.base import EvaluatorEqualityComparator
from tests.unit.helpers import COMPARATOR_FUNCTIONS, ExamplePair

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#################################################
#     Tests for EvaluatorEqualityComparator     #
#################################################


EVALUATOR_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=Evaluator(),
            expected=Evaluator(),
        ),
        id="evaluator",
    ),
    pytest.param(
        ExamplePair(
            actual=Evaluator({"accuracy": 0.42}),
            expected=Evaluator({"accuracy": 0.42}),
        ),
        id="evaluator metrics",
    ),
]

EVALUATOR_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=Evaluator(),
            expected={"accuracy": 0.42},
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=Evaluator(),
            expected=Evaluator({"accuracy": 0.42}),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_evaluator_equality_comparator_repr() -> None:
    assert repr(EvaluatorEqualityComparator()) == "EvaluatorEqualityComparator()"


def test_evaluator_equality_comparator_str() -> None:
    assert str(EvaluatorEqualityComparator()) == "EvaluatorEqualityComparator()"


def test_evaluator_equality_comparator__eq__true() -> None:
    assert EvaluatorEqualityComparator() == EvaluatorEqualityComparator()


def test_evaluator_equality_comparator__eq__false() -> None:
    assert EvaluatorEqualityComparator() != 123


def test_evaluator_equality_comparator_clone() -> None:
    op = EvaluatorEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_evaluator_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = Evaluator()
    assert EvaluatorEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", EVALUATOR_EQUAL)
def test_evaluator_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = EvaluatorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EVALUATOR_EQUAL)
def test_evaluator_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = EvaluatorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EVALUATOR_NOT_EQUAL)
def test_evaluator_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = EvaluatorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EVALUATOR_NOT_EQUAL)
def test_evaluator_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = EvaluatorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", EVALUATOR_EQUAL)
@pytest.mark.parametrize("show_difference", [True, False])
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
    show_difference: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert function(example.actual, example.expected, show_difference=show_difference)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", EVALUATOR_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", EVALUATOR_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
