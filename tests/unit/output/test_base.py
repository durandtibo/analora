from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from analora.content import ContentGenerator
from analora.evaluator import Evaluator
from analora.output import Output
from analora.output.base import OutputEqualityComparator
from tests.unit.helpers import COMPARATOR_FUNCTIONS, ExamplePair

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##############################################
#     Tests for OutputEqualityComparator     #
##############################################


OUTPUT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=Output(content=ContentGenerator("meow"), evaluator=Evaluator()),
            expected=Output(content=ContentGenerator("meow"), evaluator=Evaluator()),
        ),
        id="output",
    ),
    pytest.param(
        ExamplePair(
            actual=Output(
                content=ContentGenerator("meow"), evaluator=Evaluator({"accuracy": 0.42})
            ),
            expected=Output(
                content=ContentGenerator("meow"), evaluator=Evaluator({"accuracy": 0.42})
            ),
        ),
        id="accuracy output",
    ),
]

OUTPUT_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=Output(content=ContentGenerator("meow"), evaluator=Evaluator()),
            expected=Evaluator({"accuracy": 0.42}),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=Output(content=ContentGenerator("meow"), evaluator=Evaluator()),
            expected=Output(content=ContentGenerator("miaou"), evaluator=Evaluator()),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_output_equality_comparator_repr() -> None:
    assert repr(OutputEqualityComparator()) == "OutputEqualityComparator()"


def test_output_equality_comparator_str() -> None:
    assert str(OutputEqualityComparator()) == "OutputEqualityComparator()"


def test_output_equality_comparator__eq__true() -> None:
    assert OutputEqualityComparator() == OutputEqualityComparator()


def test_output_equality_comparator__eq__false() -> None:
    assert OutputEqualityComparator() != 123


def test_output_equality_comparator_clone() -> None:
    op = OutputEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_output_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = Output(content=ContentGenerator("meow"), evaluator=Evaluator())
    assert OutputEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", OUTPUT_EQUAL)
def test_output_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = OutputEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", OUTPUT_EQUAL)
def test_output_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = OutputEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", OUTPUT_NOT_EQUAL)
def test_output_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = OutputEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", OUTPUT_NOT_EQUAL)
def test_output_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = OutputEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", OUTPUT_EQUAL)
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
@pytest.mark.parametrize("example", OUTPUT_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", OUTPUT_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
