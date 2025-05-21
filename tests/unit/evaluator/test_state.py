from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from analora.evaluator import BaseStateEvaluator, Evaluator
from analora.state import AccuracyState, BaseState


class MyStateCachedEvaluator(BaseStateEvaluator):
    def _evaluate(self) -> dict:
        return {"metric1": 0.42, "metric2": 1.2}


##############################################
#     Tests for BaseStateCachedEvaluator     #
##############################################


@pytest.fixture
def state() -> BaseState:
    return AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )


def test_base_state_evaluator_repr(state: BaseState) -> None:
    assert repr(MyStateCachedEvaluator(state)).startswith("MyStateCachedEvaluator(")


def test_base_state_evaluator_str(state: BaseState) -> None:
    assert str(MyStateCachedEvaluator(state)).startswith("MyStateCachedEvaluator(")


def test_base_state_evaluator_state(state: BaseState) -> None:
    assert MyStateCachedEvaluator(state).state.equal(state)


def test_base_state_evaluator_compute(state: BaseState) -> None:
    assert (
        MyStateCachedEvaluator(state).compute().equal(Evaluator({"metric1": 0.42, "metric2": 1.2}))
    )


def test_base_state_evaluator_equal_true(state: BaseState) -> None:
    assert MyStateCachedEvaluator(state).equal(MyStateCachedEvaluator(state))


def test_base_state_evaluator_equal_false_different_state(state: BaseState) -> None:
    assert not MyStateCachedEvaluator(state).equal(
        MyStateCachedEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1, 0]),
                y_pred=np.array([1, 0, 0, 1, 1, 0]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    )


def test_base_state_evaluator_equal_false_different_type(state: BaseState) -> None:
    assert not MyStateCachedEvaluator(state).equal(42)


def test_base_state_evaluator_evaluate(state: BaseState) -> None:
    evaluator = MyStateCachedEvaluator(state)
    out = evaluator.evaluate()
    assert objects_are_equal(out, {"metric1": 0.42, "metric2": 1.2})


def test_base_state_evaluator_evaluate_multi(state: BaseState) -> None:
    evaluator = MyStateCachedEvaluator(state)
    out1 = evaluator.evaluate()
    out2 = evaluator.evaluate()
    assert objects_are_equal(out1, out2)
    assert out1 is not out2


def test_base_state_evaluator_evaluate_prefix_suffix(state: BaseState) -> None:
    evaluator = MyStateCachedEvaluator(state)
    out = evaluator.evaluate(prefix="prefix_", suffix="_suffix")
    assert objects_are_equal(out, {"prefix_metric1_suffix": 0.42, "prefix_metric2_suffix": 1.2})
