from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from coola import objects_are_equal

from analora.evaluator import BalancedAccuracyEvaluator, Evaluator
from analora.state import AccuracyState
from analora.testing import sklearn_available

###############################################
#     Tests for BalancedAccuracyEvaluator     #
###############################################


@sklearn_available
def test_balanced_accuracy_evaluator_repr() -> None:
    assert repr(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("BalancedAccuracyEvaluator(")


@sklearn_available
def test_balanced_accuracy_evaluator_str() -> None:
    assert str(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("BalancedAccuracyEvaluator(")


@sklearn_available
def test_balanced_accuracy_evaluator_state() -> None:
    assert BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).state.equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


@sklearn_available
def test_balanced_accuracy_evaluator_equal_true() -> None:
    assert BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


@sklearn_available
def test_balanced_accuracy_evaluator_equal_false_different_state() -> None:
    assert not BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


@sklearn_available
def test_balanced_accuracy_evaluator_equal_false_different_type() -> None:
    assert not BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(42)


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_correct() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(evaluator.evaluate(), {"balanced_accuracy": 1.0, "count": 5})


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_binary_incorrect() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1]),
            y_pred=np.array([0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(evaluator.evaluate(), {"balanced_accuracy": 0.0, "count": 4})


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_empty() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"balanced_accuracy": float("nan"), "count": 0},
        equal_nan=True,
    )


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_prefix_suffix() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {"prefix_balanced_accuracy_suffix": 1.0, "prefix_count_suffix": 5},
    )


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_omit() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(evaluator.evaluate(), {"balanced_accuracy": 1.0, "count": 5})


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_omit_y_true() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(evaluator.evaluate(), {"balanced_accuracy": 1.0, "count": 5})


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_omit_y_pred() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(evaluator.evaluate(), {"balanced_accuracy": 1.0, "count": 5})


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_propagate() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"balanced_accuracy": float("nan"), "count": 6},
        equal_nan=True,
    )


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_propagate_y_true() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"balanced_accuracy": float("nan"), "count": 6},
        equal_nan=True,
    )


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_propagate_y_pred() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"balanced_accuracy": float("nan"), "count": 6},
        equal_nan=True,
    )


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_raise() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="raise",
        ),
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        evaluator.evaluate()


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_raise_y_true() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="raise",
        ),
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        evaluator.evaluate()


@sklearn_available
def test_balanced_accuracy_evaluator_evaluate_nan_raise_y_pred() -> None:
    evaluator = BalancedAccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="raise",
        ),
    )
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        evaluator.evaluate()


@sklearn_available
def test_balanced_accuracy_evaluator_compute() -> None:
    assert (
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
        .compute()
        .equal(Evaluator({"balanced_accuracy": 1.0, "count": 5}))
    )


@patch("analora.utils.imports.is_sklearn_available", lambda: False)
def test_balanced_accuracy_evaluator_no_sklearn() -> None:
    with pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."):
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
