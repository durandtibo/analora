from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from analora.evaluator import AccuracyEvaluator, Evaluator
from analora.state import AccuracyState
from analora.testing import sklearn_available

#######################################
#     Tests for AccuracyEvaluator     #
#######################################


@sklearn_available
def test_accuracy_evaluator_repr() -> None:
    assert repr(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("AccuracyEvaluator(")


@sklearn_available
def test_accuracy_evaluator_str() -> None:
    assert str(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("AccuracyEvaluator(")


@sklearn_available
def test_accuracy_evaluator_state() -> None:
    assert AccuracyEvaluator(
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
def test_accuracy_evaluator_equal_true() -> None:
    assert AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


@sklearn_available
def test_accuracy_evaluator_equal_false_different_state() -> None:
    assert not AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


@sklearn_available
def test_accuracy_evaluator_equal_false_different_type() -> None:
    assert not AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(42)


@sklearn_available
def test_accuracy_evaluator_evaluate_binary_correct() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_binary_incorrect() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1]),
            y_pred=np.array([0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 0.0, "count": 4, "count_correct": 0, "count_incorrect": 4, "error": 1.0},
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_multiclass_correct() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 6, "count_correct": 6, "count_incorrect": 0, "error": 0.0},
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_multiclass_incorrect() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([0, 0, 1, 1, 2]),
            y_pred=np.array([0, 0, 1, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {"accuracy": 0.8, "count": 5, "count_correct": 4, "count_incorrect": 1, "error": 0.2},
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_empty() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "accuracy": float("nan"),
            "count": 0,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_prefix_suffix() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_accuracy_suffix": 1.0,
            "prefix_count_suffix": 5,
            "prefix_count_correct_suffix": 5,
            "prefix_count_incorrect_suffix": 0,
            "prefix_error_suffix": 0.0,
        },
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_nan_omit() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_nan_omit_y_true() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_nan_omit_y_pred() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_nan_propagate() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_nan_propagate_y_true() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_nan_propagate_y_pred() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


@sklearn_available
def test_accuracy_evaluator_evaluate_nan_raise() -> None:
    evaluator = AccuracyEvaluator(
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
def test_accuracy_evaluator_evaluate_nan_raise_y_true() -> None:
    evaluator = AccuracyEvaluator(
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
def test_accuracy_evaluator_evaluate_nan_raise_y_pred() -> None:
    evaluator = AccuracyEvaluator(
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
def test_accuracy_evaluator_compute() -> None:
    assert (
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([0, 0, 1, 1, 2, 2]),
                y_pred=np.array([0, 0, 1, 1, 2, 2]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
        .compute()
        .equal(
            Evaluator(
                {
                    "accuracy": 1.0,
                    "count": 6,
                    "count_correct": 6,
                    "count_incorrect": 0,
                    "error": 0.0,
                }
            )
        )
    )


@patch("analora.utils.imports.is_sklearn_available", lambda: False)
def test_accuracy_evaluator_no_sklearn() -> None:
    with pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."):
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
