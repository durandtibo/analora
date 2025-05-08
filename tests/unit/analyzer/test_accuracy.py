from __future__ import annotations

import warnings

import numpy as np
import pytest

from analora.analyzer import AccuracyAnalyzer
from analora.output import EmptyOutput, Output
from analora.state import AccuracyState

######################################
#     Tests for AccuracyAnalyzer     #
######################################


def test_accuracy_analyzer_repr() -> None:
    assert repr(AccuracyAnalyzer(y_true="target", y_pred="pred")).startswith("AccuracyAnalyzer(")


def test_accuracy_analyzer_str() -> None:
    assert str(AccuracyAnalyzer(y_true="target", y_pred="pred")).startswith("AccuracyAnalyzer(")


def test_accuracy_analyzer_analyze() -> None:
    assert (
        AccuracyAnalyzer(y_true="target", y_pred="pred")
        .analyze({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]})
        .equal(
            Output(
                state=AccuracyState(
                    y_true=np.array([1, 2, 3, 2, 1]),
                    y_pred=np.array([3, 2, 0, 1, 0]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        )
    )


def test_accuracy_analyzer_analyze_lazy_false() -> None:
    assert isinstance(
        AccuracyAnalyzer(y_true="target", y_pred="pred").analyze(
            {"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}, lazy=False
        ),
        Output,
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_analyzer_analyze_nan_policy(nan_policy: str) -> None:
    assert (
        AccuracyAnalyzer(y_true="target", y_pred="pred", nan_policy=nan_policy)
        .analyze(
            {
                "pred": [3, 2, 0, 1, 0, None],
                "target": [1, 2, 3, 2, 1, None],
            }
        )
        .equal(
            Output(
                state=AccuracyState(
                    y_true=np.array([1, 2, 3, 2, 1]),
                    y_pred=np.array([3, 2, 0, 1, 0]),
                    y_true_name="target",
                    y_pred_name="pred",
                    nan_policy=nan_policy,
                ),
            )
        )
    )


def test_accuracy_analyzer_analyze_missing_policy_ignore() -> None:
    data = {"col1": np.array([3, 2, 0, 1, 0]), "col2": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(data)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_ignore_y_true() -> None:
    data = {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="gt_target", y_pred="pred", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(data)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_ignore_y_pred() -> None:
    data = {"col": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(data)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_policy_raise() -> None:
    data = {"col1": np.array([3, 2, 0, 1, 0]), "col2": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred")
    with pytest.raises(KeyError, match="column 'target' is missing in the DataFrame"):
        analyzer.analyze(data)


def test_accuracy_analyzer_analyze_missing_raise_y_true() -> None:
    data = {"pred": np.array([3, 2, 0, 1, 0]), "col": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred")
    with pytest.raises(KeyError, match="column 'target' is missing in the DataFrame"):
        analyzer.analyze(data)


def test_accuracy_analyzer_analyze_missing_raise_y_pred() -> None:
    data = {"col": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred")
    with pytest.raises(KeyError, match="column 'pred' is missing in the DataFrame"):
        analyzer.analyze(data)


def test_accuracy_analyzer_analyze_missing_policy_warn() -> None:
    data = {"col1": np.array([3, 2, 0, 1, 0]), "col2": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="warn")
    with (
        pytest.warns(
            RuntimeWarning,
            match="column 'target' is missing in the DataFrame and will be ignored",
        ),
        pytest.warns(
            RuntimeWarning,
            match="column 'pred' is missing in the DataFrame and will be ignored",
        ),
    ):
        out = analyzer.analyze(data)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_warn_y_true() -> None:
    data = {"pred": np.array([3, 2, 0, 1, 0]), "col": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="warn")
    with pytest.warns(
        RuntimeWarning,
        match="column 'target' is missing in the DataFrame and will be ignored",
    ):
        out = analyzer.analyze(data)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_warn_y_pred() -> None:
    data = {"col": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])}
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="warn")
    with pytest.warns(
        RuntimeWarning, match="column 'pred' is missing in the DataFrame and will be ignored"
    ):
        out = analyzer.analyze(data)
    assert out.equal(EmptyOutput())
