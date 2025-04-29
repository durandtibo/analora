r"""Contain functions to compute metrics."""

from __future__ import annotations

__all__ = [
    "accuracy",
    "average_precision",
    "balanced_accuracy",
    "binary_average_precision",
    "binary_confusion_matrix",
    "binary_fbeta_score",
    "binary_jaccard",
    "binary_precision",
    "binary_recall",
    "binary_roc_auc",
    "binary_top_k_accuracy",
    "confusion_matrix",
    "energy_distance",
    "fbeta_score",
    "jaccard",
    "jensen_shannon_divergence",
    "kl_div",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "mean_tweedie_deviance",
    "median_absolute_error",
    "multiclass_average_precision",
    "multiclass_confusion_matrix",
    "multiclass_fbeta_score",
    "multiclass_jaccard",
    "multiclass_precision",
    "multiclass_recall",
    "multiclass_roc_auc",
    "multiclass_top_k_accuracy",
    "multilabel_average_precision",
    "multilabel_confusion_matrix",
    "multilabel_fbeta_score",
    "multilabel_jaccard",
    "multilabel_precision",
    "multilabel_recall",
    "multilabel_roc_auc",
    "ndcg",
    "pearsonr",
    "precision",
    "r2_score",
    "recall",
    "regression_errors",
    "roc_auc",
    "root_mean_squared_error",
    "spearmanr",
    "top_k_accuracy",
    "wasserstein_distance",
]


from analora.metric.classification.accuracy import accuracy
from analora.metric.classification.ap import (
    average_precision,
    binary_average_precision,
    multiclass_average_precision,
    multilabel_average_precision,
)
from analora.metric.classification.balanced_accuracy import balanced_accuracy
from analora.metric.classification.confmat import (
    binary_confusion_matrix,
    confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from analora.metric.classification.fbeta import (
    binary_fbeta_score,
    fbeta_score,
    multiclass_fbeta_score,
    multilabel_fbeta_score,
)
from analora.metric.classification.jaccard import (
    binary_jaccard,
    jaccard,
    multiclass_jaccard,
    multilabel_jaccard,
)
from analora.metric.classification.ndcg import ndcg
from analora.metric.classification.precision import (
    binary_precision,
    multiclass_precision,
    multilabel_precision,
    precision,
)
from analora.metric.classification.recall import (
    binary_recall,
    multiclass_recall,
    multilabel_recall,
    recall,
)
from analora.metric.classification.roc_auc import (
    binary_roc_auc,
    multiclass_roc_auc,
    multilabel_roc_auc,
    roc_auc,
)
from analora.metric.classification.topk_accuracy import (
    binary_top_k_accuracy,
    multiclass_top_k_accuracy,
    top_k_accuracy,
)
from analora.metric.correlation.pearson import pearsonr
from analora.metric.correlation.spearman import spearmanr
from analora.metric.distribution.energy import energy_distance
from analora.metric.distribution.jensen_shannon import jensen_shannon_divergence
from analora.metric.distribution.kl import kl_div
from analora.metric.distribution.wasserstein import wasserstein_distance
from analora.metric.regression.abs_error import (
    mean_absolute_error,
    median_absolute_error,
)
from analora.metric.regression.mape import mean_absolute_percentage_error
from analora.metric.regression.mse import mean_squared_error
from analora.metric.regression.msle import mean_squared_log_error
from analora.metric.regression.r2 import r2_score
from analora.metric.regression.rmse import root_mean_squared_error
from analora.metric.regression.tweedie_deviance import mean_tweedie_deviance
from analora.metric.regression.universal import regression_errors
