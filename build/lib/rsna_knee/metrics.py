"""Leakage-safe multilabel validation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .constants import N_TARGETS, TARGETS


def per_target_auc(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute AUC per label, returning NaN when a fold has one class only."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.shape != scores.shape or y_true.ndim != 2 or y_true.shape[1] != N_TARGETS:
        raise ValueError(
            f"Expected matching (n, {N_TARGETS}) y_true/scores arrays; got "
            f"{y_true.shape!r} and {scores.shape!r}."
        )
    result: dict[str, float] = {}
    for index, target in enumerate(TARGETS):
        truth = y_true[:, index]
        if len(np.unique(truth)) < 2:
            result[target] = float("nan")
        else:
            result[target] = float(roc_auc_score(truth, scores[:, index]))
    return result


def macro_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Competition metric: unweighted mean of the twelve ROC-AUC values."""
    values = np.asarray(list(per_target_auc(y_true, scores).values()), dtype=float)
    if np.isnan(values).any():
        missing = [target for target, value in zip(TARGETS, values) if np.isnan(value)]
        raise ValueError(
            "Cannot compute a reliable macro AUC because validation contains only "
            f"one class for {missing!r}. Use grouped repeated CV or OOF predictions."
        )
    return float(values.mean())
