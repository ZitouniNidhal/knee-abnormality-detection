"""Tests for the medical evaluation metrics."""

from __future__ import annotations

import numpy as np

from rsna_knee.constants import TARGETS
from rsna_knee.evaluation.metrics import MedicalMetricsEvaluator
from rsna_knee.metrics import per_target_auc


def test_per_target_auc_computes_correctly() -> None:
    n_samples = 60
    rng = np.random.default_rng(42)
    # Ensure both 0 and 1 exist in every target column
    y_true = rng.integers(0, 2, size=(n_samples, len(TARGETS)))
    for col in range(len(TARGETS)):
        y_true[0, col] = 0
        y_true[1, col] = 1

    # Add random noise to simulate predicted probabilities
    y_scores = np.clip(y_true * 0.7 + rng.uniform(0.1, 0.3, size=y_true.shape), 0.0, 1.0)

    aucs = per_target_auc(y_true, y_scores)
    assert len(aucs) == len(TARGETS)
    for target in TARGETS:
        assert target in aucs
        assert 0.0 <= aucs[target] <= 1.0


def test_medical_metrics_evaluator_returns_expected_results() -> None:
    n_samples = 50
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 2, size=(n_samples, len(TARGETS)))
    for col in range(len(TARGETS)):
        y_true[0, col] = 0
        y_true[1, col] = 1
    y_scores = rng.uniform(0.0, 1.0, size=(n_samples, len(TARGETS)))

    evaluator = MedicalMetricsEvaluator(threshold=0.5)
    results = evaluator.evaluate(y_true, y_scores, model_name="test_model")

    assert len(results.per_target) == len(TARGETS)
    assert 0.0 <= results.macro_roc_auc <= 1.0
    assert 0.0 <= results.macro_f1 <= 1.0
    assert results.model_name == "test_model"
