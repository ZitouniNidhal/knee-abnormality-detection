"""Safe creation and validation of competition submission files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, N_TARGETS, REQUIRED_SUBMISSION_COLUMNS, TARGETS
from .metadata import require_unique_ids, study_id_column


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Convert scores to [0, 1] ranks, preserving per-target AUC ordering.

    Averaging ranks is more robust than averaging poorly calibrated fold logits;
    macro ROC-AUC only depends on within-label ordering.  Constant predictions
    intentionally become 0.5 rather than generating NaNs.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_TARGETS:
        raise ValueError(f"Expected (n, {N_TARGETS}) predictions, got {values.shape!r}.")
    count = len(values)
    if count == 0:
        return values.astype(np.float32)
    output = np.empty_like(values, dtype=np.float64)
    for target_index in range(N_TARGETS):
        column = values[:, target_index]
        if not np.isfinite(column).all():
            raise ValueError(f"Non-finite prediction found for {TARGETS[target_index]!r}.")
        order = np.argsort(column, kind="mergesort")
        sorted_values = column[order]
        ranks = np.empty(count, dtype=np.float64)
        start = 0
        while start < count:
            end = start + 1
            while end < count and sorted_values[end] == sorted_values[start]:
                end += 1
            # Average rank gives tied studies exactly the same probability.
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        output[:, target_index] = 0.5 if count == 1 else ranks / (count - 1)
    return output.astype(np.float32)


def make_submission(
    test_frame: pd.DataFrame,
    predictions: np.ndarray | Sequence[Sequence[float]],
    *,
    apply_rank_normalization: bool = False,
) -> pd.DataFrame:
    """Build an official-schema frame from test metadata and model scores."""
    id_column = study_id_column(test_frame)
    require_unique_ids(test_frame, id_column)
    predictions = np.asarray(predictions, dtype=np.float64)
    expected_shape = (len(test_frame), N_TARGETS)
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Prediction shape must be {expected_shape!r}, got {predictions.shape!r}."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinity.")
    if apply_rank_normalization:
        predictions = rank_normalize(predictions)
    else:
        predictions = np.clip(predictions, 0.0, 1.0)

    submission = pd.DataFrame(predictions, columns=TARGETS)
    submission.insert(0, ID_COLUMN, test_frame[id_column].astype(str).to_numpy())
    validate_submission(submission, expected_ids=test_frame[id_column].astype(str).tolist())
    return submission


def validate_submission(
    submission: pd.DataFrame,
    *,
    expected_ids: Sequence[str] | None = None,
) -> None:
    """Fail fast on every requirement that would invalidate a Kaggle upload."""
    columns = tuple(submission.columns)
    if columns != REQUIRED_SUBMISSION_COLUMNS:
        raise ValueError(
            "Submission columns must exactly equal "
            f"{REQUIRED_SUBMISSION_COLUMNS!r}; received {columns!r}."
        )
    require_unique_ids(submission, ID_COLUMN)
    values = submission.loc[:, TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Submission has NaN or infinite confidence scores.")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("Submission confidence scores must lie in [0, 1].")
    if expected_ids is not None:
        actual = submission[ID_COLUMN].astype(str).tolist()
        expected = [str(value) for value in expected_ids]
        if actual != expected:
            raise ValueError(
                "Submission IDs do not match test.csv in its original order. "
                "Do not sort either file before submitting."
            )


def write_submission(
    submission: pd.DataFrame,
    output_path: str | Path = "submission.csv",
    *,
    expected_ids: Sequence[str] | None = None,
) -> Path:
    """Validate then write the exactly named CSV required by the competition."""
    validate_submission(submission, expected_ids=expected_ids)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, float_format="%.8f")
    return output_path
