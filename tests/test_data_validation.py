"""Tests for dataset schema validation and integrity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna_knee.constants import TARGETS
from rsna_knee.data.validate import DatasetValidator


def test_validator_detects_valid_schema(tmp_path: Path) -> None:
    validator = DatasetValidator(tmp_path)
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["study_001", "study_002", "study_003"],
            "SeriesInstanceUID": ["series_01", "series_02", "series_03"],
            **{target: [0, 1, 0] for target in TARGETS},
        }
    )

    validator.validate_metadata(df, split="train")
    error_issues = [issue for issue in validator._issues if issue.severity == "error"]
    assert len(error_issues) == 0


def test_validator_detects_missing_target_columns(tmp_path: Path) -> None:
    validator = DatasetValidator(tmp_path)
    # Missing targets
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["study_001", "study_002"],
            "SeriesInstanceUID": ["series_01", "series_02"],
        }
    )

    validator.validate_metadata(df, split="train")
    error_issues = [issue for issue in validator._issues if issue.severity == "error"]
    assert len(error_issues) > 0
    assert any("Missing required label column" in issue.message for issue in error_issues)
