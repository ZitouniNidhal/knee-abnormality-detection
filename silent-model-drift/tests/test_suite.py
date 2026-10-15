from __future__ import annotations

import json

import pytest

from silent_model_drift.models import Sentinel, SentinelSuite
from silent_model_drift.suite import load_suite, save_suite


def test_suite_round_trip_preserves_digest(tmp_path) -> None:
    suite = SentinelSuite(
        name="unit-suite",
        sentinels=(Sentinel(id="one", prompt="Say one.", tags=("format",)),),
    )
    path = tmp_path / "suite.json"

    save_suite(suite, path)
    loaded = load_suite(path)

    assert loaded == suite
    assert loaded.digest == suite.digest


def test_duplicate_sentinel_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        SentinelSuite(
            name="bad-suite",
            sentinels=(
                Sentinel(id="duplicate", prompt="A"),
                Sentinel(id="duplicate", prompt="B"),
            ),
        )


def test_invalid_suite_json_is_reported(tmp_path) -> None:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"name": "bad", "sentinels": "not-a-list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="sentinels"):
        load_suite(path)
