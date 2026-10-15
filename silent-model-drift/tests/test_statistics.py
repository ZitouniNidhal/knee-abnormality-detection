from __future__ import annotations

import pytest

from silent_model_drift.fingerprint import make_fingerprint
from silent_model_drift.models import (
    CaptureRun,
    Observation,
    RequestConfig,
    Sentinel,
    SentinelSuite,
    utc_now,
)
from silent_model_drift.statistics import (
    DriftConfig,
    analyze_runs,
    analyze_windows,
    benjamini_hochberg,
)


def _fingerprints(texts: list[str]):
    return [make_fingerprint(text) for text in texts]


def _config() -> DriftConfig:
    return DriftConfig(permutations=199, seed=11, minimum_samples_per_window=3)


def test_clear_persistent_change_is_detected() -> None:
    report = analyze_windows(
        {"color": _fingerprints(["blue"] * 8)},
        {"color": _fingerprints(["green"] * 8)},
        config=_config(),
    )

    result = report.sentinels[0]
    assert report.conclusion == "behavioral_drift_detected"
    assert result.classification == "behavioral_shift"
    assert result.mmd_q_value <= 0.05
    assert result.medoid_distance > 0.12
    assert result.center_q_value <= 0.05


def test_equal_windows_are_not_reported_as_drift() -> None:
    samples = _fingerprints(["stable response"] * 8)

    report = analyze_windows({"stable": samples}, {"stable": samples}, config=_config())

    assert report.conclusion == "no_statistically_supported_change"
    assert report.sentinels[0].classification == "stable"


def test_variability_increase_is_labeled_without_medoid_shift() -> None:
    report = analyze_windows(
        {"noise": _fingerprints(["same response"] * 10)},
        {
            "noise": _fingerprints(
                [
                    "same response",
                    "same response",
                    "same response",
                    "same response",
                    "same response",
                    "variant one",
                    "variant two",
                    "variant three",
                    "variant four",
                    "variant five",
                ]
            )
        },
        config=_config(),
    )

    result = report.sentinels[0]
    assert report.conclusion == "nondeterminism_increase_detected"
    assert result.classification == "nondeterminism_increase"
    assert result.medoid_distance == 0.0
    assert result.dispersion_increase is not None and result.dispersion_increase > 0


def test_benjamini_hochberg_is_monotone() -> None:
    adjusted = benjamini_hochberg({"first": 0.01, "second": 0.02, "third": 0.8})

    assert adjusted["first"] <= adjusted["second"] <= adjusted["third"]
    assert adjusted["third"] == pytest.approx(0.8)


def test_incompatible_capture_manifests_are_rejected() -> None:
    suite = SentinelSuite(name="suite", sentinels=(Sentinel(id="one", prompt="one"),))
    observation = Observation(
        sentinel_id="one",
        repeat_index=0,
        collected_at=utc_now(),
        fingerprint=make_fingerprint("one"),
    )
    baseline = CaptureRun.create(
        suite=suite,
        request_config=RequestConfig(provider="replay", model="model-a"),
        repeats=1,
        observations=[observation],
    )
    candidate = CaptureRun.create(
        suite=suite,
        request_config=RequestConfig(provider="replay", model="model-b"),
        repeats=1,
        observations=[observation],
    )

    with pytest.raises(ValueError, match="request configuration"):
        analyze_runs(baseline, candidate, config=_config())
