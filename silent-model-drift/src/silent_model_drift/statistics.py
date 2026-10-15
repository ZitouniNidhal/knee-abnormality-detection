"""Distributional drift tests for repeated sentinel responses.

The implementation is deliberately dependency-free. It uses random-label
permutations instead of relying on a normal approximation for LLM output.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from statistics import median

from .fingerprint import (
    fingerprint_distance,
    gaussian_kernel,
    mean_pairwise_distance,
    medoid_distance,
)
from .models import CaptureRun, ResponseFingerprint


@dataclass(frozen=True)
class DriftConfig:
    """Pre-registered choices controlling a comparison run."""

    alpha: float = 0.05
    permutations: int = 999
    seed: int = 2026
    minimum_effect: float = 0.05
    minimum_center_shift: float = 0.12
    minimum_dispersion_increase: float = 0.05
    minimum_samples_per_window: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if self.permutations < 99:
            raise ValueError("At least 99 permutations are required for a useful p-value.")
        if self.minimum_effect < 0 or self.minimum_center_shift < 0:
            raise ValueError("Effect thresholds cannot be negative.")
        if self.minimum_dispersion_increase < 0:
            raise ValueError("minimum_dispersion_increase cannot be negative.")
        if self.minimum_samples_per_window < 1:
            raise ValueError("minimum_samples_per_window must be at least one.")


@dataclass(frozen=True)
class SentinelResult:
    """Statistical evidence for one fixed sentinel prompt."""

    sentinel_id: str
    baseline_samples: int
    candidate_samples: int
    mmd2: float
    mmd_p_value: float
    mmd_q_value: float
    center_separation: float
    center_p_value: float
    center_q_value: float
    medoid_distance: float
    baseline_dispersion: float | None
    candidate_dispersion: float | None
    dispersion_increase: float | None
    dispersion_p_value: float | None
    dispersion_q_value: float | None
    classification: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible output with explicit field names."""

        return asdict(self)


@dataclass(frozen=True)
class DriftReport:
    """A complete comparison result suitable for persistence or CI output."""

    conclusion: str
    global_mmd2: float
    global_p_value: float
    fisher_p_value: float
    config: DriftConfig
    sentinels: tuple[SentinelResult, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible report."""

        return {
            "conclusion": self.conclusion,
            "global_mmd2": self.global_mmd2,
            "global_p_value": self.global_p_value,
            "fisher_p_value": self.fisher_p_value,
            "config": asdict(self.config),
            "warnings": list(self.warnings),
            "sentinels": [result.to_dict() for result in self.sentinels],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report deterministically."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _seed_for(seed: int, label: str) -> int:
    """Derive independent stable random streams for each test."""

    payload = f"{seed}:{label}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _median_bandwidth(values: Sequence[ResponseFingerprint]) -> float:
    """Apply the median heuristic with a safe fallback for duplicate outputs."""

    distances = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            distance = fingerprint_distance(values[left], values[right])
            if distance > 0:
                distances.append(distance)
    distances.sort()
    return median(distances) if distances else 1.0


def mmd2(
    baseline: Sequence[ResponseFingerprint], candidate: Sequence[ResponseFingerprint]
) -> float:
    """Calculate biased RBF-kernel MMD squared between two non-empty windows."""

    if not baseline or not candidate:
        raise ValueError("MMD requires non-empty baseline and candidate windows.")
    pooled = [*baseline, *candidate]
    bandwidth = _median_bandwidth(pooled)

    def average_kernel(
        left: Sequence[ResponseFingerprint], right: Sequence[ResponseFingerprint]
    ) -> float:
        return sum(
            gaussian_kernel(fingerprint_distance(item_left, item_right), bandwidth)
            for item_left in left
            for item_right in right
        ) / (len(left) * len(right))

    value = average_kernel(baseline, baseline) + average_kernel(candidate, candidate)
    value -= 2 * average_kernel(baseline, candidate)
    return max(0.0, value)


def _permutation_p_value(
    baseline: Sequence[ResponseFingerprint],
    candidate: Sequence[ResponseFingerprint],
    statistic: Callable[[Sequence[ResponseFingerprint], Sequence[ResponseFingerprint]], float],
    *,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Return an upper-tail random-label p-value and the observed statistic."""

    if not baseline or not candidate:
        raise ValueError("A permutation test needs non-empty windows.")
    observed = statistic(baseline, candidate)
    pooled = [*baseline, *candidate]
    baseline_count = len(baseline)
    indices = list(range(len(pooled)))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        rng.shuffle(indices)
        permuted_baseline = [pooled[index] for index in indices[:baseline_count]]
        permuted_candidate = [pooled[index] for index in indices[baseline_count:]]
        if statistic(permuted_baseline, permuted_candidate) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (permutations + 1), observed


def _dispersion_increase(
    baseline: Sequence[ResponseFingerprint], candidate: Sequence[ResponseFingerprint]
) -> float:
    """Calculate candidate-minus-baseline within-window diversity."""

    baseline_value = mean_pairwise_distance(baseline)
    candidate_value = mean_pairwise_distance(candidate)
    if baseline_value is None or candidate_value is None:
        raise ValueError("A dispersion comparison requires at least two samples per window.")
    return candidate_value - baseline_value


def _center_separation(
    baseline: Sequence[ResponseFingerprint], candidate: Sequence[ResponseFingerprint]
) -> float:
    """Return average cross-window distance as a robust central-separation statistic."""

    return sum(
        fingerprint_distance(baseline_item, candidate_item)
        for baseline_item in baseline
        for candidate_item in candidate
    ) / (len(baseline) * len(candidate))


def _one_sided_dispersion_test(
    baseline: Sequence[ResponseFingerprint],
    candidate: Sequence[ResponseFingerprint],
    *,
    permutations: int,
    seed: int,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Test whether candidate nondeterminism increased relative to baseline."""

    baseline_value = mean_pairwise_distance(baseline)
    candidate_value = mean_pairwise_distance(candidate)
    if baseline_value is None or candidate_value is None:
        return baseline_value, candidate_value, None, None, None
    p_value, increase = _permutation_p_value(
        baseline,
        candidate,
        _dispersion_increase,
        permutations=permutations,
        seed=seed,
    )
    if increase <= 0:
        p_value = 1.0
    return baseline_value, candidate_value, increase, p_value, p_value


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Adjust p-values using the Benjamini-Hochberg false-discovery-rate procedure."""

    if not p_values:
        return {}
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("P-values must lie in [0, 1].")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for index in range(count - 1, -1, -1):
        label, value = ordered[index]
        rank = index + 1
        running = min(running, value * count / rank)
        adjusted[label] = min(1.0, running)
    return adjusted


def fisher_combined_p_value(p_values: Sequence[float]) -> float:
    """Combine independent-looking p-values with Fisher's method.

    The survival function has a closed form because Fisher's statistic has an
    even number of chi-square degrees of freedom. Correlated sentinels make
    this a descriptive summary rather than the primary alerting statistic.
    """

    if not p_values:
        return 1.0
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("P-values must lie in [0, 1].")
    statistic = -2 * sum(math.log(max(value, 1e-300)) for value in p_values)
    half_statistic = statistic / 2
    terms = sum(half_statistic**index / math.factorial(index) for index in range(len(p_values)))
    return min(1.0, math.exp(-half_statistic) * terms)


def _global_permutation_test(
    pairs: Mapping[str, tuple[Sequence[ResponseFingerprint], Sequence[ResponseFingerprint]]],
    *,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Run one global test while shuffling labels independently within each sentinel."""

    if not pairs:
        return 0.0, 1.0

    def suite_statistic(
        selected: Mapping[str, tuple[Sequence[ResponseFingerprint], Sequence[ResponseFingerprint]]],
    ) -> float:
        return sum(mmd2(baseline, candidate) for baseline, candidate in selected.values()) / len(
            selected
        )

    observed = suite_statistic(pairs)
    rng = random.Random(seed)
    state: dict[str, tuple[list[ResponseFingerprint], int]] = {}
    for sentinel_id, (baseline, candidate) in pairs.items():
        state[sentinel_id] = ([*baseline, *candidate], len(baseline))

    extreme = 0
    for _ in range(permutations):
        permuted: dict[str, tuple[list[ResponseFingerprint], list[ResponseFingerprint]]] = {}
        for sentinel_id, (pooled, baseline_count) in state.items():
            indexes = list(range(len(pooled)))
            rng.shuffle(indexes)
            permuted[sentinel_id] = (
                [pooled[index] for index in indexes[:baseline_count]],
                [pooled[index] for index in indexes[baseline_count:]],
            )
        if suite_statistic(permuted) >= observed - 1e-12:
            extreme += 1
    return observed, (extreme + 1) / (permutations + 1)


def analyze_windows(
    baseline_by_sentinel: Mapping[str, Sequence[ResponseFingerprint]],
    candidate_by_sentinel: Mapping[str, Sequence[ResponseFingerprint]],
    *,
    config: DriftConfig | None = None,
) -> DriftReport:
    """Compare repeated baseline and candidate windows without raw response text."""

    settings = config or DriftConfig()
    baseline_ids = set(baseline_by_sentinel)
    candidate_ids = set(candidate_by_sentinel)
    shared_ids = sorted(baseline_ids & candidate_ids)
    if not shared_ids:
        raise ValueError("Baseline and candidate windows have no shared sentinel ids.")

    warnings: list[str] = []
    only_baseline = sorted(baseline_ids - candidate_ids)
    only_candidate = sorted(candidate_ids - baseline_ids)
    if only_baseline:
        warnings.append("Missing candidate samples for: " + ", ".join(only_baseline))
    if only_candidate:
        warnings.append("Unexpected candidate-only samples for: " + ", ".join(only_candidate))

    raw_rows: list[dict[str, object]] = []
    global_pairs: dict[
        str, tuple[Sequence[ResponseFingerprint], Sequence[ResponseFingerprint]]
    ] = {}
    for sentinel_id in shared_ids:
        baseline = list(baseline_by_sentinel[sentinel_id])
        candidate = list(candidate_by_sentinel[sentinel_id])
        if not baseline or not candidate:
            warnings.append(f"Skipped {sentinel_id}: an empty response window cannot be tested.")
            continue
        if (
            len(baseline) < settings.minimum_samples_per_window
            or len(candidate) < settings.minimum_samples_per_window
        ):
            warnings.append(
                f"{sentinel_id} has fewer than "
                f"{settings.minimum_samples_per_window} samples in a window; "
                "its p-value has low resolution."
            )
        mmd_p_value, observed_mmd2 = _permutation_p_value(
            baseline,
            candidate,
            mmd2,
            permutations=settings.permutations,
            seed=_seed_for(settings.seed, f"mmd:{sentinel_id}"),
        )
        center_p_value, observed_center_separation = _permutation_p_value(
            baseline,
            candidate,
            _center_separation,
            permutations=settings.permutations,
            seed=_seed_for(settings.seed, f"center:{sentinel_id}"),
        )
        observed_medoid_distance = medoid_distance(baseline, candidate)
        (
            baseline_dispersion,
            candidate_dispersion,
            dispersion_increase,
            dispersion_p_value,
            _,
        ) = _one_sided_dispersion_test(
            baseline,
            candidate,
            permutations=settings.permutations,
            seed=_seed_for(settings.seed, f"dispersion:{sentinel_id}"),
        )
        raw_rows.append(
            {
                "sentinel_id": sentinel_id,
                "baseline_samples": len(baseline),
                "candidate_samples": len(candidate),
                "mmd2": observed_mmd2,
                "mmd_p_value": mmd_p_value,
                "center_separation": observed_center_separation,
                "center_p_value": center_p_value,
                "medoid_distance": observed_medoid_distance,
                "baseline_dispersion": baseline_dispersion,
                "candidate_dispersion": candidate_dispersion,
                "dispersion_increase": dispersion_increase,
                "dispersion_p_value": dispersion_p_value,
            }
        )
        global_pairs[sentinel_id] = (baseline, candidate)

    if not raw_rows:
        raise ValueError("No shared sentinel has non-empty baseline and candidate windows.")

    mmd_q_values = benjamini_hochberg(
        {str(row["sentinel_id"]): float(row["mmd_p_value"]) for row in raw_rows}
    )
    center_q_values = benjamini_hochberg(
        {str(row["sentinel_id"]): float(row["center_p_value"]) for row in raw_rows}
    )
    dispersion_q_values = benjamini_hochberg(
        {
            str(row["sentinel_id"]): float(row["dispersion_p_value"])
            for row in raw_rows
            if row["dispersion_p_value"] is not None
        }
    )

    results: list[SentinelResult] = []
    for row in raw_rows:
        sentinel_id = str(row["sentinel_id"])
        mmd2_value = float(row["mmd2"])
        mmd_q_value = mmd_q_values[sentinel_id]
        center_separation = float(row["center_separation"])
        center_q_value = center_q_values[sentinel_id]
        medoid_value = float(row["medoid_distance"])
        dispersion_increase = row["dispersion_increase"]
        dispersion_q_value = dispersion_q_values.get(sentinel_id)
        distribution_changed = (
            mmd_q_value <= settings.alpha and mmd2_value >= settings.minimum_effect
        )
        central_behavior_changed = (
            center_q_value <= settings.alpha
            and center_separation >= settings.minimum_center_shift
            and medoid_value >= settings.minimum_center_shift
        )
        variability_increased = (
            dispersion_increase is not None
            and dispersion_q_value is not None
            and float(dispersion_increase) >= settings.minimum_dispersion_increase
            and dispersion_q_value <= settings.alpha
        )
        if distribution_changed and central_behavior_changed:
            classification = "behavioral_shift"
        elif distribution_changed and variability_increased and not central_behavior_changed:
            classification = "nondeterminism_increase"
        elif distribution_changed:
            classification = "distribution_shift"
        else:
            classification = "stable"
        results.append(
            SentinelResult(
                sentinel_id=sentinel_id,
                baseline_samples=int(row["baseline_samples"]),
                candidate_samples=int(row["candidate_samples"]),
                mmd2=mmd2_value,
                mmd_p_value=float(row["mmd_p_value"]),
                mmd_q_value=mmd_q_value,
                center_separation=center_separation,
                center_p_value=float(row["center_p_value"]),
                center_q_value=center_q_value,
                medoid_distance=medoid_value,
                baseline_dispersion=row["baseline_dispersion"],
                candidate_dispersion=row["candidate_dispersion"],
                dispersion_increase=dispersion_increase,
                dispersion_p_value=row["dispersion_p_value"],
                dispersion_q_value=dispersion_q_value,
                classification=classification,
            )
        )

    global_mmd2, global_p_value = _global_permutation_test(
        global_pairs,
        permutations=settings.permutations,
        seed=_seed_for(settings.seed, "global"),
    )
    fisher_p_value = fisher_combined_p_value([result.mmd_p_value for result in results])
    classifications = {result.classification for result in results}
    global_change = global_p_value <= settings.alpha and global_mmd2 >= settings.minimum_effect
    if global_change and "behavioral_shift" in classifications:
        conclusion = "behavioral_drift_detected"
    elif (
        global_change
        and "nondeterminism_increase" in classifications
        and classifications
        <= {
            "stable",
            "nondeterminism_increase",
        }
    ):
        conclusion = "nondeterminism_increase_detected"
    elif global_change and classifications - {"stable"}:
        conclusion = "behavioral_distribution_shift_detected"
    elif "behavioral_shift" in classifications or "distribution_shift" in classifications:
        conclusion = "localized_change_detected"
    elif "nondeterminism_increase" in classifications:
        conclusion = "localized_nondeterminism_increase_detected"
    else:
        conclusion = "no_statistically_supported_change"

    return DriftReport(
        conclusion=conclusion,
        global_mmd2=global_mmd2,
        global_p_value=global_p_value,
        fisher_p_value=fisher_p_value,
        config=settings,
        sentinels=tuple(results),
        warnings=tuple(warnings),
    )


def analyze_runs(
    baseline: CaptureRun,
    candidate: CaptureRun,
    *,
    config: DriftConfig | None = None,
    require_matching_manifests: bool = True,
) -> DriftReport:
    """Compare two persisted captures after enforcing measurement compatibility."""

    if require_matching_manifests:
        problems: list[str] = []
        if baseline.manifest.suite_digest != candidate.manifest.suite_digest:
            problems.append("suite digest")
        if baseline.manifest.request_config_digest != candidate.manifest.request_config_digest:
            problems.append("request configuration digest")
        if problems:
            raise ValueError(
                "Cannot compare incompatible captures; mismatched " + ", ".join(problems) + "."
            )
    return analyze_windows(baseline.by_sentinel(), candidate.by_sentinel(), config=config)
