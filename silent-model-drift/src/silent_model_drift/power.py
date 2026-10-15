"""Transparent approximate planning tools for sentinel-suite sample sizes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

_NORMAL = NormalDist()


@dataclass(frozen=True)
class PowerPlan:
    """A planning result with assumptions exposed for review."""

    alpha: float
    power: float
    standardized_effect: float
    independent_observations_per_window: int
    repeats_per_sentinel: int
    intra_prompt_correlation: float
    effective_observations_per_sentinel: float
    required_sentinels: int

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-compatible values for a CLI report."""

        return asdict(self)


@dataclass(frozen=True)
class ProportionPowerPlan:
    """Initial planning result for a binary task-success metric."""

    alpha: float
    power: float
    baseline_rate: float
    candidate_rate: float
    observations_per_window: int

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-compatible values for a CLI report."""

        return asdict(self)


def _validate_probability(value: float, label: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{label} must be between 0 and 1.")


def independent_samples_per_window(
    standardized_effect: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Approximate equal-sized two-sample observations required in each window.

    This formula is appropriate for a mean-like standardized effect. It is a
    planning approximation, not a closed-form power calculation for MMD.
    """

    if standardized_effect <= 0:
        raise ValueError("standardized_effect must be positive.")
    _validate_probability(alpha, "alpha")
    _validate_probability(power, "power")
    z_alpha = _NORMAL.inv_cdf(1 - alpha / 2)
    z_power = _NORMAL.inv_cdf(power)
    return math.ceil(2 * ((z_alpha + z_power) / standardized_effect) ** 2)


def plan_sentinels(
    standardized_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    repeats_per_sentinel: int = 1,
    intra_prompt_correlation: float = 0.0,
) -> PowerPlan:
    """Convert independent-observation needs to an inflation-aware sentinel count."""

    if repeats_per_sentinel < 1:
        raise ValueError("repeats_per_sentinel must be at least one.")
    if not 0 <= intra_prompt_correlation < 1:
        raise ValueError("intra_prompt_correlation must be in [0, 1).")
    independent = independent_samples_per_window(
        standardized_effect,
        alpha=alpha,
        power=power,
    )
    effective_per_sentinel = repeats_per_sentinel / (
        1 + (repeats_per_sentinel - 1) * intra_prompt_correlation
    )
    return PowerPlan(
        alpha=alpha,
        power=power,
        standardized_effect=standardized_effect,
        independent_observations_per_window=independent,
        repeats_per_sentinel=repeats_per_sentinel,
        intra_prompt_correlation=intra_prompt_correlation,
        effective_observations_per_sentinel=effective_per_sentinel,
        required_sentinels=math.ceil(independent / effective_per_sentinel),
    )


def samples_per_window_for_proportions(
    baseline_rate: float,
    candidate_rate: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> ProportionPowerPlan:
    """Approximate equal-size two-proportion sample requirement per window."""

    _validate_probability(baseline_rate, "baseline_rate")
    _validate_probability(candidate_rate, "candidate_rate")
    _validate_probability(alpha, "alpha")
    _validate_probability(power, "power")
    difference = abs(candidate_rate - baseline_rate)
    if difference == 0:
        raise ValueError("Baseline and candidate rates must differ.")
    pooled = (baseline_rate + candidate_rate) / 2
    z_alpha = _NORMAL.inv_cdf(1 - alpha / 2)
    z_power = _NORMAL.inv_cdf(power)
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power
        * math.sqrt(baseline_rate * (1 - baseline_rate) + candidate_rate * (1 - candidate_rate))
    ) ** 2
    return ProportionPowerPlan(
        alpha=alpha,
        power=power,
        baseline_rate=baseline_rate,
        candidate_rate=candidate_rate,
        observations_per_window=math.ceil(numerator / difference**2),
    )
