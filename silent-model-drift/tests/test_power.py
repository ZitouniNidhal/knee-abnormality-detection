from __future__ import annotations

from silent_model_drift.power import (
    independent_samples_per_window,
    plan_sentinels,
    samples_per_window_for_proportions,
)


def test_smaller_effect_requires_more_samples() -> None:
    assert independent_samples_per_window(0.3) > independent_samples_per_window(0.8)


def test_repeated_correlated_calls_have_diminishing_returns() -> None:
    independent = plan_sentinels(0.5, repeats_per_sentinel=5, intra_prompt_correlation=0.0)
    correlated = plan_sentinels(0.5, repeats_per_sentinel=5, intra_prompt_correlation=0.8)

    assert (
        correlated.effective_observations_per_sentinel
        < independent.effective_observations_per_sentinel
    )
    assert correlated.required_sentinels > independent.required_sentinels


def test_larger_proportion_difference_requires_fewer_observations() -> None:
    small_change = samples_per_window_for_proportions(0.50, 0.60)
    large_change = samples_per_window_for_proportions(0.50, 0.80)

    assert large_change.observations_per_window < small_change.observations_per_window
