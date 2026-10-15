from __future__ import annotations

import pytest

from silent_model_drift.fingerprint import (
    fingerprint_distance,
    make_fingerprint,
    mean_pairwise_distance,
)


def test_whitespace_and_unicode_normalization_produce_same_fingerprint() -> None:
    left = make_fingerprint("Caf\u00e9\n\nblue")
    right = make_fingerprint("Cafe\u0301   blue")

    assert left.exact_hash == right.exact_hash
    assert fingerprint_distance(left, right) == 0.0


def test_json_key_order_is_canonicalized() -> None:
    left = make_fingerprint('{"alpha": 1, "beta": [true, false]}')
    right = make_fingerprint('{"beta":[true,false],"alpha":1}')

    assert left.is_json is True
    assert right.is_json is True
    assert left.exact_hash == right.exact_hash


def test_fingerprint_distance_is_bounded_and_detects_changes() -> None:
    left = make_fingerprint("The answer is blue.")
    right = make_fingerprint("A longer response says the answer is green today.")

    assert 0.0 < fingerprint_distance(left, right) <= 1.0
    assert mean_pairwise_distance([left]) is None
    assert mean_pairwise_distance([left, right]) is not None


def test_non_string_responses_are_rejected() -> None:
    with pytest.raises(TypeError):
        make_fingerprint(42)  # type: ignore[arg-type]
