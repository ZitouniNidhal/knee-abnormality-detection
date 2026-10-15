"""Privacy-conscious response fingerprints and transparent distances."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from hashlib import sha256
from statistics import mean

from .canonicalize import canonicalize_response, character_ngrams, tokenize
from .models import ResponseFingerprint


def _hash(value: str) -> str:
    """Return a stable digest for one feature value."""

    return sha256(value.encode("utf-8")).hexdigest()


def _feature_hashes(values: Iterable[str], limit: int = 256) -> tuple[str, ...]:
    """Hash and cap a feature set deterministically to bound artifact size."""

    if limit <= 0:
        raise ValueError("Feature hash limit must be positive.")
    return tuple(sorted(_hash(value) for value in values)[:limit])


def make_fingerprint(
    text: str, *, ngram_width: int = 3, feature_limit: int = 256
) -> ResponseFingerprint:
    """Build the default fingerprint without retaining the original response text."""

    canonical = canonicalize_response(text)
    token_values = tokenize(canonical.text)
    ngram_values = character_ngrams(canonical.text, width=ngram_width)
    return ResponseFingerprint(
        exact_hash=_hash(canonical.text),
        token_hashes=_feature_hashes(token_values, limit=feature_limit),
        ngram_hashes=_feature_hashes(ngram_values, limit=feature_limit),
        character_count=len(canonical.text),
        token_count=len(token_values),
        structure_hash=_hash(canonical.structure),
        is_json=canonical.is_json,
    )


def _jaccard_distance(left: Sequence[str], right: Sequence[str]) -> float:
    """Return set Jaccard distance with well-defined empty-set behavior."""

    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(left_set | right_set)


def fingerprint_distance(left: ResponseFingerprint, right: ResponseFingerprint) -> float:
    """Return a bounded hybrid distance between two response fingerprints."""

    if left.version != right.version:
        raise ValueError("Cannot compare fingerprints from different schema versions.")
    if left.exact_hash == right.exact_hash:
        return 0.0
    token_distance = _jaccard_distance(left.token_hashes, right.token_hashes)
    ngram_distance = _jaccard_distance(left.ngram_hashes, right.ngram_hashes)
    denominator = max(left.character_count, right.character_count, 1)
    length_distance = abs(left.character_count - right.character_count) / denominator
    structure_distance = float(
        left.structure_hash != right.structure_hash or left.is_json != right.is_json
    )
    return min(
        1.0,
        0.40 * token_distance
        + 0.35 * ngram_distance
        + 0.15 * length_distance
        + 0.10 * structure_distance,
    )


def mean_pairwise_distance(values: Sequence[ResponseFingerprint]) -> float | None:
    """Return mean within-window distance, or None when a window has fewer than two samples."""

    if len(values) < 2:
        return None
    distances = [
        fingerprint_distance(values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return mean(distances)


def medoid(values: Sequence[ResponseFingerprint]) -> ResponseFingerprint:
    """Return the observed response with minimum mean distance to its window."""

    if not values:
        raise ValueError("Cannot compute a medoid of an empty window.")
    scores = [sum(fingerprint_distance(current, other) for other in values) for current in values]
    return values[min(range(len(values)), key=scores.__getitem__)]


def medoid_distance(
    left: Sequence[ResponseFingerprint], right: Sequence[ResponseFingerprint]
) -> float:
    """Return the distance between observed baseline and candidate medoids."""

    return fingerprint_distance(medoid(left), medoid(right))


def gaussian_kernel(distance: float, bandwidth: float) -> float:
    """Map a nonnegative distance to an RBF similarity."""

    if bandwidth <= 0:
        raise ValueError("Kernel bandwidth must be positive.")
    return math.exp(-distance / bandwidth)
