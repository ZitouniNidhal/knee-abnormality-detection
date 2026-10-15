"""Canonicalization primitives used before response fingerprinting."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_TOKENS = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)


@dataclass(frozen=True)
class CanonicalResponse:
    """Normalized response text plus a coarse structural description."""

    text: str
    is_json: bool
    structure: str


def normalize_text(text: str) -> str:
    """Normalize Unicode, line endings, and whitespace without erasing case changes."""

    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return _WHITESPACE.sub(" ", normalized).strip()


def _json_shape(value: Any) -> str:
    """Return a bounded structural representation without retaining JSON values."""

    if isinstance(value, dict):
        children = [
            f"{json.dumps(str(key), ensure_ascii=False)}:{_json_shape(item)}"
            for key, item in value.items()
        ]
        return "object{" + ",".join(sorted(children)) + "}"
    if isinstance(value, list):
        child_shapes = sorted({_json_shape(item) for item in value})
        return "array[" + ",".join(child_shapes) + "]"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def canonicalize_response(text: str) -> CanonicalResponse:
    """Canonicalize text and canonical JSON when the entire response is JSON."""

    if not isinstance(text, str):
        raise TypeError("A response must be a string.")
    normalized = normalize_text(text)
    try:
        decoded = json.loads(normalized)
    except json.JSONDecodeError:
        return CanonicalResponse(text=normalized, is_json=False, structure="plain-text")
    return CanonicalResponse(
        text=json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        is_json=True,
        structure=_json_shape(decoded),
    )


def tokenize(text: str) -> tuple[str, ...]:
    """Return distinct lexical tokens from an already canonical text value."""

    return tuple(sorted(set(_TOKENS.findall(text.casefold()))))


def character_ngrams(text: str, width: int = 3) -> tuple[str, ...]:
    """Return distinct padded character n-grams for a compact lexical sketch."""

    if width < 1:
        raise ValueError("Character n-gram width must be positive.")
    padded = f"{'^' * (width - 1)}{text}{'$' * (width - 1)}"
    if not padded:
        return ()
    return tuple(
        sorted({padded[index : index + width] for index in range(len(padded) - width + 1)})
    )
