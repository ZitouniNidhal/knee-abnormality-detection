"""Sentinel-suite loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from .models import SentinelSuite


def load_suite(path: str | Path) -> SentinelSuite:
    """Load a versioned suite from a JSON file."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Suite file does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Suite file is not valid JSON: {source}: {error.msg}") from error
    if not isinstance(raw, dict):
        raise ValueError("Suite root must be a JSON object.")
    return SentinelSuite.from_dict(raw)


def save_suite(suite: SentinelSuite, path: str | Path) -> None:
    """Write a human-readable deterministic JSON suite."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(suite.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
