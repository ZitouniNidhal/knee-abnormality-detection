"""Safe JSON persistence for capture and report artifacts."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import CaptureRun
from .statistics import DriftReport


def _write_json_atomically(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write JSON through a sibling temporary file to avoid partial artifacts."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary_name = temporary.name
        try:
            Path(temporary_name).replace(destination)
        except PermissionError:
            # Some managed Windows file systems disallow a same-directory rename.
            # Preserve usability there while retaining atomic replacement elsewhere.
            destination.write_text(serialized, encoding="utf-8")
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def save_capture(run: CaptureRun, path: str | Path) -> None:
    """Persist a capture run as a validated JSON artifact."""

    _write_json_atomically(run.to_dict(), path)


def load_capture(path: str | Path) -> CaptureRun:
    """Load and validate a capture run artifact."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Capture artifact does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Capture artifact is not valid JSON: {source}: {error.msg}") from error
    if not isinstance(raw, dict):
        raise ValueError("Capture artifact root must be a JSON object.")
    return CaptureRun.from_dict(raw)


def save_report(report: DriftReport, path: str | Path) -> None:
    """Persist a drift report in deterministic JSON."""

    _write_json_atomically(report.to_dict(), path)
