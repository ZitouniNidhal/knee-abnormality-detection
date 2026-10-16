"""Schema-tolerant helpers for the competition CSV files.

Kaggle occasionally varies spelling/case in metadata exports.  The functions
here resolve columns by normalized aliases and refuse ambiguous inputs instead
of producing a submission with a silently wrong study/series pairing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .constants import ID_COLUMN, TARGETS


def normalized_name(value: object) -> str:
    """Normalize a CSV header or metadata value for case-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def resolve_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    """Return the unique actual column matching one of *candidates*.

    Matching is deliberately exact after punctuation/case normalization.  A
    useful error is safer than guessing between two similarly named columns.
    """
    aliases = {normalized_name(candidate) for candidate in candidates}
    matches = [column for column in frame.columns if normalized_name(column) in aliases]
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    if not matches:
        raise KeyError(
            f"Could not find one of {list(candidates)!r}. Available columns: "
            f"{list(frame.columns)!r}"
        )
    raise KeyError(f"Ambiguous aliases {list(candidates)!r}; matched columns {matches!r}.")


def read_competition_csv(path: str | Path) -> pd.DataFrame:
    """Read a competition CSV without corrupting multilingual reports.

    The official training metadata has appeared in Latin-1.  The fallback is
    important only for train.csv; it is harmless for ASCII test metadata.
    """
    path = Path(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def study_id_column(frame: pd.DataFrame) -> str:
    return resolve_column(
        frame,
        (ID_COLUMN, "study_uid", "studyinstanceuid", "study_id"),
    )  # type: ignore[return-value]


def series_id_column(frame: pd.DataFrame) -> str:
    return resolve_column(
        frame,
        ("SeriesInstanceUID", "series_uid", "seriesinstanceuid", "series_id"),
    )  # type: ignore[return-value]


def report_column(frame: pd.DataFrame, *, required: bool = False) -> str | None:
    return resolve_column(
        frame,
        ("Report", "report_text", "radiology_report", "findings"),
        required=required,
    )


def target_columns(frame: pd.DataFrame, *, required: bool = True) -> list[str]:
    """Return target columns in the immutable official order.

    Header normalization accepts harmless formatting differences while the
    returned names retain the exact CSV spelling for pandas indexing.
    """
    resolved: list[str] = []
    for target in TARGETS:
        column = resolve_column(frame, (target,), required=required)
        if column is None:
            return []
        resolved.append(column)
    return resolved


def require_unique_ids(frame: pd.DataFrame, id_column: str | None = None) -> None:
    id_column = id_column or study_id_column(frame)
    if frame[id_column].isna().any():
        raise ValueError(f"{id_column!r} contains missing study identifiers.")
    duplicates = frame[id_column].duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, id_column].astype(str).head(5).tolist()
        raise ValueError(
            f"{id_column!r} must be unique for submission rows; examples: {examples!r}"
        )
