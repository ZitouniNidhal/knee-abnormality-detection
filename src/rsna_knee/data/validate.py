"""Enterprise-grade dataset validation and integrity checking.

Implements Phase 2 requirements:
- Dataset validation (schema, label integrity)
- Corruption detection (unreadable/truncated images)
- Duplicate image detection via perceptual hash
- Missing label detection
- Automatic dataset reports in Markdown and JSON
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import TARGETS
from ..utils.logging import get_logger

logger = get_logger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class ValidationIssue:
    """A single data quality issue."""

    severity: str  # "error" | "warning" | "info"
    category: str
    message: str
    affected_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "affected_count": len(self.affected_ids),
            "affected_ids_sample": self.affected_ids[:5],
        }


@dataclass
class DatasetReport:
    """Full dataset validation report."""

    dataset_name: str
    generated_at: str
    num_studies: int
    num_series: int
    num_targets: int
    target_prevalence: dict[str, float]
    issues: list[ValidationIssue]
    is_valid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "generated_at": self.generated_at,
            "statistics": {
                "num_studies": self.num_studies,
                "num_series": self.num_series,
                "num_targets": self.num_targets,
            },
            "target_prevalence": self.target_prevalence,
            "is_valid": self.is_valid,
            "issues": [i.as_dict() for i in self.issues],
        }

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    def to_markdown(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            f"# Dataset Report: {self.dataset_name}",
            f"\n**Generated:** {self.generated_at}",
            f"**Valid:** {'✅ Yes' if self.is_valid else '❌ No'}",
            "\n## Statistics\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Studies | {self.num_studies:,} |",
            f"| Series | {self.num_series:,} |",
            f"| Targets | {self.num_targets} |",
            "\n## Target Prevalence\n",
            "| Target | Prevalence |",
            "|--------|-----------|",
        ]
        for target, prev in self.target_prevalence.items():
            lines.append(f"| {target} | {prev:.1%} |")
        lines.append("\n## Issues\n")
        if not self.issues:
            lines.append("_No issues found._")
        else:
            errors = [i for i in self.issues if i.severity == "error"]
            warnings = [i for i in self.issues if i.severity == "warning"]
            for issue in errors + warnings:
                icon = "🔴" if issue.severity == "error" else "🟡"
                lines.append(f"### {icon} [{issue.severity.upper()}] {issue.category}")
                lines.append(f"\n{issue.message}")
                if issue.affected_ids:
                    lines.append(
                        f"\n_Affected ({len(issue.affected_ids)} items):_ "
                        + ", ".join(issue.affected_ids[:5])
                        + ("..." if len(issue.affected_ids) > 5 else "")
                    )
                lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


# ── Core validators ───────────────────────────────────────────────────────────


class DatasetValidator:
    """Validates the RSNA Knee dataset against production-grade quality rules."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._issues: list[ValidationIssue] = []

    def _add_issue(
        self,
        severity: str,
        category: str,
        message: str,
        affected_ids: list[str] | None = None,
    ) -> None:
        self._issues.append(
            ValidationIssue(
                severity=severity,
                category=category,
                message=message,
                affected_ids=affected_ids or [],
            )
        )

    def validate_metadata(self, df: pd.DataFrame, split: str = "train") -> None:
        """Validate CSV metadata schema and label integrity."""
        logger.info("Validating metadata", split=split, rows=len(df))

        # Check required ID column
        if "StudyInstanceUID" not in df.columns:
            self._add_issue("error", "schema", "Missing required column: StudyInstanceUID")
        else:
            # Duplicate IDs
            dups = df[df["StudyInstanceUID"].duplicated(keep=False)]
            if not dups.empty:
                self._add_issue(
                    "error",
                    "duplicates",
                    f"Found {len(dups)} rows with duplicate StudyInstanceUID",
                    dups["StudyInstanceUID"].astype(str).unique().tolist(),
                )

            # Missing IDs
            missing = df["StudyInstanceUID"].isna()
            if missing.any():
                self._add_issue(
                    "error",
                    "missing_values",
                    f"Found {missing.sum()} rows with null StudyInstanceUID",
                )

        # Check label columns (training only)
        if split == "train":
            for target in TARGETS:
                if target not in df.columns:
                    self._add_issue(
                        "error",
                        "schema",
                        f"Missing required label column: '{target}'",
                    )
                    continue

                col = df[target]
                invalid_labels = col[~col.isin([0, 1, 0.0, 1.0]) & col.notna()]
                if not invalid_labels.empty:
                    self._add_issue(
                        "warning",
                        "label_integrity",
                        f"Target '{target}' has {len(invalid_labels)} values outside {{0, 1}}",
                    )

                null_count = col.isna().sum()
                if null_count > 0:
                    self._add_issue(
                        "warning",
                        "missing_labels",
                        f"Target '{target}' has {null_count} missing values",
                    )

    def compute_target_prevalence(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute positive rate per target label."""
        prevalence: dict[str, float] = {}
        for target in TARGETS:
            if target in df.columns:
                values = pd.to_numeric(df[target], errors="coerce")
                prevalence[target] = float(values.mean()) if values.notna().any() else float("nan")
        return prevalence

    def detect_image_corruption(
        self,
        image_dir: Path,
        *,
        sample_size: int | None = None,
    ) -> list[str]:
        """Detect corrupted or unreadable image files. Returns list of bad paths."""
        corrupted: list[str] = []
        all_dcm = list(image_dir.rglob("*.dcm"))
        if sample_size is not None:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_dcm), size=min(sample_size, len(all_dcm)), replace=False)
            all_dcm = [all_dcm[i] for i in indices]

        logger.info("Checking image files for corruption", count=len(all_dcm))
        for dcm_path in all_dcm:
            try:
                import pydicom  # noqa: PLC0415

                ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=False)
                _ = ds.pixel_array  # Force pixel decode
            except Exception as exc:  # noqa: BLE001
                logger.warning("Corrupted DICOM", path=str(dcm_path), error=str(exc))
                corrupted.append(str(dcm_path))

        if corrupted:
            self._add_issue(
                "error",
                "corruption",
                f"Found {len(corrupted)} corrupted DICOM files",
                corrupted,
            )
        return corrupted

    def detect_duplicate_images(
        self,
        image_dir: Path,
        *,
        hash_size: int = 8,
    ) -> dict[str, list[str]]:
        """Detect visually duplicate images via perceptual hash (dHash).

        Returns a mapping from hash → list[file_path] for duplicates only.
        """
        try:
            import imagehash  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            logger.warning("imagehash/Pillow not installed; skipping duplicate detection")
            return {}

        hash_map: dict[str, list[str]] = {}
        for img_path in image_dir.rglob("*.png"):
            try:
                img = Image.open(img_path).convert("L")
                h = str(imagehash.dhash(img, hash_size=hash_size))
                hash_map.setdefault(h, []).append(str(img_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not hash image", path=str(img_path), error=str(exc))

        duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
        if duplicates:
            total = sum(len(v) - 1 for v in duplicates.values())
            self._add_issue(
                "warning",
                "duplicates",
                f"Found {total} duplicate images across {len(duplicates)} hash groups",
            )
        return duplicates

    def generate_report(
        self,
        df: pd.DataFrame,
        *,
        dataset_name: str = "rsna-knee",
        split: str = "train",
        image_dir: Path | None = None,
    ) -> DatasetReport:
        """Run all validators and produce a unified DatasetReport."""
        self._issues = []
        self.validate_metadata(df, split=split)
        prevalence = self.compute_target_prevalence(df) if split == "train" else {}

        if image_dir is not None and image_dir.exists():
            self.detect_image_corruption(image_dir)

        is_valid = not any(i.severity == "error" for i in self._issues)
        report = DatasetReport(
            dataset_name=dataset_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            num_studies=df["StudyInstanceUID"].nunique() if "StudyInstanceUID" in df.columns else 0,
            num_series=len(df),
            num_targets=len(TARGETS),
            target_prevalence=prevalence,
            issues=list(self._issues),
            is_valid=is_valid,
        )
        logger.info(
            "Dataset validation complete",
            is_valid=is_valid,
            errors=sum(1 for i in self._issues if i.severity == "error"),
            warnings=sum(1 for i in self._issues if i.severity == "warning"),
        )
        return report


def _checksum_file(path: Path) -> str:
    """Compute SHA-256 of a file for integrity tracking."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_manifest(data_dir: Path) -> dict[str, Any]:
    """Build a reproducibility manifest with checksums for all CSV files."""
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "files": {},
    }
    for csv_path in sorted(data_dir.rglob("*.csv")):
        rel = csv_path.relative_to(data_dir)
        manifest["files"][str(rel)] = {
            "sha256": _checksum_file(csv_path),
            "size_bytes": csv_path.stat().st_size,
        }
    return manifest


def main() -> None:
    """CLI entry-point: rsna-validate-data."""
    import argparse  # noqa: PLC0415

    from ..utils.logging import configure_logging  # noqa: PLC0415

    configure_logging()
    parser = argparse.ArgumentParser(description="Validate RSNA Knee dataset")
    parser.add_argument("--csv", required=True, help="Path to train.csv or test.csv")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output-dir", default="results/reports")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    validator = DatasetValidator(Path(args.csv).parent)
    report = validator.generate_report(df, split=args.split)

    out = Path(args.output_dir)
    report.to_json(out / "dataset_report.json")
    report.to_markdown(out / "dataset_report.md")
    logger.info("Reports saved", output_dir=str(out))
