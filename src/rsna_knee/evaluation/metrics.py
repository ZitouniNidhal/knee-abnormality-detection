"""Comprehensive medical AI evaluation metrics.

Implements Phase 4 requirements:
- Per-target and macro-averaged: Accuracy, Precision, Recall, Specificity, F1
- ROC-AUC, PR-AUC, Cohen's Kappa, MCC
- Confusion matrix, ROC and PR curve generation
- Clinical report generation in Markdown
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..constants import TARGETS
from ..utils.logging import get_logger

logger = get_logger(__name__)


# ── Per-target metrics ────────────────────────────────────────────────────────


@dataclass
class TargetMetrics:
    """All evaluation metrics for a single binary target."""

    target: str
    accuracy: float
    precision: float
    recall: float  # sensitivity
    specificity: float
    f1: float
    roc_auc: float
    pr_auc: float
    cohen_kappa: float
    mcc: float
    support: int
    positive_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall_sensitivity": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "mcc": round(self.mcc, 4),
            "support": self.support,
            "positive_rate": round(self.positive_rate, 4),
        }


@dataclass
class EvaluationResults:
    """Aggregated evaluation results across all targets."""

    per_target: list[TargetMetrics]
    macro_roc_auc: float
    macro_f1: float
    macro_pr_auc: float
    fold: int | None = None
    model_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "fold": self.fold,
            "macro": {
                "roc_auc": round(self.macro_roc_auc, 4),
                "f1": round(self.macro_f1, 4),
                "pr_auc": round(self.macro_pr_auc, 4),
            },
            "per_target": [t.as_dict() for t in self.per_target],
            **self.extra,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([t.as_dict() for t in self.per_target])

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path


# ── Core evaluator ────────────────────────────────────────────────────────────


class MedicalMetricsEvaluator:
    """Evaluates multilabel predictions with full medical AI metric suite."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def compute_target_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        target_idx: int,
        target_name: str,
    ) -> TargetMetrics:
        """Compute all metrics for one binary target."""
        truth = y_true[:, target_idx].astype(int)
        scores = y_scores[:, target_idx]
        preds = (scores >= self.threshold).astype(int)

        n_classes = len(np.unique(truth))

        if n_classes < 2:
            logger.warning(
                "Single class in target; setting AUC metrics to NaN",
                target=target_name,
            )
            return TargetMetrics(
                target=target_name,
                accuracy=float(accuracy_score(truth, preds)),
                precision=float("nan"),
                recall=float("nan"),
                specificity=float("nan"),
                f1=float("nan"),
                roc_auc=float("nan"),
                pr_auc=float("nan"),
                cohen_kappa=float("nan"),
                mcc=float("nan"),
                support=int(truth.sum()),
                positive_rate=float(truth.mean()),
            )

        tn, fp, fn, tp = confusion_matrix(truth, preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return TargetMetrics(
            target=target_name,
            accuracy=float(accuracy_score(truth, preds)),
            precision=float(precision_score(truth, preds, zero_division=0)),
            recall=float(recall_score(truth, preds, zero_division=0)),
            specificity=specificity,
            f1=float(f1_score(truth, preds, zero_division=0)),
            roc_auc=float(roc_auc_score(truth, scores)),
            pr_auc=float(average_precision_score(truth, scores)),
            cohen_kappa=float(cohen_kappa_score(truth, preds)),
            mcc=float(matthews_corrcoef(truth, preds)),
            support=int(truth.sum()),
            positive_rate=float(truth.mean()),
        )

    def evaluate(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        *,
        model_name: str = "",
        fold: int | None = None,
    ) -> EvaluationResults:
        """Evaluate all 12 targets and compute macro aggregates."""
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        per_target = [
            self.compute_target_metrics(y_true, y_scores, idx, target)
            for idx, target in enumerate(TARGETS)
        ]

        valid_aucs = [t.roc_auc for t in per_target if not np.isnan(t.roc_auc)]
        valid_f1s = [t.f1 for t in per_target if not np.isnan(t.f1)]
        valid_prauc = [t.pr_auc for t in per_target if not np.isnan(t.pr_auc)]

        return EvaluationResults(
            per_target=per_target,
            macro_roc_auc=float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
            macro_f1=float(np.mean(valid_f1s)) if valid_f1s else float("nan"),
            macro_pr_auc=float(np.mean(valid_prauc)) if valid_prauc else float("nan"),
            model_name=model_name,
            fold=fold,
        )


# ── Plot generation ───────────────────────────────────────────────────────────


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    threshold: float = 0.5,
    output_dir: str | Path = "results/figures",
    model_name: str = "model",
) -> Path:
    """Generate per-target confusion matrix grid and save as PNG."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed; skipping confusion matrix plot")
        return Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cols = 4
    n_rows = (len(TARGETS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    preds = (y_scores >= threshold).astype(int)

    for idx, (target, ax) in enumerate(zip(TARGETS, axes)):
        cm = confusion_matrix(y_true[:, idx], preds[:, idx], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
        )
        ax.set_title(target, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[len(TARGETS) :]:
        ax.set_visible(False)

    fig.suptitle(f"Confusion Matrices — {model_name}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = output_dir / f"{model_name}_confusion_matrices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrices saved", path=str(out_path))
    return out_path


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    output_dir: str | Path = "results/figures",
    model_name: str = "model",
) -> Path:
    """Generate overlaid ROC curves for all targets."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping ROC plot")
        return Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20.colors  # type: ignore[attr-defined]

    for idx, target in enumerate(TARGETS):
        truth = y_true[:, idx]
        if len(np.unique(truth)) < 2:
            continue
        fpr, tpr, _ = roc_curve(truth, y_scores[:, idx])
        auc = roc_auc_score(truth, y_scores[:, idx])
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], label=f"{target} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = output_dir / f"{model_name}_roc_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved", path=str(out_path))
    return out_path


def plot_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    output_dir: str | Path = "results/figures",
    model_name: str = "model",
) -> Path:
    """Generate Precision-Recall curves for all targets."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping PR plot")
        return Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20.colors  # type: ignore[attr-defined]

    for idx, target in enumerate(TARGETS):
        truth = y_true[:, idx]
        if len(np.unique(truth)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(truth, y_scores[:, idx])
        ap = average_precision_score(truth, y_scores[:, idx])
        ax.plot(rec, prec, color=colors[idx % len(colors)], label=f"{target} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {model_name}", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = output_dir / f"{model_name}_pr_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curves saved", path=str(out_path))
    return out_path


# ── Clinical report ───────────────────────────────────────────────────────────


def generate_clinical_report(
    results: EvaluationResults,
    output_path: str | Path = "results/reports/clinical_report.md",
) -> Path:
    """Generate a clinical-style evaluation report in Markdown."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Clinical Evaluation Report",
        f"\n**Model:** {results.model_name or 'N/A'}",
        f"**Fold:** {results.fold if results.fold is not None else 'Full dataset'}",
        "\n---\n",
        "## Executive Summary\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Macro ROC-AUC | **{results.macro_roc_auc:.4f}** |",
        f"| Macro F1 | **{results.macro_f1:.4f}** |",
        f"| Macro PR-AUC | **{results.macro_pr_auc:.4f}** |",
        "\n## Per-Target Results\n",
        "| Target | AUC | F1 | Recall | Precision | Specificity | MCC |",
        "|--------|-----|-----|--------|-----------|-------------|-----|",
    ]

    for t in results.per_target:
        auc = f"{t.roc_auc:.4f}" if not np.isnan(t.roc_auc) else "N/A"
        f1 = f"{t.f1:.4f}" if not np.isnan(t.f1) else "N/A"
        rec = f"{t.recall:.4f}" if not np.isnan(t.recall) else "N/A"
        prec = f"{t.precision:.4f}" if not np.isnan(t.precision) else "N/A"
        spec = f"{t.specificity:.4f}" if not np.isnan(t.specificity) else "N/A"
        mcc = f"{t.mcc:.4f}" if not np.isnan(t.mcc) else "N/A"
        lines.append(f"| {t.target} | {auc} | {f1} | {rec} | {prec} | {spec} | {mcc} |")

    lines.extend(
        [
            "\n---",
            "\n> **Note:** Metrics computed at threshold=0.5. AUC is threshold-independent.",
            "\n> This report is for research purposes only and does not constitute clinical advice.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Clinical report saved", path=str(output_path))
    return output_path
