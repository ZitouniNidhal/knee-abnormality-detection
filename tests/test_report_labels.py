"""Tests for the conservative training-report pseudo-label extractor."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.report_labels import LABELS, extract_report_labels, extract_train_report_labels


def test_returns_all_known_labels_in_stable_order() -> None:
    result = extract_report_labels("No acute abnormality.", split="train")

    assert tuple(result) == LABELS
    assert all(value["score"] == 0.5 for value in result.values())
    assert all(value["mentioned"] is False for value in result.values())
    assert all(value["confidence"] == 0.0 for value in result.values())


def test_common_english_terms_produce_clear_positive_labels() -> None:
    report = """
        Complete rupture of the anterior cruciate ligament and a grade 2 MCL sprain.
        Complex tear of the posterior horn of the medial meniscus. Lateral meniscal
        extrusion is present. Medial compartment osteoarthritis and patellofemoral OA.
        Moderate joint effusion with synovial thickening. A Baker’s cyst is present.
        Bone contusion of the lateral femoral condyle. Nondisplaced fracture of the tibia.
    """

    result = extract_train_report_labels(report)

    for label in (
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ):
        assert result[label]["status"] == "positive", label
        assert result[label]["score"] == 1.0, label
        assert result[label]["mentioned"] is True, label
        assert result[label]["confidence"] >= 0.9, label

    assert result["Lateral OA"]["status"] == "unmentioned"


def test_negated_and_normal_findings_do_not_become_positive() -> None:
    report = """
        No evidence of ACL tear. The MCL is intact. No medial meniscal tear or
        lateral meniscal injury. No medial compartment osteoarthritis. No knee
        effusion, synovitis, Baker's cyst, bone contusion, or acute fracture.
    """

    result = extract_report_labels(report, split=" TRAIN ")

    for label in (
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ):
        assert result[label]["status"] == "negative", label
        assert result[label]["score"] == 0.0, label
        assert result[label]["confidence"] >= 0.8, label


def test_uncertain_or_hedged_findings_stay_neutral() -> None:
    report = (
        "Possible partial ACL tear. Cannot exclude occult fracture. "
        "No definite joint effusion. There is a history of a Baker's cyst."
    )

    result = extract_report_labels(report, split="train")

    for label in ("ACL", "Fracture", "Effusion", "Baker's"):
        assert result[label]["status"] == "uncertain", label
        assert result[label]["score"] == 0.5, label
        assert result[label]["mentioned"] is True, label
        assert result[label]["confidence"] <= 0.2, label


def test_bare_anatomy_mention_is_not_a_disease_pseudolabel() -> None:
    result = extract_report_labels("The ACL and medial meniscus are visualized.", split="train")

    assert result["ACL"]["status"] == "unresolved"
    assert result["ACL"]["mentioned"] is True
    assert result["ACL"]["score"] == 0.5
    assert result["Medial Meniscus"]["status"] == "unresolved"


def test_output_is_deterministic() -> None:
    report = "Mild joint effusion. No ACL tear. Possible bone contusion."

    assert extract_report_labels(report, split="train") == extract_report_labels(report, split="train")


def test_rejects_non_training_splits() -> None:
    with pytest.raises(ValueError, match="split='train'"):
        extract_report_labels("ACL tear", split="test")


def test_rejects_non_string_reports() -> None:
    with pytest.raises(TypeError, match="string or None"):
        extract_report_labels(123, split="train")  # type: ignore[arg-type]
