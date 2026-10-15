"""Conservative, deterministic pseudo-label extraction for *training* reports.

This module intentionally does not try to infer a label from an unmentioned
finding.  A neutral score (``0.5``) with low confidence is returned for
unmentioned, ambiguous, historical, or uncertain findings, so callers can
drop those rows or use ``confidence`` as a sample weight.

The extractor is designed for English radiology terminology.  Text is Unicode
normalised (including curly apostrophes such as ``Baker's``), case-insensitive,
and accepts common long-form and abbreviated English terms.  It is rule based
and has no model, random state, or external dependency.

Only reports explicitly identified as the training split may be processed.  Do
not use report text from a competition test set: that would be label leakage.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
import unicodedata
from typing import Final, Literal, TypedDict


LABELS: Final[tuple[str, ...]] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

Status = Literal["positive", "negative", "uncertain", "unresolved", "unmentioned"]


class LabelOutput(TypedDict):
    """A serialisable pseudo-label for one target.

    ``score`` is 1.0 for an asserted finding, 0.0 for an explicitly excluded
    finding, and 0.5 when the extractor deliberately declines to decide.
    ``mentioned`` distinguishes no mention from an ambiguous mention.
    ``confidence`` is a reliability weight rather than a calibrated clinical
    probability.  ``evidence`` contains short normalised matched snippets.
    """

    score: float
    mentioned: bool
    confidence: float
    status: Status
    evidence: tuple[str, ...]


class _Event(TypedDict):
    status: Literal["positive", "negative", "uncertain", "unresolved"]
    start: int
    end: int
    evidence: str


# The entity patterns are deliberately specific.  In particular, avoid broad
# words such as "fluid", "bruise", or "cyst" that are common but do not reliably
# encode the competition target.
_ALIASES: Final[dict[str, str]] = {
    "ACL": r"(?:\bacl\b|\banterior\s+cruciate\s+ligament\b)",
    "MCL": r"(?:\bmcl\b|\bmedial\s+collateral\s+ligament\b)",
    "Medial Meniscus": r"\bmedial\s+menisc(?:us|al)\b",
    "Lateral Meniscus": r"\blateral\s+menisc(?:us|al)\b",
    "Medial OA": (
        r"(?:\bmedial(?:\s+compartment)?\s+"
        r"(?:osteoarthritis|osteoarthrosis|arthrosis|oa|degenerative\s+joint\s+disease)\b|"
        r"\b(?:osteoarthritis|osteoarthrosis|arthrosis|oa)\s+of\s+(?:the\s+)?medial"
        r"(?:\s+compartment)?\b)"
    ),
    "Lateral OA": (
        r"(?:\blateral(?:\s+compartment)?\s+"
        r"(?:osteoarthritis|osteoarthrosis|arthrosis|oa|degenerative\s+joint\s+disease)\b|"
        r"\b(?:osteoarthritis|osteoarthrosis|arthrosis|oa)\s+of\s+(?:the\s+)?lateral"
        r"(?:\s+compartment)?\b)"
    ),
    "PF OA": (
        r"(?:\bpatello[ -]?femoral\s+"
        r"(?:osteoarthritis|osteoarthrosis|arthrosis|oa|degenerative\s+joint\s+disease)\b|"
        r"\bpf\s*(?:osteoarthritis|osteoarthrosis|arthrosis|oa)\b|"
        r"\b(?:osteoarthritis|osteoarthrosis|arthrosis|oa)\s+of\s+(?:the\s+)?"
        r"patello[ -]?femoral(?:\s+joint)?\b)"
    ),
    "Effusion": r"\b(?:knee|joint|suprapatellar)?\s*effusion\b",
    "Synovitis": r"\b(?:synovitis|synovial\s+(?:thickening|hypertrophy|proliferation))\b",
    "Baker's": r"\b(?:baker'?s\s+cyst|popliteal\s+cyst)\b",
    "Contusion": r"\b(?:bone|osseous)\s+contusion\b|\bbone\s+bruise\b",
    "Fracture": r"\b(?:fracture|fx|cortical\s+break)\b",
}

_STRUCTURAL_LABELS: Final[frozenset[str]] = frozenset(
    {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}
)

# These are abnormality words, not all diagnoses.  Their purpose is to decide
# whether an anatomical mention (e.g. "ACL") actually asserts pathology.
_STRUCTURAL_ABNORMALITY: Final[str] = (
    r"(?:tear(?:ing|s|ed)?|rupture|ruptured|sprain(?:ed)?|injur(?:y|ies|ed)|"
    r"deficien(?:cy|t)|laxity|lax|discontinu(?:ity|ous)|attenuat(?:ion|ed)|"
    r"degenerat(?:ion|ive|ed)|extrusion|extruded|macerat(?:ion|ed)|"
    r"truncat(?:ion|ed)|fray(?:ing|ed)?|abnormal(?:ity)?)"
)

_STRUCTURAL_NORMAL: Final[str] = (
    r"(?:intact|normal|unremarkable|preserved|continuous|without\s+(?:tear|injury)|"
    r"no\s+(?:tear|injury|discontinuity))"
)

_NEGATION_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"\b(?:no|without|absent|absence\s+of|negative\s+for|free\s+of|"
    r"lack(?:ing)?\s+of|lacks|neither)\b(?:\s+[\w/-]+){0,6}\s*$",
    re.IGNORECASE,
)
_HEDGED_NEGATION: Final[re.Pattern[str]] = re.compile(
    r"\b(?:no|without)\s+(?:definite|clear|obvious|convincing)\b",
    re.IGNORECASE,
)
_NEGATION_SUFFIX: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:(?:is|are|was|were|has\s+been)\s+)?"
    r"(?:not\s+(?:seen|identified|visualized|demonstrated|appreciated|present)|"
    r"absent|resolved)\b",
    re.IGNORECASE,
)
_UNCERTAINTY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:possible|possibly|probable|probably|likely|may|might|could|"
    r"cannot\s+(?:exclude|rule\s+out)|can'?t\s+(?:exclude|rule\s+out)|"
    r"suspicious\s+for|suspect(?:ed)?|question(?:able)?|equivocal|"
    r"suggestive\s+of|concerning\s+for|consistent\s+with|versus|vs\.?)\b",
    re.IGNORECASE,
)
_HISTORY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:history\s+of|prior|previous|remote|old|healed|chronic|"
    r"status\s+post|s/p|post[ -]?operative)\b",
    re.IGNORECASE,
)


def _normalise_report(report: str) -> str:
    """Normalise Unicode and punctuation without translating report content."""

    normalised = unicodedata.normalize("NFKC", report).casefold()
    # NFKC does not fold every apostrophe-like glyph used by dictation systems.
    normalised = normalised.translate(
        str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "‐": "-", "–": "-", "—": "-"})
    )
    return re.sub(r"\s+", " ", normalised).strip()


def _sentences(text: str) -> Iterable[str]:
    """Split just enough to keep negation from leaking across findings."""

    for sentence in re.split(r"(?:[.!?]+|;|\n)+", text):
        sentence = sentence.strip()
        if sentence:
            yield sentence


def _context(sentence: str, start: int, end: int) -> tuple[str, str]:
    # A bounded window avoids a negated finding at one end of a long sentence
    # cancelling an unrelated finding at the other end.
    return sentence[max(0, start - 84) : start], sentence[end : end + 64]


def _assertion_status(sentence: str, start: int, end: int) -> Literal["positive", "negative", "uncertain"]:
    """Classify a matched affirmative phrase using local report context."""

    left, right = _context(sentence, start, end)
    negated = bool(_NEGATION_PREFIX.search(left) or _NEGATION_SUFFIX.search(right))

    # "No definite fracture" is not a reliable negative pseudo-label; it is a
    # hedged statement that may precede an occult-fracture recommendation.
    if negated:
        if _HEDGED_NEGATION.search(left):
            return "uncertain"
        return "negative"

    nearby = f"{left} {sentence[start:end]} {right}"
    if _UNCERTAINTY.search(nearby) or _HISTORY.search(nearby):
        return "uncertain"
    return "positive"


def _structural_patterns(alias: str) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    """Build bounded anatomy/pathology and anatomy/normal patterns once per use."""

    # Five intervening words covers ordinary dictated wording such as
    # "ACL demonstrates a full thickness tear" while limiting cross-finding
    # associations in compound sentences.
    gap = r"(?:\s+(?:[\w/-]+)){0,5}\s+"
    positive = re.compile(
        rf"(?:{alias}){gap}{_STRUCTURAL_ABNORMALITY}|"
        rf"{_STRUCTURAL_ABNORMALITY}{gap}(?:{alias})",
        re.IGNORECASE,
    )
    negative = re.compile(
        rf"(?:{alias}){gap}{_STRUCTURAL_NORMAL}|"
        rf"{_STRUCTURAL_NORMAL}{gap}(?:{alias})",
        re.IGNORECASE,
    )
    mention = re.compile(alias, re.IGNORECASE)
    return positive, negative, mention


def _direct_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(alias, re.IGNORECASE)


def _add_event(
    events: list[_Event],
    status: Literal["positive", "negative", "uncertain", "unresolved"],
    start: int,
    end: int,
    evidence: str,
) -> None:
    events.append({"status": status, "start": start, "end": end, "evidence": evidence})


def _events_for_structural_label(sentence: str, alias: str) -> list[_Event]:
    positive_pattern, negative_pattern, mention_pattern = _structural_patterns(alias)
    events: list[_Event] = []
    covered: list[tuple[int, int]] = []

    for match in positive_pattern.finditer(sentence):
        status = _assertion_status(sentence, match.start(), match.end())
        _add_event(events, status, match.start(), match.end(), match.group().strip())
        covered.append((match.start(), match.end()))

    for match in negative_pattern.finditer(sentence):
        # A normality statement is an explicit negative unless it is itself
        # hedged (for example, "ACL may be intact").
        left, right = _context(sentence, match.start(), match.end())
        nearby = f"{left} {match.group()} {right}"
        status: Literal["negative", "uncertain"] = (
            "uncertain" if _UNCERTAINTY.search(nearby) else "negative"
        )
        _add_event(events, status, match.start(), match.end(), match.group().strip())
        covered.append((match.start(), match.end()))

    for match in mention_pattern.finditer(sentence):
        if not any(start <= match.start() and match.end() <= end for start, end in covered):
            # A bare anatomical mention is useful provenance but not evidence of
            # disease.  Keep it neutral instead of fabricating a negative label.
            _add_event(events, "unresolved", match.start(), match.end(), match.group().strip())

    return events


def _events_for_direct_label(sentence: str, alias: str) -> list[_Event]:
    pattern = _direct_pattern(alias)
    events: list[_Event] = []
    for match in pattern.finditer(sentence):
        status = _assertion_status(sentence, match.start(), match.end())
        _add_event(events, status, match.start(), match.end(), match.group().strip())
    return events


def _deduplicate(events: list[_Event]) -> list[_Event]:
    seen: set[tuple[str, int, int, str]] = set()
    unique: list[_Event] = []
    for event in sorted(events, key=lambda item: (item["start"], item["end"], item["status"], item["evidence"])):
        key = (event["status"], event["start"], event["end"], event["evidence"])
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return unique


def _summarise(events: list[_Event]) -> LabelOutput:
    events = _deduplicate(events)
    if not events:
        return {
            "score": 0.5,
            "mentioned": False,
            "confidence": 0.0,
            "status": "unmentioned",
            "evidence": (),
        }

    statuses = {event["status"] for event in events}
    evidence = tuple(dict.fromkeys(event["evidence"] for event in events))

    # A report with contradictory assertions or any diagnostic hedge should not
    # be converted into a hard pseudo-label.  This intentionally trades recall
    # for precision, which is generally the safer training-time choice.
    if "uncertain" in statuses or ({"positive", "negative"} <= statuses):
        return {
            "score": 0.5,
            "mentioned": True,
            "confidence": 0.2,
            "status": "uncertain",
            "evidence": evidence,
        }
    if "positive" in statuses:
        return {
            "score": 1.0,
            "mentioned": True,
            "confidence": 0.95,
            "status": "positive",
            "evidence": evidence,
        }
    if "negative" in statuses:
        return {
            "score": 0.0,
            "mentioned": True,
            "confidence": 0.9,
            "status": "negative",
            "evidence": evidence,
        }
    return {
        "score": 0.5,
        "mentioned": True,
        "confidence": 0.05,
        "status": "unresolved",
        "evidence": evidence,
    }


def extract_report_labels(report: str | None, *, split: str) -> dict[str, LabelOutput]:
    """Extract conservative pseudo-labels from one **training** report.

    Parameters
    ----------
    report:
        Free-text report from the training set.  ``None`` and blank strings are
        supported and yield neutral, unmentioned outputs.
    split:
        Must be exactly ``"train"`` (case-insensitive surrounding whitespace is
        accepted).  Passing a test, validation, or unknown split raises
        ``ValueError`` to make test-report leakage explicit.

    Returns
    -------
    dict
        An insertion-ordered mapping in ``LABELS`` order.  Each value includes
        ``score``, ``mentioned``, ``confidence``, ``status``, and ``evidence``.
    """

    if not isinstance(split, str) or split.strip().casefold() != "train":
        raise ValueError("report pseudo-labels may only be extracted with split='train'")
    if report is None:
        text = ""
    elif isinstance(report, str):
        text = _normalise_report(report)
    else:
        raise TypeError("report must be a string or None")

    all_events: dict[str, list[_Event]] = {label: [] for label in LABELS}
    for sentence in _sentences(text):
        for label in LABELS:
            if label in _STRUCTURAL_LABELS:
                all_events[label].extend(_events_for_structural_label(sentence, _ALIASES[label]))
            else:
                all_events[label].extend(_events_for_direct_label(sentence, _ALIASES[label]))

    return {label: _summarise(all_events[label]) for label in LABELS}


def extract_train_report_labels(report: str | None) -> dict[str, LabelOutput]:
    """Convenience wrapper that makes the training-only constraint explicit."""

    return extract_report_labels(report, split="train")
