"""Competition-wide constants and schema checks."""

from __future__ import annotations

TARGETS: tuple[str, ...] = (
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

ID_COLUMN = "StudyInstanceUID"
N_TARGETS = len(TARGETS)

# A stable mapping is saved into checkpoints. Never infer this from a pandas
# column order, which can silently differ between training and inference.
TARGET_TO_INDEX = {name: index for index, name in enumerate(TARGETS)}

REQUIRED_SUBMISSION_COLUMNS = (ID_COLUMN, *TARGETS)
