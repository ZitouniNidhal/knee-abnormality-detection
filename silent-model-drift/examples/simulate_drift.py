"""Run a deterministic local example without an API key."""

from silent_model_drift.fingerprint import make_fingerprint
from silent_model_drift.statistics import DriftConfig, analyze_windows


def main() -> None:
    baseline = {
        "format": [make_fingerprint('{"answer":"blue"}') for _ in range(8)],
        "style": [make_fingerprint("Concise and factual.") for _ in range(8)],
    }
    candidate = {
        "format": [make_fingerprint("The answer is blue.") for _ in range(8)],
        "style": [make_fingerprint("Concise and factual.") for _ in range(8)],
    }
    report = analyze_windows(
        baseline,
        candidate,
        config=DriftConfig(permutations=999, seed=17),
    )
    print(report.to_json(indent=2))


if __name__ == "__main__":
    main()
