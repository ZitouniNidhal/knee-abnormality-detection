from __future__ import annotations

import json

from silent_model_drift.cli import main


def _suite(path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "cli-suite",
                "sentinels": [{"id": "one", "prompt": "Say one."}],
            }
        ),
        encoding="utf-8",
    )


def test_cli_can_capture_and_compare_replay_artifacts(tmp_path, capsys) -> None:
    suite = tmp_path / "suite.json"
    baseline_responses = tmp_path / "baseline-responses.json"
    candidate_responses = tmp_path / "candidate-responses.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    report = tmp_path / "report.json"
    _suite(suite)
    baseline_responses.write_text('{"one": "blue"}', encoding="utf-8")
    candidate_responses.write_text('{"one": "blue"}', encoding="utf-8")

    assert (
        main(
            [
                "capture",
                "--suite",
                str(suite),
                "--output",
                str(baseline),
                "--provider",
                "replay",
                "--model",
                "offline",
                "--replay-responses",
                str(baseline_responses),
                "--repeats",
                "3",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "capture",
                "--suite",
                str(suite),
                "--output",
                str(candidate),
                "--provider",
                "replay",
                "--model",
                "offline",
                "--replay-responses",
                str(candidate_responses),
                "--repeats",
                "3",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--output",
                str(report),
                "--permutations",
                "99",
            ]
        )
        == 0
    )

    assert report.exists()
    assert "no_statistically_supported_change" in capsys.readouterr().out


def test_cli_returns_alert_exit_code_for_detected_change(tmp_path) -> None:
    suite = tmp_path / "suite.json"
    baseline_responses = tmp_path / "baseline-responses.json"
    candidate_responses = tmp_path / "candidate-responses.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _suite(suite)
    baseline_responses.write_text('{"one": "blue"}', encoding="utf-8")
    candidate_responses.write_text('{"one": "green"}', encoding="utf-8")

    for responses, output in ((baseline_responses, baseline), (candidate_responses, candidate)):
        assert (
            main(
                [
                    "capture",
                    "--suite",
                    str(suite),
                    "--output",
                    str(output),
                    "--provider",
                    "replay",
                    "--model",
                    "offline",
                    "--replay-responses",
                    str(responses),
                    "--repeats",
                    "8",
                ]
            )
            == 0
        )

    assert (
        main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--permutations",
                "199",
            ]
        )
        == 2
    )
