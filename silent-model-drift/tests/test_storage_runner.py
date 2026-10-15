from __future__ import annotations

from dataclasses import dataclass

from silent_model_drift.models import RequestConfig, Sentinel, SentinelSuite
from silent_model_drift.runner import capture_run
from silent_model_drift.storage import load_capture, save_capture


@dataclass
class StubClient:
    """A small deterministic client used to exercise capture plumbing."""

    response: str = "fixed output"

    def complete(self, sentinel: Sentinel, request_config: RequestConfig) -> str:
        del sentinel, request_config
        return self.response


def test_capture_round_trip_omits_raw_output_by_default(tmp_path) -> None:
    suite = SentinelSuite(name="suite", sentinels=(Sentinel(id="one", prompt="one"),))
    run = capture_run(
        suite,
        StubClient(),
        RequestConfig(provider="replay", model="stub"),
        repeats=3,
    )
    path = tmp_path / "capture.json"

    save_capture(run, path)
    loaded = load_capture(path)

    assert len(loaded.observations) == 3
    assert all(item.raw_response is None for item in loaded.observations)
    assert loaded.manifest.suite_digest == suite.digest


def test_capture_records_error_as_an_observation() -> None:
    class FailingClient:
        def complete(self, sentinel: Sentinel, request_config: RequestConfig) -> str:
            del sentinel, request_config
            raise RuntimeError("network unavailable")

    suite = SentinelSuite(name="suite", sentinels=(Sentinel(id="one", prompt="one"),))
    run = capture_run(
        suite,
        FailingClient(),
        RequestConfig(provider="replay", model="stub"),
        repeats=1,
    )

    assert run.observations[0].error_type == "RuntimeError"
