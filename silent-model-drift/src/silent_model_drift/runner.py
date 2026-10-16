from __future__ import annotations

from time import perf_counter

from .clients import CompletionClient
from .fingerprint import make_fingerprint
from .models import CaptureRun, Observation, RequestConfig, SentinelSuite, utc_now


def capture_run(
    suite: SentinelSuite,
    client: CompletionClient,
    request_config: RequestConfig,
    *,
    repeats: int,
    store_raw_responses: bool = False,
    fail_on_error: bool = False,
) -> CaptureRun:

    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    observations: list[Observation] = []
    for repeat_index in range(repeats):
        for sentinel in suite.sentinels:
            started = perf_counter()
            error_type: str | None = None
            try:
                output = client.complete(sentinel, request_config)
            except Exception as error:
                if fail_on_error:
                    raise
                error_type = type(error).__name__
                output = f"__capture_error__:{error_type}"
            latency_ms = (perf_counter() - started) * 1000
            observations.append(
                Observation(
                    sentinel_id=sentinel.id,
                    repeat_index=repeat_index,
                    collected_at=utc_now(),
                    fingerprint=make_fingerprint(output),
                    latency_ms=latency_ms,
                    raw_response=output if store_raw_responses else None,
                    error_type=error_type,
                )
            )
    return CaptureRun.create(
        suite=suite,
        request_config=request_config,
        repeats=repeats,
        observations=observations,
        store_raw_responses=store_raw_responses,
    )
