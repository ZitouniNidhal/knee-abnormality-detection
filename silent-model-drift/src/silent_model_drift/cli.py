"""Command-line interface for capture, comparison, and study planning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .clients import ProviderError, build_client
from .models import RequestConfig
from .power import plan_sentinels, samples_per_window_for_proportions
from .runner import capture_run
from .statistics import DriftConfig, analyze_runs
from .storage import load_capture, save_capture, save_report
from .suite import load_suite

_DEFAULT_KEY_ENVIRONMENT = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_CHANGE_CONCLUSIONS = {
    "behavioral_drift_detected",
    "behavioral_distribution_shift_detected",
    "nondeterminism_increase_detected",
    "localized_change_detected",
    "localized_nondeterminism_increase_detected",
}


def _load_json_object(path: str | None) -> dict[str, Any]:
    """Load a JSON object used for non-secret provider request extras."""

    if path is None:
        return {}
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Extras file does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Extras file is not valid JSON: {source}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError("Extras file root must be a JSON object.")
    return value


def _emit(payload: dict[str, Any]) -> None:
    """Print machine-readable JSON to standard output."""

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _capture(args: argparse.Namespace) -> int:
    """Execute the capture subcommand."""

    suite = load_suite(args.suite)
    extras = _load_json_object(args.extras)
    request_config = RequestConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        endpoint=args.endpoint,
        extras=extras,
    )
    key_environment = args.api_key_env or _DEFAULT_KEY_ENVIRONMENT.get(args.provider.casefold())
    api_key = os.environ.get(key_environment) if key_environment else None
    client = build_client(
        args.provider,
        api_key=api_key,
        endpoint=args.endpoint,
        timeout=args.timeout,
        replay_responses=args.replay_responses,
    )
    run = capture_run(
        suite,
        client,
        request_config,
        repeats=args.repeats,
        store_raw_responses=args.store_raw,
        fail_on_error=args.fail_on_error,
    )
    save_capture(run, args.output)
    errors = sum(observation.error_type is not None for observation in run.observations)
    _emit(
        {
            "artifact": str(Path(args.output)),
            "errors_recorded": errors,
            "observations": len(run.observations),
            "request_config_digest": run.manifest.request_config_digest,
            "run_id": run.manifest.run_id,
            "suite_digest": run.manifest.suite_digest,
        }
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    """Execute the comparison subcommand."""

    baseline = load_capture(args.baseline)
    candidate = load_capture(args.candidate)
    config = DriftConfig(
        alpha=args.alpha,
        permutations=args.permutations,
        seed=args.seed,
        minimum_effect=args.minimum_effect,
        minimum_center_shift=args.minimum_center_shift,
        minimum_dispersion_increase=args.minimum_dispersion_increase,
        minimum_samples_per_window=args.minimum_samples,
    )
    report = analyze_runs(
        baseline,
        candidate,
        config=config,
        require_matching_manifests=not args.allow_manifest_mismatch,
    )
    if args.output:
        save_report(report, args.output)
    _emit(report.to_dict())
    return 2 if report.conclusion in _CHANGE_CONCLUSIONS else 0


def _plan(args: argparse.Namespace) -> int:
    """Execute an approximate analytical power plan."""

    if args.effect_size is not None:
        result = plan_sentinels(
            args.effect_size,
            alpha=args.alpha,
            power=args.power,
            repeats_per_sentinel=args.repeats,
            intra_prompt_correlation=args.intra_prompt_correlation,
        )
        _emit({"method": "standardized-effect approximation", **result.to_dict()})
        return 0
    if args.baseline_rate is not None and args.candidate_rate is not None:
        result = samples_per_window_for_proportions(
            args.baseline_rate,
            args.candidate_rate,
            alpha=args.alpha,
            power=args.power,
        )
        _emit({"method": "two-proportion approximation", **result.to_dict()})
        return 0
    raise ValueError("Provide --effect-size or both --baseline-rate and --candidate-rate.")


def _validate_suite(args: argparse.Namespace) -> int:
    """Execute suite validation without making any provider request."""

    suite = load_suite(args.suite)
    _emit(
        {
            "digest": suite.digest,
            "name": suite.name,
            "sentinel_count": len(suite.sentinels),
            "sentinels": [
                {"id": sentinel.id, "tags": list(sentinel.tags)} for sentinel in suite.sentinels
            ],
        }
    )
    return 0


def _add_capture_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register capture options on the top-level parser."""

    parser = subparsers.add_parser(
        "capture", help="Capture repeated fingerprints from one endpoint."
    )
    parser.add_argument("--suite", required=True, help="Path to a sentinel JSON suite.")
    parser.add_argument("--output", required=True, help="Destination capture artifact JSON path.")
    parser.add_argument(
        "--provider", required=True, choices=["openai", "anthropic", "google", "gemini", "replay"]
    )
    parser.add_argument(
        "--model", required=True, help="Provider model identifier recorded in the manifest."
    )
    parser.add_argument("--repeats", type=int, default=5, help="Samples per sentinel (default: 5).")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Decoding temperature (default: 0)."
    )
    parser.add_argument("--max-tokens", type=int, help="Maximum output tokens, when supported.")
    parser.add_argument(
        "--endpoint", help="Optional complete provider endpoint URL to pin in the manifest."
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Per-request timeout in seconds."
    )
    parser.add_argument(
        "--extras", help="JSON object of non-secret provider-specific request fields."
    )
    parser.add_argument(
        "--api-key-env",
        help="Environment variable holding the provider key; defaults to the provider standard.",
    )
    parser.add_argument(
        "--replay-responses",
        help="JSON response map required by the offline replay provider.",
    )
    parser.add_argument(
        "--store-raw", action="store_true", help="Store raw outputs in the artifact."
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Stop immediately instead of recording a provider failure as an observation.",
    )
    parser.set_defaults(handler=_capture)


def _add_compare_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register statistical comparison options."""

    parser = subparsers.add_parser(
        "compare", help="Compare a candidate capture against a baseline."
    )
    parser.add_argument("--baseline", required=True, help="Baseline capture artifact JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate capture artifact JSON path.")
    parser.add_argument("--output", help="Optional JSON report destination.")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR and global-test threshold.")
    parser.add_argument(
        "--permutations", type=int, default=999, help="Label permutations per test."
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Random seed for reproducible p-values."
    )
    parser.add_argument(
        "--minimum-effect", type=float, default=0.05, help="Minimum prompt or global MMD²."
    )
    parser.add_argument(
        "--minimum-center-shift",
        type=float,
        default=0.12,
        help="Minimum medoid distance for a behavioral-shift label.",
    )
    parser.add_argument(
        "--minimum-dispersion-increase",
        type=float,
        default=0.05,
        help="Minimum increase in within-window average distance.",
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=3,
        help="Warn when a window has fewer samples than this value.",
    )
    parser.add_argument(
        "--allow-manifest-mismatch",
        action="store_true",
        help="Compare even if suite or request configuration digests differ.",
    )
    parser.set_defaults(handler=_compare)


def _add_plan_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register power-planning options."""

    parser = subparsers.add_parser("plan", help="Estimate a lower-bound sentinel suite size.")
    parser.add_argument("--effect-size", type=float, help="Target standardized mean-like effect.")
    parser.add_argument("--baseline-rate", type=float, help="Baseline binary success rate.")
    parser.add_argument("--candidate-rate", type=float, help="Candidate binary success rate.")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Two-sided significance threshold."
    )
    parser.add_argument("--power", type=float, default=0.8, help="Target detection power.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeated calls per sentinel.")
    parser.add_argument(
        "--intra-prompt-correlation",
        type=float,
        default=0.0,
        help="Assumed correlation among repeats of the same sentinel.",
    )
    parser.set_defaults(handler=_plan)


def _add_validate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register local suite-validation options."""

    parser = subparsers.add_parser("validate-suite", help="Validate a suite and print its digest.")
    parser.add_argument("suite", help="Path to a sentinel JSON suite.")
    parser.set_defaults(handler=_validate_suite)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete command-line parser."""

    parser = argparse.ArgumentParser(
        prog="silent-drift",
        description="Black-box behavioral fingerprinting for silent LLM API drift.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_capture_parser(subparsers)
    _add_compare_parser(subparsers)
    _add_plan_parser(subparsers)
    _add_validate_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and map expected operational errors to exit code one."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (OSError, ProviderError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
