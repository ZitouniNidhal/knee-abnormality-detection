"""Production-grade structured logging for medical AI pipelines.

Provides a single ``get_logger`` entry-point that returns a structlog logger
pre-configured for both human-readable console output (development) and
machine-parseable JSON (production/CI).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def _is_json_mode() -> bool:
    return os.environ.get("LOG_FORMAT", "").lower() in {"json", "production"}


def configure_logging(level: str = "INFO", *, json: bool | None = None) -> None:
    """Configure structlog + stdlib logging once at application startup."""
    use_json = json if json is not None else _is_json_mode()
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if use_json:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for *name*."""
    return structlog.get_logger(name)  # type: ignore[return-value]
