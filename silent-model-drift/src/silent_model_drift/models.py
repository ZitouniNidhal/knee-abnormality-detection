"""Versioned domain models and canonical serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 1
FINGERPRINT_VERSION = 1


def utc_now() -> str:
    """Return an RFC 3339-compatible UTC timestamp with second precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Serialize public metadata deterministically for hashing and storage."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest_mapping(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Create a stable SHA-256 digest for JSON-compatible content."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Sentinel:
    """A single fixed black-box prompt probe."""

    id: str
    prompt: str
    system: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Sentinel id must be a non-empty string.")
        if not self.prompt or not self.prompt.strip():
            raise ValueError(f"Sentinel {self.id!r} must have a non-empty prompt.")
        if self.system is not None and not self.system.strip():
            raise ValueError(f"Sentinel {self.id!r} has an empty system prompt.")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError(f"Sentinel {self.id!r} has an empty tag.")

    def to_dict(self) -> dict[str, Any]:
        """Return a portable suite representation."""

        payload: dict[str, Any] = {"id": self.id, "prompt": self.prompt, "tags": list(self.tags)}
        if self.system is not None:
            payload["system"] = self.system
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Sentinel:
        """Validate and construct a sentinel from decoded JSON."""

        tags = value.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("Sentinel tags must be a list of strings.")
        sentinel_id = value.get("id")
        prompt = value.get("prompt")
        system = value.get("system")
        if not isinstance(sentinel_id, str) or not isinstance(prompt, str):
            raise ValueError("Every sentinel requires string id and prompt fields.")
        if system is not None and not isinstance(system, str):
            raise ValueError("A sentinel system field must be a string when present.")
        return cls(id=sentinel_id, prompt=prompt, system=system, tags=tuple(tags))


@dataclass(frozen=True)
class SentinelSuite:
    """A versioned set of sentinels whose digest pins the measurement contract."""

    name: str
    sentinels: tuple[Sentinel, ...]
    description: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported suite schema version "
                f"{self.schema_version}; expected {SCHEMA_VERSION}."
            )
        if not self.name or not self.name.strip():
            raise ValueError("Suite name must be a non-empty string.")
        if not self.sentinels:
            raise ValueError("A suite must contain at least one sentinel.")
        ids = [sentinel.id for sentinel in self.sentinels]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"Sentinel ids must be unique; duplicates: {', '.join(duplicates)}.")

    @property
    def digest(self) -> str:
        """Return a digest that changes whenever any sentinel behavior changes."""

        return digest_mapping(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible suite payload."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "sentinels": [sentinel.to_dict() for sentinel in self.sentinels],
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SentinelSuite:
        """Validate and construct a suite from decoded JSON."""

        schema_version = value.get("schema_version", SCHEMA_VERSION)
        name = value.get("name")
        entries = value.get("sentinels")
        description = value.get("description")
        if not isinstance(schema_version, int):
            raise ValueError("Suite schema_version must be an integer.")
        if not isinstance(name, str):
            raise ValueError("Suite name must be a string.")
        if description is not None and not isinstance(description, str):
            raise ValueError("Suite description must be a string when present.")
        if not isinstance(entries, list) or not all(isinstance(item, Mapping) for item in entries):
            raise ValueError("Suite sentinels must be a list of objects.")
        return cls(
            name=name,
            description=description,
            schema_version=schema_version,
            sentinels=tuple(Sentinel.from_dict(item) for item in entries),
        )


@dataclass(frozen=True)
class RequestConfig:
    """Public request parameters that must remain fixed across windows."""

    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None
    endpoint: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider or not self.provider.strip():
            raise ValueError("Provider must be a non-empty string.")
        if not self.model or not self.model.strip():
            raise ValueError("Model must be a non-empty string.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when supplied.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize non-secret fields used to reproduce a request."""

        payload: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "extras": dict(self.extras),
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.endpoint is not None:
            payload["endpoint"] = self.endpoint
        return payload

    @property
    def digest(self) -> str:
        """Return a reproducibility digest that excludes credentials by design."""

        return digest_mapping(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RequestConfig:
        """Construct a request configuration from persisted public metadata."""

        provider = value.get("provider")
        model = value.get("model")
        temperature = value.get("temperature", 0.0)
        max_tokens = value.get("max_tokens")
        endpoint = value.get("endpoint")
        extras = value.get("extras", {})
        if not isinstance(provider, str) or not isinstance(model, str):
            raise ValueError("Request config requires string provider and model fields.")
        if not isinstance(temperature, (int, float)):
            raise ValueError("Request config temperature must be numeric.")
        if max_tokens is not None and not isinstance(max_tokens, int):
            raise ValueError("Request config max_tokens must be an integer when present.")
        if endpoint is not None and not isinstance(endpoint, str):
            raise ValueError("Request config endpoint must be a string when present.")
        if not isinstance(extras, Mapping):
            raise ValueError("Request config extras must be an object.")
        return cls(
            provider=provider,
            model=model,
            temperature=float(temperature),
            max_tokens=max_tokens,
            endpoint=endpoint,
            extras=dict(extras),
        )


@dataclass(frozen=True)
class ResponseFingerprint:
    """A raw-text-free approximation of an endpoint response."""

    exact_hash: str
    token_hashes: tuple[str, ...]
    ngram_hashes: tuple[str, ...]
    character_count: int
    token_count: int
    structure_hash: str
    is_json: bool
    version: int = FINGERPRINT_VERSION

    def __post_init__(self) -> None:
        if self.version != FINGERPRINT_VERSION:
            raise ValueError(
                f"Unsupported fingerprint version {self.version}; expected {FINGERPRINT_VERSION}."
            )
        if self.character_count < 0 or self.token_count < 0:
            raise ValueError("Fingerprint counts cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return the privacy-conscious persisted representation."""

        return {
            "version": self.version,
            "exact_hash": self.exact_hash,
            "token_hashes": list(self.token_hashes),
            "ngram_hashes": list(self.ngram_hashes),
            "character_count": self.character_count,
            "token_count": self.token_count,
            "structure_hash": self.structure_hash,
            "is_json": self.is_json,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResponseFingerprint:
        """Construct a fingerprint from an artifact record."""

        hashes = ("token_hashes", "ngram_hashes")
        for key in hashes:
            current = value.get(key)
            if not isinstance(current, list) or not all(isinstance(item, str) for item in current):
                raise ValueError(f"Fingerprint {key} must be a list of strings.")
        required_strings = ("exact_hash", "structure_hash")
        if not all(isinstance(value.get(key), str) for key in required_strings):
            raise ValueError("Fingerprint hashes must be strings.")
        if not isinstance(value.get("character_count"), int) or not isinstance(
            value.get("token_count"), int
        ):
            raise ValueError("Fingerprint counts must be integers.")
        if not isinstance(value.get("is_json"), bool):
            raise ValueError("Fingerprint is_json must be a boolean.")
        version = value.get("version", FINGERPRINT_VERSION)
        if not isinstance(version, int):
            raise ValueError("Fingerprint version must be an integer.")
        return cls(
            version=version,
            exact_hash=value["exact_hash"],
            token_hashes=tuple(value["token_hashes"]),
            ngram_hashes=tuple(value["ngram_hashes"]),
            character_count=value["character_count"],
            token_count=value["token_count"],
            structure_hash=value["structure_hash"],
            is_json=value["is_json"],
        )


@dataclass(frozen=True)
class Observation:
    """One response sample from one sentinel in a capture run."""

    sentinel_id: str
    repeat_index: int
    collected_at: str
    fingerprint: ResponseFingerprint
    latency_ms: float | None = None
    raw_response: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not self.sentinel_id:
            raise ValueError("Observation sentinel_id cannot be empty.")
        if self.repeat_index < 0:
            raise ValueError("Observation repeat_index cannot be negative.")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("Observation latency_ms cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the observation without adding implicit fields."""

        payload: dict[str, Any] = {
            "sentinel_id": self.sentinel_id,
            "repeat_index": self.repeat_index,
            "collected_at": self.collected_at,
            "fingerprint": self.fingerprint.to_dict(),
        }
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.raw_response is not None:
            payload["raw_response"] = self.raw_response
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Observation:
        """Construct an observation from a stored run artifact."""

        sentinel_id = value.get("sentinel_id")
        repeat_index = value.get("repeat_index")
        collected_at = value.get("collected_at")
        fingerprint = value.get("fingerprint")
        latency_ms = value.get("latency_ms")
        raw_response = value.get("raw_response")
        error_type = value.get("error_type")
        if not isinstance(sentinel_id, str) or not isinstance(repeat_index, int):
            raise ValueError("Observation requires string sentinel_id and integer repeat_index.")
        if not isinstance(collected_at, str) or not isinstance(fingerprint, Mapping):
            raise ValueError("Observation requires collected_at and fingerprint fields.")
        if latency_ms is not None and not isinstance(latency_ms, (int, float)):
            raise ValueError("Observation latency_ms must be numeric when present.")
        if raw_response is not None and not isinstance(raw_response, str):
            raise ValueError("Observation raw_response must be a string when present.")
        if error_type is not None and not isinstance(error_type, str):
            raise ValueError("Observation error_type must be a string when present.")
        return cls(
            sentinel_id=sentinel_id,
            repeat_index=repeat_index,
            collected_at=collected_at,
            fingerprint=ResponseFingerprint.from_dict(fingerprint),
            latency_ms=float(latency_ms) if latency_ms is not None else None,
            raw_response=raw_response,
            error_type=error_type,
        )


@dataclass(frozen=True)
class RunManifest:
    """Reproducibility metadata for a group of sentinel observations."""

    run_id: str
    created_at: str
    suite_name: str
    suite_digest: str
    request_config: RequestConfig
    repeats: int
    store_raw_responses: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported run schema version {self.schema_version}; expected {SCHEMA_VERSION}."
            )
        if not self.run_id:
            raise ValueError("Run id cannot be empty.")
        if not self.suite_name or not self.suite_digest:
            raise ValueError("Run manifest requires suite name and digest.")
        if self.repeats <= 0:
            raise ValueError("Run repeats must be positive.")

    @property
    def request_config_digest(self) -> str:
        """Expose the request digest without persisting redundant state."""

        return self.request_config.digest

    def to_dict(self) -> dict[str, Any]:
        """Return the artifact manifest as JSON-compatible data."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "suite_name": self.suite_name,
            "suite_digest": self.suite_digest,
            "request_config": self.request_config.to_dict(),
            "request_config_digest": self.request_config_digest,
            "repeats": self.repeats,
            "store_raw_responses": self.store_raw_responses,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        """Construct a manifest and verify persisted digests when available."""

        required_strings = ("run_id", "created_at", "suite_name", "suite_digest")
        if not all(isinstance(value.get(key), str) for key in required_strings):
            raise ValueError("Run manifest has missing or invalid string fields.")
        request_config = value.get("request_config")
        repeats = value.get("repeats")
        store_raw = value.get("store_raw_responses", False)
        schema_version = value.get("schema_version", SCHEMA_VERSION)
        if not isinstance(request_config, Mapping) or not isinstance(repeats, int):
            raise ValueError("Run manifest requires request_config and integer repeats.")
        if not isinstance(store_raw, bool) or not isinstance(schema_version, int):
            raise ValueError("Run manifest has invalid boolean or schema version fields.")
        manifest = cls(
            run_id=value["run_id"],
            created_at=value["created_at"],
            suite_name=value["suite_name"],
            suite_digest=value["suite_digest"],
            request_config=RequestConfig.from_dict(request_config),
            repeats=repeats,
            store_raw_responses=store_raw,
            schema_version=schema_version,
        )
        saved_digest = value.get("request_config_digest")
        if saved_digest is not None and saved_digest != manifest.request_config_digest:
            raise ValueError("Run manifest request_config_digest does not match request_config.")
        return manifest


@dataclass(frozen=True)
class CaptureRun:
    """A portable baseline or candidate capture artifact."""

    manifest: RunManifest
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("A capture run must contain at least one observation.")

    @classmethod
    def create(
        cls,
        *,
        suite: SentinelSuite,
        request_config: RequestConfig,
        repeats: int,
        observations: Sequence[Observation],
        store_raw_responses: bool = False,
        run_id: str | None = None,
    ) -> CaptureRun:
        """Create a run with fresh metadata after a capture operation."""

        return cls(
            manifest=RunManifest(
                run_id=run_id or uuid4().hex,
                created_at=utc_now(),
                suite_name=suite.name,
                suite_digest=suite.digest,
                request_config=request_config,
                repeats=repeats,
                store_raw_responses=store_raw_responses,
            ),
            observations=tuple(observations),
        )

    def by_sentinel(self) -> dict[str, list[ResponseFingerprint]]:
        """Group fingerprints by sentinel id for statistical comparison."""

        grouped: dict[str, list[ResponseFingerprint]] = {}
        for observation in self.observations:
            grouped.setdefault(observation.sentinel_id, []).append(observation.fingerprint)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        """Return a stable capture-artifact payload."""

        return {
            "schema_version": SCHEMA_VERSION,
            "manifest": self.manifest.to_dict(),
            "observations": [observation.to_dict() for observation in self.observations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CaptureRun:
        """Load a persisted capture artifact with validation."""

        schema_version = value.get("schema_version", SCHEMA_VERSION)
        manifest = value.get("manifest")
        observations = value.get("observations")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported capture schema version {schema_version}; expected {SCHEMA_VERSION}."
            )
        if not isinstance(manifest, Mapping):
            raise ValueError("Capture artifact requires a manifest object.")
        if not isinstance(observations, list) or not all(
            isinstance(item, Mapping) for item in observations
        ):
            raise ValueError("Capture artifact observations must be a list of objects.")
        return cls(
            manifest=RunManifest.from_dict(manifest),
            observations=tuple(Observation.from_dict(item) for item in observations),
        )
