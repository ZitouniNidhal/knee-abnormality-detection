"""Small standard-library clients for common text-generation APIs.

The adapters keep credentials outside capture artifacts. They intentionally
cover a narrow, reproducible text-only request surface; use RequestConfig
extras for pinned provider-specific options such as response formats or tools.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .models import RequestConfig, Sentinel


class ProviderError(RuntimeError):
    """Raised when an adapter cannot safely obtain a provider response."""


class CompletionClient(Protocol):
    """Protocol implemented by all synchronous endpoint adapters."""

    def complete(self, sentinel: Sentinel, request_config: RequestConfig) -> str:
        """Return one text representation of a provider response."""


def _load_key(api_key: str | None, environment_variable: str) -> str:
    """Resolve a key without ever persisting it in an artifact."""

    value = api_key or os.environ.get(environment_variable)
    if not value:
        raise ProviderError(f"Set {environment_variable} before using this provider.")
    return value


def _json_post(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> dict[str, Any]:
    """Send a JSON request and produce a sanitized provider error on failure."""

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", **dict(headers)},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise ProviderError(f"Provider request failed with HTTP status {error.code}.") from error
    except URLError as error:
        raise ProviderError(
            f"Provider request could not reach its endpoint: {error.reason}."
        ) from error
    except TimeoutError as error:
        raise ProviderError("Provider request timed out.") from error
    except json.JSONDecodeError as error:
        raise ProviderError("Provider returned a non-JSON response.") from error
    if not isinstance(decoded, dict):
        raise ProviderError("Provider returned an unexpected JSON root.")
    return decoded


def _add_extras(payload: dict[str, Any], extras: Mapping[str, Any], reserved: set[str]) -> None:
    """Merge fixed extra settings while preventing hidden configuration overrides."""

    collision = reserved & set(extras)
    if collision:
        raise ValueError(
            "Request config extras cannot override required fields: " + ", ".join(sorted(collision))
        )
    payload.update(dict(extras))


class OpenAIChatCompletionsClient:
    """Adapter for the OpenAI Chat Completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = _load_key(api_key, "OPENAI_API_KEY")
        self._endpoint = endpoint
        self._timeout = timeout

    def complete(self, sentinel: Sentinel, request_config: RequestConfig) -> str:
        """Issue one fixed Chat Completions request."""

        messages: list[dict[str, str]] = []
        if sentinel.system:
            messages.append({"role": "developer", "content": sentinel.system})
        messages.append({"role": "user", "content": sentinel.prompt})
        payload: dict[str, Any] = {
            "model": request_config.model,
            "messages": messages,
            "temperature": request_config.temperature,
        }
        if request_config.max_tokens is not None:
            payload["max_completion_tokens"] = request_config.max_tokens
        _add_extras(
            payload,
            request_config.extras,
            {"model", "messages", "temperature", "max_completion_tokens"},
        )
        response = _json_post(
            request_config.endpoint or self._endpoint,
            {"Authorization": f"Bearer {self._api_key}"},
            payload,
            self._timeout,
        )
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                "OpenAI response did not contain a first completion message."
            ) from error
        if not isinstance(message, Mapping):
            raise ProviderError("OpenAI completion message has an unexpected shape.")
        content = message.get("content")
        if isinstance(content, str):
            return content
        refusal = message.get("refusal")
        if isinstance(refusal, str):
            return "__provider_refusal__ " + refusal
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            return "__provider_tool_call__ " + json.dumps(
                tool_calls, ensure_ascii=False, sort_keys=True
            )
        raise ProviderError(
            "OpenAI completion contained neither text, refusal, nor tool call output."
        )


class AnthropicMessagesClient:
    """Adapter for the Anthropic Messages endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = "https://api.anthropic.com/v1/messages",
        api_version: str = "2023-06-01",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = _load_key(api_key, "ANTHROPIC_API_KEY")
        self._endpoint = endpoint
        self._api_version = api_version
        self._timeout = timeout

    def complete(self, sentinel: Sentinel, request_config: RequestConfig) -> str:
        """Issue one fixed Messages request."""

        payload: dict[str, Any] = {
            "model": request_config.model,
            "max_tokens": request_config.max_tokens or 256,
            "temperature": request_config.temperature,
            "messages": [{"role": "user", "content": sentinel.prompt}],
        }
        if sentinel.system:
            payload["system"] = sentinel.system
        _add_extras(
            payload,
            request_config.extras,
            {"model", "max_tokens", "temperature", "messages", "system"},
        )
        response = _json_post(
            request_config.endpoint or self._endpoint,
            {"x-api-key": self._api_key, "anthropic-version": self._api_version},
            payload,
            self._timeout,
        )
        blocks = response.get("content")
        if not isinstance(blocks, list):
            raise ProviderError("Anthropic response did not contain a content list.")
        text_blocks = [
            block["text"]
            for block in blocks
            if isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if text_blocks:
            return "\n".join(text_blocks)
        if blocks:
            return "__provider_nontext_content__ " + json.dumps(
                blocks, ensure_ascii=False, sort_keys=True
            )
        raise ProviderError("Anthropic response contained no content blocks.")


class GoogleGeminiClient:
    """Adapter for the Google Gemini generateContent endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint_base: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = _load_key(api_key, "GEMINI_API_KEY")
        self._endpoint_base = endpoint_base.rstrip("/")
        self._timeout = timeout

    def complete(self, sentinel: Sentinel, request_config: RequestConfig) -> str:
        """Issue one fixed Gemini generateContent request."""

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": sentinel.prompt}]}],
            "generationConfig": {"temperature": request_config.temperature},
        }
        if sentinel.system:
            payload["systemInstruction"] = {"parts": [{"text": sentinel.system}]}
        if request_config.max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = request_config.max_tokens
        _add_extras(
            payload,
            request_config.extras,
            {"contents", "generationConfig", "systemInstruction"},
        )
        if request_config.endpoint:
            endpoint = request_config.endpoint
        else:
            model = quote(request_config.model, safe="")
            endpoint = f"{self._endpoint_base}/models/{model}:generateContent"
        response = _json_post(
            endpoint,
            {"x-goog-api-key": self._api_key},
            payload,
            self._timeout,
        )
        try:
            parts = response["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                "Gemini response did not contain candidate text content."
            ) from error
        if not isinstance(parts, list):
            raise ProviderError("Gemini content parts have an unexpected shape.")
        text_parts = [
            part["text"]
            for part in parts
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        if text_parts:
            return "\n".join(text_parts)
        return "__provider_nontext_content__ " + json.dumps(
            parts, ensure_ascii=False, sort_keys=True
        )


class ReplayClient:
    """Offline deterministic client for examples, tests, and pipeline rehearsals."""

    def __init__(self, responses: Mapping[str, str | list[str]]) -> None:
        self._responses: dict[str, list[str]] = {}
        self._offsets: dict[str, int] = {}
        for sentinel_id, output in responses.items():
            values = [output] if isinstance(output, str) else list(output)
            if not values or not all(isinstance(value, str) for value in values):
                raise ValueError(
                    "Replay responses must map each sentinel id to one or more strings."
                )
            self._responses[sentinel_id] = values

    @classmethod
    def from_json_file(cls, path: str | Path) -> ReplayClient:
        """Load a sentinel-id-to-response mapping from JSON."""

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"Replay response file does not exist: {path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Replay response file is not valid JSON: {error.msg}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("Replay response root must be an object.")
        return cls(payload)

    def complete(self, sentinel: Sentinel, request_config: RequestConfig) -> str:
        """Cycle through pre-recorded responses for a sentinel."""

        del request_config
        try:
            values = self._responses[sentinel.id]
        except KeyError as error:
            raise ProviderError(
                f"Replay file has no response for sentinel {sentinel.id!r}."
            ) from error
        offset = self._offsets.get(sentinel.id, 0)
        self._offsets[sentinel.id] = offset + 1
        return values[offset % len(values)]


def build_client(
    provider: str,
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    timeout: float = 60.0,
    replay_responses: str | Path | None = None,
) -> CompletionClient:
    """Construct a supported client without accepting credentials in artifacts."""

    normalized = provider.casefold()
    if normalized == "openai":
        return OpenAIChatCompletionsClient(
            api_key=api_key,
            endpoint=endpoint or "https://api.openai.com/v1/chat/completions",
            timeout=timeout,
        )
    if normalized == "anthropic":
        return AnthropicMessagesClient(
            api_key=api_key,
            endpoint=endpoint or "https://api.anthropic.com/v1/messages",
            timeout=timeout,
        )
    if normalized in {"google", "gemini"}:
        return GoogleGeminiClient(
            api_key=api_key,
            endpoint_base=endpoint or "https://generativelanguage.googleapis.com/v1beta",
            timeout=timeout,
        )
    if normalized == "replay":
        if replay_responses is None:
            raise ValueError("The replay provider requires --replay-responses.")
        return ReplayClient.from_json_file(replay_responses)
    raise ValueError("Unsupported provider. Choose openai, anthropic, google, or replay.")
