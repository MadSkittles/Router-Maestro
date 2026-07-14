"""Fixtures for local live-backend integration tests.

These tests intentionally reuse the developer's existing Router-Maestro
configuration and GitHub Copilot auth files. They are outside the default
pytest tree and are only run with ``uv run pytest integration_tests/ -v``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
import pytest

from router_maestro.auth import AuthManager, AuthStorage, OAuthCredential, run_async
from router_maestro.config.server import get_current_context_api_key
from router_maestro.providers import CopilotProvider, ModelInfo
from router_maestro.routing.model_ref import ModelRef
from router_maestro.utils.model_match import normalize_model_id
from router_maestro.utils.responses_bridge import RESPONSES_ELIGIBLE_MODELS

STARTUP_TIMEOUT_SECONDS = 45.0
REQUEST_TIMEOUT_SECONDS = 120.0
STREAM_TIMEOUT_SECONDS = 180.0
RETRY_BACKOFF_SECONDS = 10.0
MAX_RETRIES_ON_TIMEOUT = 2
DEFAULT_MAX_MODEL_MATRIX = 0
DEFAULT_MAX_REASONING_MATRIX = 1
DEFAULT_API_KEY = "router-maestro-integration-test"
COPILOT_PROVIDER = "github-copilot"
TEXT_PROMPT = "Reply with exactly the word pong. Do not add punctuation or any other words."
TOOL_PROMPT = "Use the provided get_weather tool for Shanghai. Do not answer directly."
REASONING_PROMPT = (
    "Three friends Alice, Bob, and Carol have ages summing to 60. "
    "Alice is twice as old as Bob was when Alice was as old as Bob is now. "
    "Carol is 4 years younger than Alice. Find each age, showing reasoning."
)
ANTHROPIC_THINKING_BUDGETS: tuple[int | None, ...] = (None, 1024, 4096, 16000)
OPENAI_REASONING_EFFORTS: tuple[str | None, ...] = (None, "low", "medium", "high")
STREAM_MODES: tuple[bool, ...] = (False, True)
COMPAT_PROBE_BACKOFF_SECONDS = 0.25
COMPAT_PROBE_MAX_ATTEMPTS = 3
_PROVIDER_FAILURE_SIGNAL_HEADER = "X-Router-Maestro-Error-Signal"
_COPILOT_BARE_BAD_REQUEST_SIGNAL = "copilot_bare_bad_request"
_MAX_COMPAT_PROBE_DIAGNOSTIC_BODY = 256
RESPONSES_ONLY_CHAT_MODELS = {
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.4-mini",
    "gpt-5.5",
}


@dataclass(frozen=True)
class LiveServer:
    """Connection details for the locally started Router-Maestro server."""

    base_url: str
    api_key: str
    process: subprocess.Popen[str]


def post_gemini_generate_content_compat_probe(
    client: httpx.Client,
    model: str,
) -> httpx.Response:
    """Run the fixed Gemini generateContent compatibility probe."""
    return _post_compat_probe_with_marked_400_retry(
        client,
        f"/api/gemini/v1beta/models/{gemini_model_path(model)}:generateContent",
        json_payload=gemini_compat_payload(),
    )


def post_openai_chat_compat_probe(
    client: httpx.Client,
    model: str,
) -> httpx.Response:
    """Run the fixed non-stream OpenAI Chat compatibility probe."""
    return _post_compat_probe_with_marked_400_retry(
        client,
        "/api/openai/v1/chat/completions",
        json_payload=openai_chat_usage_payload(model),
    )


@contextmanager
def stream_openai_chat_compat_probe(
    client: httpx.Client,
    model: str,
    *,
    timeout: float,
) -> Iterator[httpx.Response]:
    """Run the fixed streaming OpenAI Chat compatibility probe."""
    with _stream_compat_probe_with_marked_400_retry(
        client,
        "/api/openai/v1/chat/completions",
        json_payload=openai_chat_usage_payload(model, stream=True),
        timeout=timeout,
    ) as response:
        yield response


def _post_compat_probe_with_marked_400_retry(
    client: httpx.Client,
    path: str,
    *,
    json_payload: dict[str, Any],
    backoff_seconds: float = COMPAT_PROBE_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """POST one fixed idempotent probe, retrying only a marked Copilot bare 400."""
    summaries: list[str] = []
    for attempt in range(1, COMPAT_PROBE_MAX_ATTEMPTS + 1):
        response = client.post(path, json=json_payload)
        if not _is_marked_copilot_bare_bad_request(response):
            return response
        summaries.append(_compat_probe_response_summary(response))
        response.close()
        if attempt < COMPAT_PROBE_MAX_ATTEMPTS:
            sleep(backoff_seconds)

    details = "; ".join(
        f"attempt {attempt}: {summary}" for attempt, summary in enumerate(summaries, start=1)
    )
    raise AssertionError(
        f"compatibility probe failed after {COMPAT_PROBE_MAX_ATTEMPTS} attempts; {details}"
    )


@contextmanager
def _stream_compat_probe_with_marked_400_retry(
    client: httpx.Client,
    path: str,
    *,
    json_payload: dict[str, Any],
    timeout: float,
    backoff_seconds: float = COMPAT_PROBE_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[httpx.Response]:
    """Open one fixed idempotent stream probe, retrying only before yielding a response."""
    summaries: list[str] = []
    for attempt in range(1, COMPAT_PROBE_MAX_ATTEMPTS + 1):
        with client.stream("POST", path, json=json_payload, timeout=timeout) as response:
            if not _is_marked_copilot_bare_bad_request(response):
                yield response
                return
            summaries.append(_compat_probe_response_summary(response))
        if attempt < COMPAT_PROBE_MAX_ATTEMPTS:
            sleep(backoff_seconds)

    details = "; ".join(
        f"attempt {attempt}: {summary}" for attempt, summary in enumerate(summaries, start=1)
    )
    raise AssertionError(
        f"stream compatibility probe failed after {COMPAT_PROBE_MAX_ATTEMPTS} attempts; {details}"
    )


def _is_marked_copilot_bare_bad_request(response: httpx.Response) -> bool:
    return (
        response.status_code == 400
        and response.headers.get(_PROVIDER_FAILURE_SIGNAL_HEADER)
        == _COPILOT_BARE_BAD_REQUEST_SIGNAL
    )


def _compat_probe_response_summary(response: httpx.Response) -> str:
    if not response.is_stream_consumed:
        response.read()
    body = response.content[:_MAX_COMPAT_PROBE_DIAGNOSTIC_BODY]
    suffix = b"..." if len(response.content) > len(body) else b""
    return f"HTTP {response.status_code}, body={(body + suffix)!r}"


@pytest.fixture(scope="session")
def live_server() -> Iterator[LiveServer]:
    """Start a local RM server against the user's existing config/auth files."""
    _require_github_copilot_auth()

    api_key = os.environ.get("RM_INTEGRATION_API_KEY") or get_current_context_api_key()
    if not api_key:
        api_key = DEFAULT_API_KEY

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["ROUTER_MAESTRO_API_KEY"] = api_key
    env.setdefault("ROUTER_MAESTRO_LOG_LEVEL", "INFO")
    if env.get("RM_INTEGRATION_RESPONSES_CHAT") is None:
        env["ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API"] = "1"

    process = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "router_maestro.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            env["ROUTER_MAESTRO_LOG_LEVEL"].lower(),
        ],
        cwd=_repo_root(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_health(base_url, process)
        yield LiveServer(base_url=base_url, api_key=api_key, process=process)
    finally:
        _terminate_process(process)


class _RetryTransport(httpx.BaseTransport):
    """Wraps an httpx transport with automatic retry on ReadTimeout.

    Copilot rate-limits heavy usage within a single session; rather than
    failing the test, back off and retry once.
    """

    def __init__(self, transport: httpx.BaseTransport):
        self._inner = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(MAX_RETRIES_ON_TIMEOUT + 1):
            try:
                return self._inner.handle_request(request)
            except httpx.ReadTimeout:
                if attempt >= MAX_RETRIES_ON_TIMEOUT:
                    raise
                time.sleep(RETRY_BACKOFF_SECONDS)

    def close(self) -> None:
        self._inner.close()


@pytest.fixture(scope="session")
def client(live_server: LiveServer) -> Iterator[httpx.Client]:
    """HTTP client authenticated against the local RM server."""
    headers = {"Authorization": f"Bearer {live_server.api_key}"}
    transport = _RetryTransport(
        httpx.HTTPTransport(proxy=os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"))
    )
    with httpx.Client(
        base_url=live_server.base_url,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        transport=transport,
    ) as http_client:
        yield http_client


@pytest.fixture(scope="session")
def unauthenticated_client(live_server: LiveServer) -> Iterator[httpx.Client]:
    """HTTP client without the Router-Maestro API key."""
    with httpx.Client(
        base_url=live_server.base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as http_client:
        yield http_client


@pytest.fixture(scope="session")
def copilot_catalog() -> dict[str, ModelInfo]:
    """Load live Copilot metadata without mutating the user's credential file."""
    manager = AuthManager()
    credential = manager.get_credential(COPILOT_PROVIDER)
    if not isinstance(credential, OAuthCredential):
        pytest.skip(
            "GitHub Copilot auth is not configured. Run "
            f"`uv run router-maestro auth login {COPILOT_PROVIDER}` first."
        )

    provider = CopilotProvider()
    provider.auth_manager = _non_persisting_auth_snapshot(credential)
    provider._cached_token = credential.access
    provider._token_expires = credential.expires
    if credential.api_endpoint:
        provider._api_base = credential.api_endpoint
    try:
        models = run_async(provider.list_models(force_refresh=True))
    finally:
        run_async(provider.close())
    public_ids = [_provider_model_id(model.provider, model.id) for model in models]
    duplicate_ids = sorted(
        model_id for model_id in set(public_ids) if public_ids.count(model_id) > 1
    )
    if duplicate_ids:
        pytest.fail(f"Copilot provider catalog returned duplicate public IDs: {duplicate_ids}")
    catalog = dict(zip(public_ids, models, strict=True))
    if not catalog:
        pytest.fail(
            "No github-copilot models are available. Run "
            "`uv run router-maestro auth login github-copilot` first."
        )
    return catalog


class _NonPersistingAuthSnapshot:
    """Minimal in-memory AuthManager contract used by the catalog probe."""

    def __init__(self, credential: OAuthCredential) -> None:
        self.storage = AuthStorage()
        self.storage.set(COPILOT_PROVIDER, credential.model_copy(deep=True))

    def get_credential(self, provider: str):
        return self.storage.get(provider)

    def save(self) -> None:
        return None


def _non_persisting_auth_snapshot(credential: OAuthCredential) -> _NonPersistingAuthSnapshot:
    """Build an in-memory credential manager for the parallel catalog probe."""
    return _NonPersistingAuthSnapshot(credential)


@pytest.fixture(scope="session")
def copilot_models(
    client: httpx.Client,
    copilot_catalog: dict[str, ModelInfo],
) -> list[str]:
    """Read invokable Copilot IDs from Router-Maestro's public model list.

    The direct provider catalog supplies endpoint capability metadata, while
    the server endpoint owns the public identity that this matrix must round
    trip back through each invocation route.
    """
    response = client.get("/api/openai/v1/models")
    assert_http_success(response)
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        pytest.fail("OpenAI model endpoint returned a malformed list")
    public_ids = [
        model.get("id")
        for model in payload["data"]
        if isinstance(model, dict) and model.get("owned_by") == COPILOT_PROVIDER
    ]
    if any(not isinstance(model_id, str) or not model_id for model_id in public_ids):
        pytest.fail("OpenAI model endpoint returned an invalid Copilot public ID")
    duplicate_ids = sorted(
        model_id for model_id in set(public_ids) if public_ids.count(model_id) > 1
    )
    if duplicate_ids:
        pytest.fail(f"OpenAI model endpoint returned duplicate Copilot public IDs: {duplicate_ids}")
    models = public_ids
    missing_metadata = [model_id for model_id in models if model_id not in copilot_catalog]
    missing_public = [model_id for model_id in copilot_catalog if model_id not in models]
    if missing_metadata or missing_public:
        pytest.fail(
            "Copilot provider catalog and Router-Maestro public model list disagree: "
            f"missing_metadata={missing_metadata}, missing_public={missing_public}"
        )
    return models


@pytest.fixture(scope="session")
def model_matrix(copilot_models: list[str]) -> list[str]:
    """Model subset used for broad live model coverage."""
    requested = os.environ.get("RM_INTEGRATION_MODELS")
    if requested:
        models = [item.strip() for item in requested.split(",") if item.strip()]
        missing = [model for model in models if model not in copilot_models]
        if missing:
            pytest.fail(f"RM_INTEGRATION_MODELS contains unavailable models: {missing}")
        return models

    max_models = _int_env("RM_INTEGRATION_MAX_MODELS", DEFAULT_MAX_MODEL_MATRIX)
    if max_models <= 0:
        return copilot_models

    ordered = _prioritize_models(copilot_models)
    return ordered[:max_models]


@pytest.fixture(scope="session")
def anthropic_thinking_models(copilot_catalog: dict[str, ModelInfo]) -> list[str]:
    """Catalog-declared Claude reasoning models for thinking budget coverage."""
    return _required_reasoning_subset(
        list(copilot_catalog),
        predicate=lambda model: (
            _is_anthropic_thinking_model(model) and model_supports_reasoning(copilot_catalog[model])
        ),
        description="reasoning-capable Claude Copilot models",
    )


@pytest.fixture(scope="session")
def anthropic_unsupported_thinking_model(
    copilot_catalog: dict[str, ModelInfo],
) -> str:
    """One Claude model whose live catalog explicitly rejects reasoning."""
    candidates = [
        model
        for model in _prioritize_models(list(copilot_catalog))
        if bare_model(model).lower().startswith("claude-")
        and model_explicitly_rejects_reasoning(copilot_catalog[model])
    ]
    if not candidates:
        pytest.skip("No catalog-declared non-reasoning Claude model is available")
    return candidates[0]


@pytest.fixture(scope="session")
def anthropic_effort_profile(copilot_models: list[str]) -> tuple[str, str]:
    """Choose a Claude model and an effort tier verified by the Copilot endpoint."""
    profiles = (
        ("claude-opus-4.8", "xhigh"),
        ("claude-opus-4.7", "xhigh"),
        ("claude-opus-4.6", "high"),
        ("claude-sonnet-4.6", "high"),
    )
    models_by_normalized_id = {
        normalize_model_id(bare_model(model)): model for model in copilot_models
    }
    for bare_id, effort in profiles:
        model = models_by_normalized_id.get(normalize_model_id(bare_id))
        if model is not None:
            return model, effort
    pytest.skip("No Copilot Claude model with output_config.effort support is available")


@pytest.fixture(scope="session")
def anthropic_gpt5_bridge_models(copilot_catalog: dict[str, ModelInfo]) -> list[str]:
    """Catalog-declared Responses models for Anthropic bridge coverage."""
    return _required_reasoning_subset(
        list(copilot_catalog),
        predicate=lambda model: (
            bare_model(model).lower().startswith("gpt-5")
            and select_live_http_endpoint(model, copilot_catalog) is LiveHttpEndpoint.RESPONSES
        ),
        description="Responses-eligible GPT-5 Copilot models",
    )


@pytest.fixture(scope="session")
def openai_reasoning_models(copilot_models: list[str]) -> list[str]:
    """OpenAI Chat models that accept reasoning_effort."""
    return _required_reasoning_subset(
        copilot_models,
        predicate=_is_openai_chat_reasoning_model,
        description="OpenAI Chat reasoning Copilot models",
    )


@pytest.fixture(scope="session")
def gemini_family_models(copilot_models: list[str]) -> list[str]:
    """Gemini-family models for explicit Gemini API surface coverage."""
    selected = [
        model
        for model in _prioritize_models(copilot_models)
        if bare_model(model).startswith("gemini-")
    ]
    if not selected:
        pytest.skip("No Gemini-family Copilot models are available from GitHub Copilot")
    return selected


@pytest.fixture(scope="session")
def chat_model(copilot_models: list[str]) -> str:
    """Model for OpenAI Chat, Anthropic, and Gemini compatibility paths."""
    requested = os.environ.get("RM_INTEGRATION_MODEL")
    if requested:
        if requested in copilot_models:
            return requested
        pytest.fail(f"RM_INTEGRATION_MODEL={requested!r} is not in available Copilot models")

    preferred = (
        "github-copilot/gpt-4o-mini",
        "github-copilot/gpt-4o",
        "github-copilot/claude-haiku-4.5",
        "github-copilot/claude-sonnet-4.5",
    )
    return _first_available(copilot_models, preferred) or copilot_models[0]


@pytest.fixture(scope="session")
def tool_model(copilot_models: list[str]) -> str:
    """Model selected for forced tool-call scenarios.

    Used against ``/chat/completions`` and ``/anthropic/v1/messages``, so it must
    be a true chat model. Responses-only models (e.g. ``gpt-5.4-mini``) reject
    those endpoints, so they are excluded from the candidate pool.
    """
    requested = os.environ.get("RM_INTEGRATION_TOOL_MODEL")
    if requested:
        if requested in copilot_models:
            return requested
        pytest.fail(f"RM_INTEGRATION_TOOL_MODEL={requested!r} is not in available Copilot models")

    chat_capable = [
        model for model in copilot_models if bare_model(model) not in RESPONSES_ONLY_CHAT_MODELS
    ]
    preferred = (
        "github-copilot/gpt-4o",
        "github-copilot/gpt-4o-mini",
        "github-copilot/claude-sonnet-4.5",
        "github-copilot/claude-haiku-4.5",
        "github-copilot/gpt-4.1",
    )
    selected = _first_available(chat_capable, preferred)
    if selected:
        return selected
    if chat_capable:
        return chat_capable[0]
    pytest.skip("No chat-capable Copilot model available for forced tool-call tests")


@pytest.fixture(scope="session")
def responses_model(
    copilot_models: list[str],
    copilot_catalog: dict[str, ModelInfo],
) -> str:
    """Model for the OpenAI Responses path."""
    requested = os.environ.get("RM_INTEGRATION_RESPONSES_MODEL")
    if requested:
        if requested in copilot_models:
            return requested
        pytest.fail(
            f"RM_INTEGRATION_RESPONSES_MODEL={requested!r} is not in available Copilot models"
        )

    eligible = [
        model for model in copilot_models if is_responses_eligible_model(model, copilot_catalog)
    ]
    preferred = (
        "github-copilot/gpt-5.4-mini",
        "github-copilot/gpt-5.4",
        "github-copilot/gpt-5.3-codex",
        "github-copilot/gpt-5.2",
    )
    selected = _first_available(eligible, preferred)
    if selected:
        return selected
    selected = next(iter(eligible), None)
    if selected:
        return selected
    pytest.skip("No Copilot model available for the /responses endpoint")


@pytest.fixture(scope="session")
def responses_top_p_model(
    copilot_models: list[str],
    copilot_catalog: dict[str, ModelInfo],
) -> str:
    """Responses model whose upstream contract accepts top_p.

    Copilot's catalog currently declares endpoint support but not per-option
    support. Keep this verified compatibility list isolated from endpoint
    planning so an unsupported top_p is never silently discarded.
    """
    eligible = [
        model for model in copilot_models if is_responses_eligible_model(model, copilot_catalog)
    ]
    preferred = (
        "github-copilot/gpt-5.3-codex",
        "github-copilot/mai-code-1-flash-picker",
        "github-copilot/gpt-5-mini",
    )
    selected = _first_available(eligible, preferred)
    if selected:
        return selected
    pytest.skip("No Responses model with verified top_p support is available")


def openai_chat_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Small deterministic OpenAI Chat request."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
        "temperature": 0,
        "max_tokens": 16,
        "stream": stream,
    }


def model_matrix_chat_payload(model: str) -> dict[str, Any]:
    """OpenAI Chat payload for the full live model matrix.

    Reasoning-heavy models such as Gemini can spend the first dozens of output
    tokens on hidden reasoning. Keep the matrix prompt small but give enough
    output budget for visible text so the test validates model invocation
    rather than failing on a too-small local cap.
    """
    payload = openai_chat_payload(model)
    payload["max_tokens"] = 512
    return payload


def openai_reasoning_payload(
    model: str,
    *,
    effort: str | None,
    stream: bool = False,
) -> dict[str, Any]:
    """OpenAI Chat request for reasoning_effort matrix coverage."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": REASONING_PROMPT}],
        "max_tokens": 4096,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if effort is not None:
        payload["reasoning_effort"] = effort
    return payload


def openai_responses_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Small deterministic OpenAI Responses request."""
    return {
        "model": model,
        "input": TEXT_PROMPT,
        "instructions": "Return only the requested word.",
        "max_output_tokens": 512,
        "stream": stream,
    }


def anthropic_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Small deterministic Anthropic Messages request."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": stream,
    }


def anthropic_compat_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Anthropic request exercising common compatibility fields."""
    payload = anthropic_payload(model, stream=stream)
    payload.pop("temperature")
    payload.update(
        {
            "system": "Return concise answers.",
            "top_p": 1,
            "stop_sequences": ["\n\nHuman:"],
            "metadata": {"user_id": "router-maestro-integration"},
        }
    )
    return payload


def gemini_payload(*, max_output_tokens: int = 16) -> dict[str, Any]:
    """Small deterministic Gemini request."""
    return {
        "contents": [{"role": "user", "parts": [{"text": TEXT_PROMPT}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_output_tokens},
    }


def gemini_compat_payload(*, max_output_tokens: int = 16) -> dict[str, Any]:
    """Gemini request exercising systemInstruction and generationConfig."""
    payload = gemini_payload(max_output_tokens=max_output_tokens)
    payload.update(
        {
            "systemInstruction": {"parts": [{"text": "Return concise answers."}]},
            "generationConfig": {
                "temperature": 0,
                "topP": 1,
                "maxOutputTokens": max_output_tokens,
                "stopSequences": ["\n\n"],
            },
        }
    )
    return payload


def openai_chat_usage_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """OpenAI Chat payload with compatibility fields and usage assertions."""
    payload = openai_chat_payload(model, stream=stream)
    payload.update(
        {
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": ["\n\n"],
            "user": "router-maestro-integration",
        }
    )
    return payload


def openai_chat_tool_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """OpenAI Chat payload that forces a function tool call."""
    payload = openai_chat_payload(model, stream=stream)
    payload["messages"] = [{"role": "user", "content": TOOL_PROMPT}]
    payload["max_tokens"] = 128
    payload["tools"] = [openai_weather_tool()]
    payload["tool_choice"] = {"type": "function", "function": {"name": "get_weather"}}
    return payload


def openai_chat_tool_required_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """OpenAI Chat payload that forces *some* tool via ``tool_choice="required"``.

    Regression guard for the Anthropic→OpenAI ``tool_choice`` translation:
    ``{"type": "any"}`` must become ``"required"`` and actually force a call.
    """
    payload = openai_chat_payload(model, stream=stream)
    payload["messages"] = [{"role": "user", "content": TOOL_PROMPT}]
    payload["max_tokens"] = 128
    payload["tools"] = [openai_weather_tool()]
    payload["tool_choice"] = "required"
    return payload


def openai_responses_tool_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """OpenAI Responses payload that forces a function tool call."""
    payload = openai_responses_payload(model, stream=stream)
    payload["input"] = TOOL_PROMPT
    payload["max_output_tokens"] = 512
    payload["tools"] = [responses_weather_tool()]
    payload["tool_choice"] = "required"
    return payload


def anthropic_reasoning_payload(
    model: str,
    *,
    budget: int | None,
    stream: bool = False,
) -> dict[str, Any]:
    """Anthropic Messages request for thinking budget matrix coverage."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": REASONING_PROMPT}],
        "max_tokens": max(2048, min(16384, (budget or 0) + 1024)),
        "stream": stream,
    }
    if budget is not None:
        payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return payload


def anthropic_effort_payload(
    model: str,
    *,
    effort: str,
    budget: int = 1024,
    stream: bool = False,
) -> dict[str, Any]:
    """Anthropic adaptive-thinking request with a deliberately conflicting budget."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": REASONING_PROMPT}],
        "max_tokens": max(4096, budget + 1024),
        "stream": stream,
        "thinking": {"type": "adaptive", "budget_tokens": budget},
        "output_config": {"effort": effort},
    }


def anthropic_tool_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Anthropic payload that forces a tool_use block."""
    payload = anthropic_payload(model, stream=stream)
    payload["messages"] = [{"role": "user", "content": TOOL_PROMPT}]
    payload["max_tokens"] = 128
    payload["tools"] = [anthropic_weather_tool()]
    payload["tool_choice"] = {"type": "tool", "name": "get_weather"}
    return payload


def anthropic_tool_choice_any_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Anthropic payload using ``tool_choice={"type":"any"}`` (force *some* tool).

    Regression guard: ``_translate_tool_choice`` must map a Pydantic
    ``AnthropicToolChoice(type="any")`` to OpenAI ``"required"`` — previously it
    fell through to ``None`` and the model was free to answer without a tool.
    """
    payload = anthropic_payload(model, stream=stream)
    payload["messages"] = [{"role": "user", "content": TOOL_PROMPT}]
    payload["max_tokens"] = 128
    payload["tools"] = [anthropic_weather_tool()]
    payload["tool_choice"] = {"type": "any"}
    return payload


def anthropic_thinking_replay_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Multi-turn Anthropic payload whose history includes a prior thinking block.

    Regression guard for the thinking-leak fix: a replayed assistant turn that
    carries a ``thinking`` block must not poison the next turn's history. The
    request must still succeed and return fresh assistant text.
    """
    return {
        "model": model,
        "max_tokens": 64,
        "temperature": 0,
        "stream": stream,
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Reply with just the number."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "INTERNAL_SCRATCHPAD_SHOULD_NOT_LEAK: 2+2=4.",
                        "signature": "sig-test",
                    },
                    {"type": "text", "text": "4"},
                ],
            },
            {"role": "user", "content": "Now reply with exactly the word pong."},
        ],
    }


def gemini_tool_payload(*, max_output_tokens: int = 16) -> dict[str, Any]:
    """Gemini payload that requires a function call."""
    payload = gemini_payload(max_output_tokens=max_output_tokens)
    payload["contents"] = [{"role": "user", "parts": [{"text": TOOL_PROMPT}]}]
    payload["tools"] = [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        }
    ]
    payload["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
    return payload


def gemini_reasoning_payload() -> dict[str, Any]:
    """Gemini request used for explicit Gemini-family model coverage."""
    return {
        "contents": [{"role": "user", "parts": [{"text": TEXT_PROMPT}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
    }


def anthropic_count_tokens_payload(model: str) -> dict[str, Any]:
    """Anthropic count_tokens payload."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
    }


def openai_weather_tool() -> dict[str, Any]:
    """OpenAI Chat function tool."""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def responses_weather_tool() -> dict[str, Any]:
    """OpenAI Responses function tool."""
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


def anthropic_weather_tool() -> dict[str, Any]:
    """Anthropic function tool."""
    return {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


def parse_sse_events(response: httpx.Response) -> list[tuple[str | None, Any]]:
    """Parse an SSE response into ``(event_name, data)`` tuples."""
    events: list[tuple[str | None, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    for line in response.iter_lines():
        if line == "":
            if data_lines:
                raw_data = "\n".join(data_lines)
                if raw_data == "[DONE]":
                    events.append((event_name, raw_data))
                else:
                    events.append((event_name, _json_or_raw(raw_data)))
                event_name = None
                data_lines = []
            continue
        if line.startswith("event: "):
            event_name = line[len("event: ") :]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: ") :])

    if data_lines:
        raw_data = "\n".join(data_lines)
        events.append((event_name, _json_or_raw(raw_data)))
    return events


def assert_text_response(text: str | None) -> None:
    """Assert the live backend returned visible assistant text."""
    assert isinstance(text, str)
    assert text.strip()


def assert_http_success(response: httpx.Response) -> None:
    """Assert an HTTP response succeeded, preserving body context on failure."""
    if response.status_code == 200:
        return
    if not response.is_stream_consumed:
        response.read()
    raise AssertionError(f"HTTP {response.status_code}: {response.text}")


def bare_model(model: str) -> str:
    """Return the unqualified model id for display labels and family checks."""
    return model.split("/", 1)[1] if "/" in model else model


def gemini_model_path(model: str) -> str:
    """Preserve the provider-qualified public ID in Router-Maestro Gemini paths."""
    return model


class LiveHttpEndpoint(StrEnum):
    """HTTP invocation surface selected from a live catalog profile."""

    CHAT = "chat"
    RESPONSES = "responses"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


def select_live_http_endpoint(
    model: str,
    catalog: dict[str, ModelInfo],
) -> LiveHttpEndpoint:
    """Select an HTTP endpoint without inferring support from a negative flag.

    Only raw ``supported_endpoints`` proves an HTTP contract; derived operation
    capabilities can contain model-name heuristics. Only exact HTTP paths count:
    a websocket Responses endpoint does not imply HTTP Responses. If both HTTP
    APIs are advertised, Responses wins so GPT-5-family models use the same
    bridge path as the rest of the integration harness.
    """
    info = catalog.get(model)
    if info is None:
        return LiveHttpEndpoint.UNSUPPORTED

    if info.supported_endpoints is None:
        return LiveHttpEndpoint.UNKNOWN

    endpoints = set(info.supported_endpoints)
    if "/responses" in endpoints:
        return LiveHttpEndpoint.RESPONSES
    if "/chat/completions" in endpoints:
        return LiveHttpEndpoint.CHAT
    return LiveHttpEndpoint.UNSUPPORTED


def is_responses_eligible_model(
    model: str,
    catalog: dict[str, ModelInfo] | None = None,
) -> bool:
    """Whether live metadata selects the Responses endpoint for this model."""
    if catalog is None:
        return bare_model(model) in RESPONSES_ELIGIBLE_MODELS
    return select_live_http_endpoint(model, catalog) is LiveHttpEndpoint.RESPONSES


def model_supports_reasoning(model: ModelInfo) -> bool:
    """Whether the live catalog explicitly advertises a reasoning control."""
    return bool(
        model.reasoning_effort_values
        or model.supports_thinking
        or model.feature_capabilities.get("reasoning") is True
    )


def model_explicitly_rejects_reasoning(model: ModelInfo) -> bool:
    """Whether the catalog explicitly advertises no reasoning control."""
    return (
        model.reasoning_effort_values == [] or model.feature_capabilities.get("reasoning") is False
    )


def event_payloads(events: list[tuple[str | None, Any]]) -> list[Any]:
    """Return parsed data payloads from SSE events, excluding ``[DONE]``."""
    return [payload for _name, payload in events if payload != "[DONE]"]


def assert_tool_call_name(tool_call: dict[str, Any], expected_name: str) -> None:
    """Assert a tool-call object names the expected function."""
    assert tool_call.get("type", "function") == "function"
    function = tool_call.get("function", {})
    assert function.get("name") == expected_name
    assert function.get("arguments") is not None


def assert_response_has_function_call(data: dict[str, Any], expected_name: str) -> None:
    """Assert an OpenAI Responses payload contains a function_call item."""
    calls = [item for item in data.get("output", []) if item.get("type") == "function_call"]
    assert calls, data
    assert any(call.get("name") == expected_name for call in calls)


def assert_anthropic_has_tool_use(data: dict[str, Any], expected_name: str) -> None:
    """Assert an Anthropic message contains a tool_use block."""
    blocks = [block for block in data.get("content", []) if block.get("type") == "tool_use"]
    assert blocks, data
    assert any(block.get("name") == expected_name for block in blocks)


def assert_gemini_has_function_call(data: dict[str, Any], expected_name: str) -> None:
    """Assert a Gemini response contains a functionCall part."""
    calls = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            if "functionCall" in part:
                calls.append(part["functionCall"])
    assert calls, data
    assert any(call.get("name") == expected_name for call in calls)


def assert_openai_usage(usage: dict[str, Any] | None) -> None:
    """Assert OpenAI chat usage has positive total tokens."""
    assert isinstance(usage, dict)
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] >= 0
    assert usage["total_tokens"] >= usage["prompt_tokens"]


def assert_responses_usage(usage: dict[str, Any] | None) -> None:
    """Assert OpenAI Responses usage has positive total tokens."""
    assert isinstance(usage, dict)
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] >= 0
    assert usage["total_tokens"] >= usage["input_tokens"]


def assert_anthropic_usage(usage: dict[str, Any] | None) -> None:
    """Assert Anthropic usage has positive input tokens."""
    assert isinstance(usage, dict)
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] >= 0


def assert_gemini_usage(usage: dict[str, Any] | None) -> None:
    """Assert Gemini usageMetadata has positive total tokens."""
    assert isinstance(usage, dict)
    assert usage["promptTokenCount"] > 0
    assert usage["totalTokenCount"] >= usage["promptTokenCount"]


def _json_or_raw(raw_data: str) -> Any:
    try:
        import json

        return json.loads(raw_data)
    except ValueError:
        return raw_data


def _require_github_copilot_auth() -> None:
    manager = AuthManager()
    if not manager.is_authenticated(COPILOT_PROVIDER):
        pytest.skip(
            "GitHub Copilot auth is not configured. Run "
            f"`uv run router-maestro auth login {COPILOT_PROVIDER}` first."
        )


def _provider_model_id(provider: str, model_id: str) -> str:
    return ModelRef.from_catalog_id(provider, model_id).qualified_id


def _prioritize_models(models: list[str]) -> list[str]:
    preferred_prefixes = (
        "github-copilot/gpt-4o-mini",
        "github-copilot/gpt-4o",
        "github-copilot/gpt-5.4-mini",
        "github-copilot/gpt-5.4",
        "github-copilot/gpt-5.3-codex",
        "github-copilot/claude-haiku",
        "github-copilot/claude-sonnet",
        "github-copilot/claude-opus",
    )
    selected: list[str] = []
    for prefix in preferred_prefixes:
        selected.extend(
            model for model in models if model.startswith(prefix) and model not in selected
        )
    selected.extend(model for model in models if model not in selected)
    return selected


def _required_reasoning_subset(
    models: list[str],
    *,
    predicate,
    description: str,
) -> list[str]:
    selected = [model for model in _prioritize_models(models) if predicate(model)]
    if not selected:
        pytest.skip(f"No {description} are available from GitHub Copilot")

    max_models = _int_env("RM_INTEGRATION_MAX_REASONING_MODELS", DEFAULT_MAX_REASONING_MATRIX)
    if max_models > 0:
        selected = selected[:max_models]
    return selected


def _is_anthropic_thinking_model(model: str) -> bool:
    bare = bare_model(model).lower()
    if not bare.startswith("claude-"):
        return False
    return not bare.endswith(("-high", "-xhigh", "-1m-internal"))


def _is_openai_chat_reasoning_model(model: str) -> bool:
    bare = bare_model(model).lower()
    if bare in RESPONSES_ONLY_CHAT_MODELS:
        return False
    return (
        bare.startswith("gpt-5")
        or bare.startswith("o1")
        or bare.startswith("o3")
        or bare.startswith("o4")
        or bare.startswith("claude-opus-4.7")
        or bare.startswith("claude-opus-4.6")
        or bare.startswith("claude-sonnet-4.6")
    )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        pytest.fail(f"{name} must be an integer, got {raw!r}")


def _first_available(models: list[str], preferred: tuple[str, ...]) -> str | None:
    available = set(models)
    return next((model for model in preferred if model in available), None)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    with httpx.Client(timeout=2.0) as startup_client:
        while time.time() < deadline:
            if process.poll() is not None:
                output = _read_process_output(process)
                pytest.fail(f"Router-Maestro server exited during startup:\n{output}")
            try:
                response = startup_client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_error = exc
            time.sleep(0.25)

    output = _read_process_output(process)
    pytest.fail(f"Router-Maestro server did not become healthy: {last_error}\n{output}")


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _read_process_output(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ""
    try:
        return process.stdout.read() or ""
    except Exception:
        return ""
