"""GitHub Copilot provider implementation."""

import asyncio
import contextlib
import json
import re
import time
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any, NoReturn
from uuid import uuid4

import httpx

from router_maestro.auth import AuthManager, AuthType
from router_maestro.auth.github_oauth import (
    GitHubOAuthError,
    get_copilot_token,
)
from router_maestro.auth.storage import OAuthCredential
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
    resolve_terminal_outcome,
)
from router_maestro.providers.tool_parsing import recover_tool_calls_from_content
from router_maestro.routing.capabilities import Feature, Operation, ProviderCapabilities
from router_maestro.utils import get_logger
from router_maestro.utils.cache import TTLCache
from router_maestro.utils.reasoning import (
    VALID_EFFORTS,
    budget_to_effort,
    downgrade_for_upstream,
    pick_closest_effort,
)
from router_maestro.utils.responses_bridge import is_model_responses_eligible

logger = get_logger("providers.copilot")

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_CHAT_PATH = "/chat/completions"
COPILOT_MODELS_PATH = "/models"
COPILOT_RESPONSES_PATH = "/responses"

# Upstream /responses events we intentionally don't consume because the route
# (server/routes/responses.py) synthesizes its own equivalents from the deltas
# we DO consume. Filtering them keeps the ``unknown_event_counts`` warning
# focused on event types that genuinely need our attention.
_BENIGN_UPSTREAM_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
        # Reasoning ``part`` events are pure structure envelopes (no text
        # payload — that arrives via ``reasoning_summary_text.delta``). The
        # route synthesizes its own added/done events from the deltas we DO
        # consume, mirroring how ``content_part.*`` is handled for messages.
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
    }
)
_BENIGN_DONE_ITEM_TYPES = frozenset({"message"})
_COPILOT_UNSUPPORTED_OPERATION_CODE = "unsupported_api_for_model"
_MAX_COPILOT_ERROR_BODY_BYTES = 64 * 1024

# Copilot catalogs may advertise provider-owned sentinels alongside public
# reasoning-effort tiers. Preserve known sentinels as capability metadata, but
# do not add them to the client request domain in ``utils.reasoning``.
_COPILOT_REASONING_EFFORT_SENTINELS = frozenset({"none"})
_COPILOT_CATALOG_REASONING_EFFORT_VALUES = frozenset(VALID_EFFORTS).union(
    _COPILOT_REASONING_EFFORT_SENTINELS
)


def _responses_terminal_outcome(response: Any) -> TerminalOutcome:
    """Preserve one native Responses terminal payload without chat mapping."""

    def protocol_error(message: str) -> TerminalOutcome:
        return TerminalOutcome(
            transport=TransportTermination.EXCEPTION,
            response_status=ResponseStatus.FAILED,
            error=TerminalError(code="upstream_protocol_error", message=message),
        )

    if not isinstance(response, dict):
        return protocol_error("Copilot Responses terminal response must be an object")

    raw_status = response.get("status")
    if not isinstance(raw_status, str):
        return protocol_error("Copilot Responses status must be a string terminal value")
    try:
        status = ResponseStatus(raw_status)
    except ValueError:
        return protocol_error("Copilot Responses status is not a recognized terminal value")

    raw_error = response.get("error")
    if raw_error is not None and not isinstance(raw_error, dict):
        return protocol_error("Copilot Responses error must be an object or null")
    error = None
    if isinstance(raw_error, dict):
        code = raw_error.get("code")
        message = raw_error.get("message")
        if code is not None and not isinstance(code, str):
            return protocol_error("Copilot Responses error code must be a string or null")
        if message is not None and not isinstance(message, str):
            return protocol_error("Copilot Responses error message must be a string or null")
        error = TerminalError(
            code=code or "upstream_error",
            message=message or "Upstream response failed",
        )

    details = response.get("incomplete_details")
    if details is not None and not isinstance(details, dict):
        return protocol_error("Copilot Responses incomplete_details must be an object or null")
    if isinstance(details, dict):
        reason = details.get("reason")
        if reason is not None and not isinstance(reason, str):
            return protocol_error("Copilot Responses incomplete reason must be a string or null")
    outcome = TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=status,
        incomplete_details=details,
        error=error,
    )
    return resolve_terminal_outcome(outcome, None) or protocol_error(
        "Copilot Responses terminal payload has no outcome"
    )


_CLAUDE_VERSION_PATTERN = re.compile(
    r"^claude-(?P<family>opus|sonnet|haiku)-(?P<major>\d+)"
    r"(?:(?:\.(?P<minor_dot>\d+))|(?:-(?P<minor_dash>\d{1,2})(?=-|\[|$)))?"
    r"(?:-|\[|$)"
)


def _claude_family_version(bare_lower: str) -> tuple[str, int, int] | None:
    """Parse the family and numeric generation without consuming dated suffixes."""
    match = _CLAUDE_VERSION_PATTERN.match(bare_lower)
    if match is None:
        return None
    return (
        match.group("family"),
        int(match.group("major")),
        int(match.group("minor_dot") or match.group("minor_dash") or 0),
    )


def _known_claude_reasoning_support(bare_lower: str) -> bool | None:
    """Return static support only for Claude versions verified on Copilot.

    Per the Copilot model picker (vscode-copilot-chat), the 4.6+ generations
    accept ``reasoning_effort``. Older models (sonnet-4, sonnet-4.5,
    opus-4.5, haiku-4.5) have no reasoning control surface.
    """
    parsed = _claude_family_version(bare_lower)
    if parsed is None:
        return None
    family, major, minor = parsed
    if major != 4:
        return None
    if family == "haiku" and minor <= 5:
        return False
    if minor < 6:
        return False
    if family == "opus" and minor in {6, 7, 8}:
        return True
    if family == "sonnet" and minor == 6:
        return True
    return None


def _claude_supports_reasoning(bare_lower: str) -> bool:
    """Whether static compatibility data confirms Claude reasoning support."""
    return _known_claude_reasoning_support(bare_lower) is True


def _known_reasoning_support(model: str) -> bool | None:
    """Return static reasoning support for known Copilot model families."""
    bare = model.split("/", 1)[1] if "/" in model else model
    bare_lower = bare.lower()
    claude_support = _known_claude_reasoning_support(bare_lower)
    if claude_support is not None:
        return claude_support
    if bare_lower.startswith(("gpt-5", "o1", "o3", "o4")):
        return True
    if bare_lower.startswith(("gpt-4", "gemini-2.5")):
        return False
    return None


def normalize_supported_endpoints(model: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Preserve missing versus explicit Copilot endpoint catalog state."""
    if "supported_endpoints" not in model:
        return None
    supported_endpoints = model.get("supported_endpoints")
    if not isinstance(supported_endpoints, (list, tuple)):
        raise TypeError("Copilot model supported_endpoints must be a list or tuple")
    if not all(isinstance(endpoint, str) for endpoint in supported_endpoints):
        raise TypeError("Copilot model supported_endpoints entries must be strings")
    return tuple(supported_endpoints)


def _normalize_catalog_boolean(supports: Mapping[str, Any], key: str) -> bool | None:
    """Preserve a missing capability while rejecting malformed present values."""
    if key not in supports:
        return None
    value = supports[key]
    if not isinstance(value, bool):
        raise TypeError(f"Copilot model capability {key} must be a boolean")
    return value


def _normalize_reasoning_effort_values(supports: Mapping[str, Any]) -> list[str] | None:
    """Normalize the catalog reasoning allowlist without silently dropping bad entries."""
    if "reasoning_effort" not in supports:
        return None
    raw = supports["reasoning_effort"]
    if isinstance(raw, dict):
        if "values" not in raw:
            raise TypeError("Copilot reasoning_effort object must contain values")
        raw = raw["values"]
    if not isinstance(raw, (list, tuple)):
        raise TypeError("Copilot reasoning_effort must be a list, tuple, or values object")

    values: list[str] = []
    for item in raw:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict) and set(item) == {"value"} and isinstance(item["value"], str):
            value = item["value"]
        else:
            raise TypeError("Copilot reasoning_effort entries must be strings or value objects")
        if not value.strip() or value not in _COPILOT_CATALOG_REASONING_EFFORT_VALUES:
            raise ValueError("Copilot reasoning_effort entry must be a supported non-empty tier")
        if value not in values:
            values.append(value)
    return values


def _normalize_catalog_limit(limits: Mapping[str, Any], key: str) -> int | None:
    """Preserve a missing limit while rejecting invalid present values."""
    if key not in limits:
        return None
    value = limits[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"Copilot model capability limit {key} must be a positive integer")
    return value


def copilot_operation_capabilities(model: Mapping[str, Any]) -> dict[str, bool]:
    """Derive operation support, preferring the live catalog endpoint contract."""
    model_id = str(model.get("id", ""))
    bare_model_id = model_id.split("/", 1)[1] if "/" in model_id else model_id
    supported_endpoints = normalize_supported_endpoints(model)
    if supported_endpoints is not None:
        endpoints = set(supported_endpoints)
        chat = "/chat/completions" in endpoints
        responses = "/responses" in endpoints
        return {
            Operation.CHAT: chat,
            Operation.CHAT_STREAM: chat,
            Operation.RESPONSES: responses,
            Operation.RESPONSES_STREAM: responses,
            Operation.NATIVE_ANTHROPIC: any(
                endpoint.endswith("/messages") for endpoint in endpoints
            ),
        }

    operations: dict[str, bool] = {
        Operation.CHAT: True,
        Operation.CHAT_STREAM: True,
        Operation.NATIVE_ANTHROPIC: bare_model_id.lower().startswith("claude-"),
    }
    if is_model_responses_eligible(bare_model_id):
        operations[Operation.RESPONSES] = True
        operations[Operation.RESPONSES_STREAM] = True
    return operations


def apply_copilot_chat_reasoning(
    payload: dict,
    model: str,
    thinking_budget: int | None,
    reasoning_effort: str | None,
    catalog_effort_values: list[str] | None = None,
) -> None:
    """Inject reasoning fields into a Copilot ``/chat/completions`` payload.

    When ``catalog_effort_values`` is provided (from the model's
    ``capabilities.supports.reasoning_effort`` advertisement), it is the
    authoritative allowlist — we map the desired effort onto the catalog's
    enum and send it. This means we automatically pick up new tiers (e.g.
    if Copilot opens ``high`` on opus-4.7) without a code change.

    When the catalog says nothing (``None``), we fall back to the hardcoded
    per-family heuristic:

    * ``claude-opus-4.6+`` / ``claude-sonnet-4.6`` accept ``low``/``medium``/``high``
      (and tolerate ``max`` being downgraded into the same set).
    * Older Claudes (4.5 / sonnet-4 / haiku) take no reasoning field.
    * ``gpt-5*`` / ``o1`` / ``o3`` / ``o4`` accept ``low``/``medium``/``high``/``xhigh``
      (``max`` is downgraded to ``xhigh``).
    * ``gpt-4*``, ``gemini-*`` take no reasoning field.

    For ``gpt-5.4*`` the gateway also requires ``max_completion_tokens`` instead
    of ``max_tokens``; this function performs that rewrite when present.
    """
    bare = model.split("/", 1)[1] if "/" in model else model
    bare_lower = bare.lower()

    known_reasoning_support = _known_reasoning_support(model)

    def unsupported_reasoning() -> NoReturn:
        parameter = "reasoning_effort" if reasoning_effort is not None else "thinking_budget"
        raise RequestOptionError(
            "GitHub Copilot does not support the requested reasoning option for this model",
            provider="github-copilot",
            model=model,
            parameter=parameter,
        )

    # Catalog-driven path: trust whatever Copilot advertises.
    if catalog_effort_values is not None:
        if not catalog_effort_values:
            if reasoning_effort is not None or thinking_budget is not None:
                unsupported_reasoning()
        else:
            desired = reasoning_effort or budget_to_effort(thinking_budget)
            if desired is None and thinking_budget is not None:
                unsupported_reasoning()
            if desired is not None:
                picked = pick_closest_effort(desired, catalog_effort_values)
                if picked is None:
                    raise RequestOptionError(
                        "GitHub Copilot has no reasoning tier at or below the requested tier",
                        provider="github-copilot",
                        model=model,
                        parameter=(
                            "reasoning_effort"
                            if reasoning_effort is not None
                            else "thinking_budget"
                        ),
                    )
                payload["reasoning_effort"] = picked
        if bare_lower.startswith("gpt-5.4") and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        return

    # Hardcoded fallback when the catalog hasn't been fetched yet.
    if known_reasoning_support is False:
        if reasoning_effort is not None or thinking_budget is not None:
            unsupported_reasoning()
    elif known_reasoning_support is True or known_reasoning_support is None:
        effort = reasoning_effort or budget_to_effort(thinking_budget)
        if bare_lower.startswith("claude-") and effort in ("xhigh", "max"):
            effort = "high"
        elif effort == "max":
            effort = "xhigh"
        elif known_reasoning_support is None and effort == "xhigh":
            effort = "high"
        if effort is None and thinking_budget is not None:
            unsupported_reasoning()
        if effort in ("minimal", "low", "medium", "high", "xhigh"):
            payload["reasoning_effort"] = effort

    if bare_lower.startswith("gpt-5.4") and "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")


# Model cache TTL in seconds (5 minutes)
MODELS_CACHE_TTL = 300

# HTTP statuses that may indicate a stale/invalid Copilot token.
_AUTH_RETRY_STATUSES = frozenset({401, 403})


def _thinking_requested(request: ChatRequest) -> bool:
    """Whether the client opted into reasoning passthrough.

    We only surface upstream chain-of-thought to clients that explicitly asked
    for it. Reasoning traces can leak prompt fragments, hidden instructions,
    and tool-planning state, so emitting them by default would be a
    sensitive-data exposure surface.
    """
    return request.thinking_type in ("enabled", "adaptive")


def _extract_reasoning_from_chunk(part: dict | None) -> tuple[str, str | None]:
    """Pull reasoning text/signature out of a Copilot message or delta.

    Mirrors vscode-copilot-chat's ``extractThinkingDeltaFromChoice``: Copilot
    streams reasoning under several legacy field names depending on the
    upstream model family.

    Returns ``(text, signature)`` where either may be empty/None.
    """
    if not part:
        return "", None

    text = ""
    for key in ("reasoning_text", "cot_summary", "thinking"):
        val = part.get(key)
        if isinstance(val, str) and val:
            text = val
            break
        if isinstance(val, dict):
            inner = val.get("text") or val.get("content")
            if isinstance(inner, str) and inner:
                text = inner
                break

    sig: str | None = None
    for key in ("reasoning_opaque", "cot_id", "signature"):
        val = part.get(key)
        if isinstance(val, str) and val:
            sig = val
            break

    return text, sig


class CopilotProvider(BaseProvider):
    """GitHub Copilot provider."""

    name = "github-copilot"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            operations=frozenset(Operation),
            operation_bridges={
                Operation.CHAT: Operation.RESPONSES,
                Operation.CHAT_STREAM: Operation.RESPONSES_STREAM,
            },
        )

    # Recycle the HTTP/2 client after this many seconds to avoid GOAWAY races
    _CLIENT_MAX_AGE = 300  # 5 minutes

    def __init__(self) -> None:
        self.auth_manager = AuthManager()
        self._cached_token: str | None = None
        self._token_expires: int = 0
        self._api_base = COPILOT_BASE_URL
        # Model cache
        self._models_ttl_cache: TTLCache[list[ModelInfo]] = TTLCache(MODELS_CACHE_TTL)
        # Reusable HTTP client
        self._client: httpx.AsyncClient | None = None
        self._client_created_at: float = 0.0
        # Serializes token refresh so concurrent requests don't all refresh at
        # once (avoids redundant GitHub calls and overlapping auth.json writes).
        self._token_refresh_lock = asyncio.Lock()

    def is_authenticated(self) -> bool:
        """Check if authenticated with GitHub Copilot."""
        cred = self.auth_manager.get_credential("github-copilot")
        return cred is not None and cred.type == AuthType.OAUTH

    async def ensure_token(self, force: bool = False) -> None:
        """Ensure we have a valid Copilot token, refreshing if needed.

        Args:
            force: Re-mint the Copilot token even if the locally-cached one
                looks unexpired. Used by the 401/403 retry path when the
                upstream rejects a token our clock still considers valid.
        """
        cred = self.auth_manager.get_credential("github-copilot")
        if not cred or not isinstance(cred, OAuthCredential):
            logger.error("Not authenticated with GitHub Copilot")
            raise ProviderError(
                "Not authenticated with GitHub Copilot",
                status_code=401,
                kind=ProviderFailureKind.AUTHENTICATION,
                provider=self.name,
            )

        if cred.api_endpoint:
            self._api_base = cred.api_endpoint

        current_time = int(time.time())

        # Check if we need to refresh (token expired or will expire soon)
        if not force and self._cached_token and self._token_expires > current_time + 60:
            return  # Token still valid

        # Snapshot the token we're trying to replace so that, after we win the
        # lock, we can tell whether another coroutine already re-minted it.
        token_before_lock = self._cached_token

        async with self._token_refresh_lock:
            # Double-checked: another coroutine may have refreshed while we waited.
            current_time = int(time.time())
            if not force and self._cached_token and self._token_expires > current_time + 60:
                return
            # Force path: if another coroutine already swapped in a fresh token
            # while we waited for the lock, reuse it instead of re-minting. This
            # collapses an N-request 401 storm into a single mint.
            if force and self._cached_token and self._cached_token != token_before_lock:
                return

            # Re-read the credential under the lock: another coroutine may have
            # persisted a fresh Copilot token while we waited; use the latest.
            cred = self.auth_manager.get_credential("github-copilot")
            if not cred or not isinstance(cred, OAuthCredential):
                logger.error("Not authenticated with GitHub Copilot")
                raise ProviderError(
                    "Not authenticated with GitHub Copilot",
                    status_code=401,
                    kind=ProviderFailureKind.AUTHENTICATION,
                    provider=self.name,
                )

            logger.debug("Refreshing Copilot token")
            # Use a fresh short-lived client with a SHORT timeout for token
            # refresh — if GitHub is slow, fail fast rather than blocking all
            # requests behind the lock for minutes.
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
                ) as client:
                    copilot_token = await get_copilot_token(client, cred.refresh)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    logger.error(
                        "GitHub Copilot authentication failed (%s)", e.response.status_code
                    )
                    # Keep retryable=True so the router can still fall back to
                    # other configured providers; the message tells the user how
                    # to recover if Copilot is their only provider.
                    raise ProviderError(
                        "GitHub Copilot authentication expired. If Copilot is your only "
                        "provider, re-authenticate: `router-maestro auth login github-copilot`.",
                        status_code=401,
                        retryable=True,
                        kind=ProviderFailureKind.AUTHENTICATION,
                        upstream_status_code=e.response.status_code,
                        provider=self.name,
                        cause=e,
                    ) from e
                logger.error(
                    "Failed to refresh Copilot token: status=%d",
                    e.response.status_code,
                )
                kind = (
                    ProviderFailureKind.RATE_LIMIT
                    if e.response.status_code in (429, 529)
                    else ProviderFailureKind.UPSTREAM_STATUS
                )
                raise ProviderError(
                    "Failed to refresh Copilot token",
                    status_code=e.response.status_code,
                    retryable=e.response.status_code == 429 or e.response.status_code >= 500,
                    kind=kind,
                    upstream_status_code=e.response.status_code,
                    provider=self.name,
                    cause=e,
                ) from e
            except (httpx.HTTPError, GitHubOAuthError) as e:
                logger.error("Failed to refresh Copilot token (%s)", type(e).__name__)
                raise ProviderError(
                    "Failed to refresh Copilot token",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.TRANSPORT,
                    provider=self.name,
                    cause=e,
                ) from e

            self._cached_token = copilot_token.token
            self._token_expires = copilot_token.expires_at
            self._api_base = copilot_token.api_endpoint or self._api_base or COPILOT_BASE_URL

            # Update stored credential with the new access token (immutable pattern).
            await self._persist_credential(
                OAuthCredential(
                    refresh=cred.refresh,
                    access=copilot_token.token,
                    expires=copilot_token.expires_at,
                    api_endpoint=copilot_token.api_endpoint or cred.api_endpoint,
                )
            )
            logger.debug("Copilot token refreshed, expires at %d", copilot_token.expires_at)

    async def _persist_credential(self, cred: OAuthCredential) -> None:
        """Store and flush a credential to auth.json off the event loop."""
        self.auth_manager.storage.set("github-copilot", cred)
        await asyncio.to_thread(self.auth_manager.save)

    async def _refresh_for_auth_status(self, path: str, status_code: int) -> bool:
        """Decide whether an auth-status response warrants a forced re-mint.

        Shared by every Copilot call site (chat, responses, models — streaming
        and not) so the retry policy lives in exactly one place. Only 401/403
        trigger a refresh; 429/5xx pass through to the caller. A 403 we can't
        heal (policy/entitlement/quota) just costs one wasted re-mint+retry,
        which is cheaper than a token-age heuristic that would blind us to a
        genuinely revoked token in its first seconds. Returns True iff a forced
        refresh was performed and the caller should retry once.
        """
        if status_code not in _AUTH_RETRY_STATUSES:
            return False
        logger.info(
            "Copilot %s returned %d; forcing token refresh and retrying",
            path,
            status_code,
        )
        await self.ensure_token(force=True)
        return True

    def _raise_auth_failure(
        self, path: str, status_code: int, *, model: str | None = None
    ) -> NoReturn:
        """Raise a retryable auth error for a 401/403 that survived the retry.

        Centralizes the policy so every call site (chat, responses, models —
        streaming and not) treats a post-retry auth rejection identically:
        retryable so the router can fall back to other configured providers,
        with a message that tells the user how to recover if Copilot is their
        only provider.
        """
        logger.error("Copilot %s still returned %d after token refresh", path, status_code)
        raise ProviderError(
            f"Copilot authentication rejected ({status_code}) after refresh. If Copilot is "
            "your only provider, re-authenticate: `router-maestro auth login github-copilot`.",
            status_code=status_code,
            retryable=True,
            kind=ProviderFailureKind.AUTHENTICATION,
            upstream_status_code=status_code,
            provider=self.name,
            model=model,
        )

    def _raise_unsupported_operation_if_applicable(
        self,
        status_code: int,
        body: bytes,
        *,
        model: str,
        cause: BaseException | None = None,
    ) -> None:
        """Classify Copilot's exact, bounded unsupported-API error payload."""
        if status_code != 400 or len(body) > _MAX_COPILOT_ERROR_BODY_BYTES:
            return
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        error = payload.get("error")
        if not isinstance(error, dict):
            return
        if error.get("code") != _COPILOT_UNSUPPORTED_OPERATION_CODE:
            return
        raise ProviderError(
            "Copilot does not support this API operation for the requested model",
            status_code=400,
            retryable=False,
            kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
            upstream_status_code=400,
            provider=self.name,
            model=model,
            cause=cause,
        ) from cause

    async def _send_with_auth_retry(
        self,
        method: str,
        path: str,
        *,
        client: httpx.AsyncClient | None = None,
        json: dict | None = None,
        headers_kwargs: dict | None = None,
        timeout: Any = TIMEOUT_NON_STREAMING,
        model: str | None = None,
    ) -> httpx.Response:
        """Send a non-streaming Copilot request, force-refreshing once on 401/403.

        Shared by chat/responses (POST, shared pool) and models (GET, fresh
        short-lived ``client``). Auth rejections on a token our clock still
        considers valid self-heal by re-minting and retrying once; a 401/403
        that survives the retry raises a retryable auth error so the router can
        fall back. 429/5xx pass through unchanged for the caller to handle.

        Also handles connection-level errors (HTTP/2 GOAWAY, pool timeouts) by
        recycling the HTTP client and retrying once.
        """
        client = client or self._get_client()
        headers_kwargs = headers_kwargs or {}
        for attempt in range(2):
            try:
                if method == "GET":
                    response = await client.get(
                        self._url(path),
                        headers=self._get_headers(**headers_kwargs),
                        timeout=timeout,
                    )
                else:
                    response = await client.post(
                        self._url(path),
                        json=json,
                        headers=self._get_headers(**headers_kwargs),
                        timeout=timeout,
                    )
            except (httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.ConnectError) as e:
                if attempt == 0:
                    logger.warning(
                        "Connection error on %s, recycling client (%s)", path, type(e).__name__
                    )
                    await self._recycle_client()
                    client = self._get_client()
                    continue
                raise ProviderError(
                    f"Connection failed after retry ({type(e).__name__})",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.TRANSPORT,
                    provider=self.name,
                    model=model,
                    cause=e,
                ) from e

            if attempt == 0 and await self._refresh_for_auth_status(path, response.status_code):
                continue
            if response.status_code in _AUTH_RETRY_STATUSES:
                self._raise_auth_failure(path, response.status_code, model=model)
            return response
        # Unreachable: attempt 1 always returns or raises above.
        return response

    async def _recycle_client(self) -> None:
        """Close and discard the current HTTP client so the next call creates a fresh one."""
        if self._client and not self._client.is_closed:
            try:
                await self._client.aclose()
            except Exception:
                pass
        self._client = None

    @contextlib.asynccontextmanager
    async def _stream_with_auth_retry(
        self,
        path: str,
        *,
        json: dict,
        headers_kwargs: dict,
        model: str | None = None,
    ) -> "AsyncIterator[httpx.Response]":
        """Open a Copilot stream, force-refreshing and retrying once on 401/403.

        Mirrors ``_send_with_auth_retry`` for streaming. The 401/403 is detected
        from the response status *before* any chunk is yielded, so the retry
        never replays partial output. A 401/403 that survives the retry raises a
        retryable auth error (consistent with the non-streaming path); other
        statuses are yielded for the caller to pre-read and raise on.

        Also handles connection-level errors by recycling the HTTP client.
        """
        client = self._get_client()
        for attempt in range(2):
            try:
                cm = client.stream(
                    "POST", self._url(path), json=json, headers=self._get_headers(**headers_kwargs)
                )
                response = await cm.__aenter__()
            except (httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.ConnectError) as e:
                if attempt == 0:
                    logger.warning(
                        "Stream connection error on %s, recycling client (%s)",
                        path,
                        type(e).__name__,
                    )
                    await self._recycle_client()
                    client = self._get_client()
                    continue
                raise ProviderError(
                    f"Stream connection failed after retry ({type(e).__name__})",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.TRANSPORT,
                    provider=self.name,
                    model=model,
                    cause=e,
                ) from e

            if attempt == 0 and response.status_code in _AUTH_RETRY_STATUSES:
                with contextlib.suppress(Exception):
                    await response.aread()
                await cm.__aexit__(None, None, None)
                if await self._refresh_for_auth_status(path, response.status_code):
                    continue
            # A 401/403 surviving the retry is an unrecoverable auth failure —
            # raise the shared retryable error rather than yielding a dead
            # response for the caller to re-read.
            if response.status_code in _AUTH_RETRY_STATUSES:
                with contextlib.suppress(Exception):
                    await response.aread()
                await cm.__aexit__(None, None, None)
                self._raise_auth_failure(path, response.status_code, model=model)
            try:
                yield response
            finally:
                await cm.__aexit__(None, None, None)
            return

    def _url(self, path: str) -> str:
        """Build a Copilot API URL from the token-advertised API base."""
        return f"{self._api_base.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def _chat_initiator(messages: list[Message] | None) -> str:
        """Infer Copilot X-Initiator for chat-completions payloads."""
        if not messages:
            return "user"
        for message in messages:
            if message.role in ("assistant", "tool"):
                return "agent"
        return "user"

    @staticmethod
    def _responses_initiator(response_input: str | list[dict[str, Any]] | None) -> str:
        """Infer Copilot X-Initiator for Responses API payloads."""
        if isinstance(response_input, str) or not response_input:
            return "user"
        for item in response_input:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if not role or (isinstance(role, str) and role.lower() == "assistant"):
                return "agent"
        return "user"

    def _get_headers(
        self,
        vision_request: bool = False,
        *,
        messages: list[Message] | None = None,
        response_input: str | list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Get headers for Copilot API requests.

        Args:
            vision_request: Whether this request contains images (vision)
        """
        if not self._cached_token:
            raise ProviderError(
                "No valid token available",
                status_code=401,
                kind=ProviderFailureKind.AUTHENTICATION,
                provider=self.name,
            )

        headers = {
            "Authorization": f"Bearer {self._cached_token}",
            "Content-Type": "application/json",
            "Editor-Version": "vscode/1.95.0",
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "Copilot-Integration-Id": "vscode-chat",
            "User-Agent": "GitHubCopilotChat/0.26.7",
            "OpenAI-Intent": "conversation-panel",
            "X-GitHub-Api-Version": "2025-04-01",
            "X-Request-Id": str(uuid4()),
            "X-Vscode-User-Agent-Library-Version": "electron-fetch",
        }
        if response_input is not None:
            headers["X-Initiator"] = self._responses_initiator(response_input)
        elif messages is not None:
            headers["X-Initiator"] = self._chat_initiator(messages)

        if vision_request:
            headers["Copilot-Vision-Request"] = "true"

        return headers

    def _catalog_effort_values(self, model: str) -> list[str] | None:
        """Look up the catalog-advertised reasoning_effort allowlist for ``model``.

        Pulls from the in-memory model cache only — never blocks the request
        on a network fetch. Returns ``None`` if the cache is cold or the model
        isn't in it, in which case ``apply_copilot_chat_reasoning`` falls back
        to the hardcoded heuristic.
        """
        cached = self._models_ttl_cache.get()
        if not cached:
            return None
        bare = model.split("/", 1)[1] if "/" in model else model
        for info in cached:
            if info.id == bare or info.id == model:
                return info.reasoning_effort_values
        return None

    def _catalog_operation_support(self, model: str, operation: Operation) -> bool | None:
        """Return catalog operation support without triggering a model fetch."""
        cached = self._models_ttl_cache.get()
        if not cached:
            return None
        bare = model.split("/", 1)[1] if "/" in model else model
        for info in cached:
            if info.id == bare or info.id == model:
                return info.operation_capabilities.get(operation.value)
        return None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create a reusable HTTP client.

        Recycles the client after _CLIENT_MAX_AGE seconds to avoid HTTP/2
        GOAWAY races from server-side connection lifetime enforcement.
        """
        now = time.time()
        needs_recycle = (
            self._client is not None
            and not self._client.is_closed
            and self._client_created_at > 0
            and now - self._client_created_at >= self._CLIENT_MAX_AGE
        )
        if needs_recycle:
            asyncio.ensure_future(self._client.aclose())
            self._client = None

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=600.0,
                    write=30.0,
                    pool=30.0,
                ),
                http2=True,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
            self._client_created_at = now
        return self._client

    async def close(self) -> None:
        """Close the HTTP client. Called during app shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _sanitize_surrogates(text: str) -> str:
        """Remove lone surrogate characters that cannot be encoded as UTF-8."""
        return text.encode("utf-8", errors="replace").decode("utf-8")

    def _sanitize_content(self, content: str | list) -> str | list:
        """Sanitize message content to remove lone surrogate characters."""
        if isinstance(content, str):
            return self._sanitize_surrogates(content)
        if isinstance(content, list):
            result = []
            for part in content:
                is_text = (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                )
                if is_text:
                    result.append({**part, "text": self._sanitize_surrogates(part["text"])})
                else:
                    result.append(part)
            return result
        return content

    def _build_messages_payload(self, request: ChatRequest) -> tuple[list[dict], bool]:
        """Build messages payload and detect if images are present.

        Args:
            request: The chat request

        Returns:
            Tuple of (messages list, has_images flag)
        """
        messages = []
        has_images = False

        for m in request.messages:
            msg: dict = {"role": m.role, "content": self._sanitize_content(m.content)}
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.refusal is not None:
                msg["refusal"] = m.refusal
            messages.append(msg)

            # Check if this message contains images (multimodal content)
            if isinstance(m.content, list):
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_images = True
                        break

        return messages, has_images

    def _responses_input_has_vision(self, value: Any, depth: int = 0) -> bool:
        """Whether a Responses API input contains an image block."""
        if depth > 32 or value is None:
            return False
        if isinstance(value, list):
            return any(self._responses_input_has_vision(item, depth + 1) for item in value)
        if not isinstance(value, dict):
            return False
        item_type = value.get("type")
        if isinstance(item_type, str) and item_type.lower() in ("input_image", "image_url"):
            return True
        if "image_url" in value:
            return True
        content = value.get("content")
        if isinstance(content, list):
            return any(self._responses_input_has_vision(item, depth + 1) for item in content)
        return False

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion via Copilot."""
        # Experimental: route GPT-5.x ChatRequests through /responses when the
        # entry route opted in. Anthropic/Gemini set use_responses_api=True
        # under the ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API flag.
        from router_maestro.utils.responses_bridge import (
            chat_request_to_responses_request,
            responses_response_to_chat_response,
            should_use_responses_for_chat,
        )

        if should_use_responses_for_chat(
            request,
            self.name,
            responses_supported=self._catalog_operation_support(
                request.model,
                Operation.RESPONSES,
            ),
        ):
            logger.info(
                "Routing chat request via /responses (experimental): model=%s",
                request.model,
            )
            responses_req = chat_request_to_responses_request(request)
            responses_resp = await self.responses_completion(responses_req)
            return responses_response_to_chat_response(responses_resp, request.model)

        await self.ensure_token()

        messages, has_images = self._build_messages_payload(request)

        payload = self._build_chat_payload(request, stream=False)

        logger.debug(
            "Copilot chat completion: model=%s thinking_budget=%s reasoning_effort=%s "
            "payload_thinking_budget=%s payload_reasoning_effort=%s",
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            payload.get("thinking_budget"),
            payload.get("reasoning_effort"),
        )
        try:
            response = await self._send_with_auth_retry(
                "POST",
                COPILOT_CHAT_PATH,
                json=payload,
                headers_kwargs={"vision_request": has_images, "messages": request.messages},
                model=request.model,
            )
            response.raise_for_status()
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("chat response must be an object")
                choices = data.get("choices")
                if not isinstance(choices, list):
                    raise TypeError("chat response choices must be a list")
                model = self._validated_response_model(data, request.model)
                usage = self._validated_token_usage(
                    data.get("usage"),
                    fields=("prompt_tokens", "completion_tokens", "total_tokens"),
                    label="chat response",
                    detail_fields={
                        "prompt_tokens_details": ("cached_tokens",),
                        "completion_tokens_details": ("reasoning_tokens",),
                    },
                )
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                self._raise_protocol_error(self.name, request.model, e)

            completion_tokens = (usage or {}).get("completion_tokens", 0)

            bare_lower = (
                request.model.split("/", 1)[1] if "/" in request.model else request.model
            ).lower()
            reasoning_capable = bare_lower.startswith("claude-") and _claude_supports_reasoning(
                bare_lower
            )

            if not choices:
                # Reasoning-capable Claude models can spend their whole budget
                # on hidden reasoning and finish without emitting visible text.
                # Copilot reports that truncation as choices=[] with usage at
                # the output cap. ``thinking_budget`` is not a numeric wire
                # limit here: it is translated to ``reasoning_effort``, so only
                # the max_tokens value actually sent upstream can prove that
                # the empty response was caused by exhaustion.
                max_tokens = request.max_tokens
                has_positive_output_cap = (
                    isinstance(max_tokens, int)
                    and not isinstance(max_tokens, bool)
                    and max_tokens > 0
                )
                thinking_budget = request.thinking_budget
                exhausted_explicit_thinking_budget = (
                    request.reasoning_effort is None
                    and request.thinking_type == "enabled"
                    and isinstance(thinking_budget, int)
                    and not isinstance(thinking_budget, bool)
                    and thinking_budget > 0
                    and completion_tokens == thinking_budget
                )
                if (
                    reasoning_capable
                    and _thinking_requested(request)
                    and has_positive_output_cap
                    and (completion_tokens >= max_tokens or exhausted_explicit_thinking_budget)
                ):
                    logger.warning(
                        "Copilot returned empty choices after exhausting a requested cap: "
                        "model=%s completion_tokens=%d max_tokens=%d thinking_budget=%s",
                        request.model,
                        completion_tokens,
                        max_tokens,
                        thinking_budget,
                    )
                    return ChatResponse(
                        content="",
                        model=model,
                        finish_reason="length",
                        usage=usage or None,
                        tool_calls=None,
                    )
                logger.error(
                    "Copilot API returned empty choices: model=%s has_usage=%s",
                    request.model,
                    "usage" in data,
                )
                self._raise_protocol_error(
                    self.name,
                    request.model,
                    ValueError("chat response choices must be a non-empty list"),
                )

            logger.debug("Copilot chat completion successful")

            # Copilot may return multiple choices: one with text content and
            # separate ones each containing a single tool_call. Merge them all.
            content = None
            refusal = None
            tool_calls = []
            finish_reason = "stop"
            thinking_text = ""
            thinking_signature: str | None = None

            try:
                parsed_choices: list[tuple[dict, dict, str | None]] = []
                for choice in choices:
                    if not isinstance(choice, dict):
                        raise TypeError("chat response choice must be an object")
                    msg = choice["message"]
                    if not isinstance(msg, dict):
                        raise TypeError("chat response message must be an object")
                    message_content = msg.get("content")
                    message_refusal = msg.get("refusal")
                    message_tool_calls = msg.get("tool_calls")
                    if message_content is not None and not isinstance(message_content, str):
                        raise TypeError("chat response content must be a string or null")
                    if "refusal" in msg and (
                        not isinstance(message_refusal, str) or not message_refusal
                    ):
                        raise TypeError("chat response refusal must be a non-empty string")
                    if message_tool_calls is not None and not isinstance(message_tool_calls, list):
                        raise TypeError("chat response tool_calls must be a list or null")
                    if message_tool_calls:
                        for tool_call in message_tool_calls:
                            if not isinstance(tool_call, dict):
                                raise TypeError("chat response tool call must be an object")
                            function = tool_call.get("function")
                            if (
                                not isinstance(tool_call.get("id"), str)
                                or not isinstance(function, dict)
                                or not isinstance(function.get("name"), str)
                                or not isinstance(function.get("arguments"), str)
                            ):
                                raise TypeError("chat response tool call is malformed")
                    choice_finish_reason = self._validated_optional_string(choice, "finish_reason")
                    parsed_choices.append((choice, msg, choice_finish_reason))
            except (KeyError, TypeError) as e:
                self._raise_protocol_error(self.name, request.model, e)

            for choice, msg, choice_finish_reason in parsed_choices:
                # Take content from the first choice that has it
                if content is None and msg.get("content"):
                    content = msg["content"]
                if refusal is None and msg.get("refusal"):
                    refusal = msg["refusal"]
                # Collect tool_calls from all choices
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
                # Use finish_reason from any choice (they should all match)
                if choice_finish_reason:
                    finish_reason = choice_finish_reason
                # Collect reasoning text/signature only if the client opted in
                if _thinking_requested(request):
                    t, sig = _extract_reasoning_from_chunk(msg)
                    if t:
                        thinking_text += t
                    if sig and thinking_signature is None:
                        thinking_signature = sig

            if (
                not content
                and not refusal
                and not tool_calls
                and not thinking_text
                and not thinking_signature
            ):
                self._raise_protocol_error(
                    self.name,
                    request.model,
                    ValueError("chat response contains no deliverable output"),
                )

            if len(choices) > 1:
                logger.info(
                    "Copilot returned %d choices: content=%s, tool_calls=%d, finish_reason=%s",
                    len(choices),
                    len(content) if content else 0,
                    len(tool_calls),
                    finish_reason,
                )

            tool_calls = tool_calls or None

            content, tool_calls = recover_tool_calls_from_content(
                content, tool_calls, finish_reason
            )
            if tool_calls and finish_reason in (None, "stop"):
                finish_reason = "tool_calls"

            return ChatResponse(
                content=content,
                model=model,
                refusal=refusal,
                finish_reason=finish_reason,
                usage=usage,
                tool_calls=tool_calls,
                thinking=thinking_text or None,
                thinking_signature=thinking_signature,
            )
        except httpx.HTTPStatusError as e:
            self._raise_unsupported_operation_if_applicable(
                e.response.status_code,
                e.response.content,
                model=request.model,
                cause=e,
            )
            self._raise_http_status_error(
                "Copilot",
                e,
                logger,
                include_body=True,
                provider=self.name,
                model=request.model,
            )
        except httpx.TimeoutException as e:
            self._raise_timeout_error("Copilot", e, logger, provider=self.name, model=request.model)
        except httpx.HTTPError as e:
            self._raise_http_error("Copilot", e, logger, provider=self.name, model=request.model)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion via Copilot."""
        from router_maestro.utils.responses_bridge import (
            chat_request_to_responses_request,
            responses_chunk_to_chat_chunk,
            should_use_responses_for_chat,
        )

        if should_use_responses_for_chat(
            request,
            self.name,
            responses_supported=self._catalog_operation_support(
                request.model,
                Operation.RESPONSES_STREAM,
            ),
        ):
            logger.info(
                "Streaming chat request via /responses (experimental): model=%s",
                request.model,
            )
            responses_req = chat_request_to_responses_request(request)
            emitted_tool_call = False
            async for resp_chunk in self.responses_completion_stream(responses_req):
                chat_chunk = responses_chunk_to_chat_chunk(resp_chunk)
                if resp_chunk.tool_call is not None:
                    emitted_tool_call = True
                terminal_outcome = resp_chunk.terminal_outcome
                can_upgrade_tool_finish = terminal_outcome is None or (
                    terminal_outcome.response_status is ResponseStatus.COMPLETED
                )
                if (
                    emitted_tool_call
                    and can_upgrade_tool_finish
                    and chat_chunk.finish_reason == "stop"
                ):
                    chat_chunk.finish_reason = "tool_calls"
                yield chat_chunk
            return

        await self.ensure_token()

        messages, has_images = self._build_messages_payload(request)

        payload = self._build_chat_payload(request, stream=True)

        logger.debug(
            "Copilot streaming chat: model=%s thinking_budget=%s reasoning_effort=%s "
            "payload_thinking_budget=%s payload_reasoning_effort=%s",
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            payload.get("thinking_budget"),
            payload.get("reasoning_effort"),
        )
        try:
            async with self._stream_with_auth_retry(
                COPILOT_CHAT_PATH,
                json=payload,
                headers_kwargs={"vision_request": has_images, "messages": request.messages},
                model=request.model,
            ) as response:
                # Streamed responses defer body reads; if the upstream returns
                # an error status, pull the body *inside* the stream context
                # so the connection is still open. Reading after the
                # `async with` exits raises StreamClosed and the helper would
                # log "Copilot stream API error: 4xx -" with no upstream
                # detail (the original symptom this fix addresses).
                if response.status_code >= 400:
                    with contextlib.suppress(Exception):
                        await response.aread()
                response.raise_for_status()

                stream_finished = False
                emitted_tool_call = False
                observed_usage: dict | None = None
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        if not isinstance(data, dict):
                            raise TypeError("stream event must be an object")
                        usage = self._validated_token_usage(
                            data.get("usage"),
                            fields=("prompt_tokens", "completion_tokens", "total_tokens"),
                            label="stream",
                            detail_fields={
                                "prompt_tokens_details": ("cached_tokens",),
                                "completion_tokens_details": ("reasoning_tokens",),
                            },
                        )
                        choices = data.get("choices")
                        if choices is not None and not isinstance(choices, list):
                            raise TypeError("stream choices must be a list")
                        if choices:
                            for choice in choices:
                                if not isinstance(choice, dict):
                                    raise TypeError("stream choice must be an object")
                                delta = choice.get("delta", {})
                                if not isinstance(delta, dict):
                                    raise TypeError("stream delta must be an object")
                                content = delta.get("content")
                                refusal = delta.get("refusal")
                                tool_calls = delta.get("tool_calls")
                                finish_reason = choice.get("finish_reason")
                                if content is not None and not isinstance(content, str):
                                    raise TypeError("stream content must be a string or null")
                                if refusal is not None and not isinstance(refusal, str):
                                    raise TypeError("stream refusal must be a string or null")
                                if tool_calls is not None and not isinstance(tool_calls, list):
                                    raise TypeError("stream tool_calls must be a list or null")
                                if tool_calls:
                                    for tool_call in tool_calls:
                                        if not isinstance(tool_call, dict):
                                            raise TypeError("stream tool call delta must be object")
                                        index = tool_call.get("index")
                                        if index is not None and (
                                            not isinstance(index, int) or isinstance(index, bool)
                                        ):
                                            raise TypeError(
                                                "stream tool call index must be integer"
                                            )
                                        for field in ("id", "type"):
                                            value = tool_call.get(field)
                                            if value is not None and not isinstance(value, str):
                                                raise TypeError(
                                                    f"stream tool call {field} must be string"
                                                )
                                        function = tool_call.get("function")
                                        if function is not None:
                                            if not isinstance(function, dict):
                                                raise TypeError(
                                                    "stream tool call function must be object"
                                                )
                                            for field in ("name", "arguments"):
                                                value = function.get(field)
                                                if value is not None and not isinstance(value, str):
                                                    raise TypeError(
                                                        "stream tool function "
                                                        f"{field} must be string"
                                                    )
                                if finish_reason is not None and not isinstance(finish_reason, str):
                                    raise TypeError("stream finish_reason must be a string or null")
                    except (json.JSONDecodeError, TypeError) as e:
                        self._raise_protocol_error(self.name, request.model, e)

                    usage_to_emit = None
                    if usage:
                        if observed_usage is not None and usage != observed_usage:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("conflicting usage in one chat stream"),
                            )
                        if observed_usage is None:
                            observed_usage = usage
                            usage_to_emit = usage

                    if stream_finished:
                        if choices != [] or not usage:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("non-usage event followed a terminal chat choice"),
                            )
                        if usage_to_emit is not None:
                            yield ChatStreamChunk(
                                content="",
                                finish_reason=None,
                                usage=usage_to_emit,
                            )
                        continue

                    if choices:
                        terminal_chunks = []
                        for choice in choices:
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            refusal = delta.get("refusal")
                            finish_reason = choice.get("finish_reason")
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                emitted_tool_call = True
                            if finish_reason == "stop" and emitted_tool_call:
                                finish_reason = "tool_calls"
                            if _thinking_requested(request):
                                thinking_text, thinking_sig = _extract_reasoning_from_chunk(delta)
                            else:
                                thinking_text, thinking_sig = "", None

                            if (
                                content
                                or refusal
                                or finish_reason
                                or usage_to_emit
                                or tool_calls
                                or thinking_text
                                or thinking_sig
                            ):
                                chunk = ChatStreamChunk(
                                    content=content,
                                    refusal=refusal or None,
                                    finish_reason=finish_reason,
                                    usage=usage_to_emit,
                                    tool_calls=tool_calls,
                                    thinking=thinking_text or None,
                                    thinking_signature=thinking_sig,
                                )
                                usage_to_emit = None
                                if finish_reason:
                                    terminal_chunks.append(chunk)
                                else:
                                    yield chunk

                            # Mark stream as finished after processing every
                            # choice in this SSE event.
                            if finish_reason:
                                stream_finished = True
                        for chunk in terminal_chunks:
                            yield chunk
                    elif usage_to_emit:
                        # Handle usage-only chunks (no choices)
                        yield ChatStreamChunk(
                            content="",
                            finish_reason=None,
                            usage=usage_to_emit,
                        )
        except httpx.HTTPStatusError as e:
            try:
                error_body = e.response.content
            except httpx.ResponseNotRead:
                error_body = b""
            self._raise_unsupported_operation_if_applicable(
                e.response.status_code,
                error_body,
                model=request.model,
                cause=e,
            )
            self._raise_http_status_error(
                "Copilot",
                e,
                logger,
                stream=True,
                include_body=True,
                provider=self.name,
                model=request.model,
            )
        except httpx.TimeoutException as e:
            self._raise_timeout_error(
                "Copilot",
                e,
                logger,
                stream=True,
                provider=self.name,
                model=request.model,
            )
        except httpx.HTTPError as e:
            logger.error(
                "Copilot stream HTTP error: type=%s model=%s vision=%s url=%s",
                type(e).__name__,
                request.model,
                has_images,
                self._url(COPILOT_CHAT_PATH),
            )
            self._raise_http_error(
                "Copilot",
                e,
                logger,
                stream=True,
                provider=self.name,
                model=request.model,
            )

    async def list_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """List available Copilot models from API with caching.

        Args:
            force_refresh: Force refresh the cache

        Returns:
            List of available models
        """
        # Return cached models if valid
        if not force_refresh:
            cached = self._models_ttl_cache.get()
            if cached is not None:
                logger.debug("Using cached Copilot models (%d models)", len(cached))
                return deepcopy(cached)

        logger.debug("Fetching Copilot models from API")
        # Use a fresh short-lived client to avoid blocking on the shared
        # streaming HTTP/2 connection pool
        try:
            # Mint inside the try so a dead-token failure degrades to the stale
            # cache (below) instead of hard-failing the models endpoint.
            await self.ensure_token()
            async with httpx.AsyncClient(timeout=TIMEOUT_NON_STREAMING) as client:
                response = await self._send_with_auth_retry(
                    "GET", COPILOT_MODELS_PATH, client=client, model=None
                )
                response.raise_for_status()
                try:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise TypeError("Copilot model catalog must be an object")
                    if "data" not in data or not isinstance(data["data"], list):
                        raise TypeError("Copilot model catalog data must be a list")

                    models: list[ModelInfo] = []
                    for model in data["data"]:
                        if not isinstance(model, dict):
                            raise TypeError("Copilot model catalog entry must be an object")
                        model_id = model.get("id")
                        if (
                            not isinstance(model_id, str)
                            or not model_id
                            or model_id != model_id.strip()
                        ):
                            raise TypeError(
                                "Copilot model catalog id must be a non-empty, unpadded string"
                            )

                        if "name" in model:
                            name = model["name"]
                            if not isinstance(name, str) or not name.strip():
                                raise TypeError(
                                    "Copilot model catalog name must be a non-empty string"
                                )
                            name = name.strip()
                        else:
                            name = model_id

                        model_picker_enabled = model.get("model_picker_enabled", True)
                        if not isinstance(model_picker_enabled, bool):
                            raise TypeError("Copilot model model_picker_enabled must be a boolean")

                        caps = model.get("capabilities", {})
                        if not isinstance(caps, dict):
                            raise TypeError("Copilot model capabilities must be an object")
                        capability_type = caps.get("type")
                        if "type" in caps and (
                            not isinstance(capability_type, str) or not capability_type.strip()
                        ):
                            raise TypeError(
                                "Copilot model capability type must be a non-empty string"
                            )
                        limits = caps.get("limits", {})
                        if not isinstance(limits, dict):
                            raise TypeError("Copilot model capability limits must be an object")
                        supports = caps.get("supports", {})
                        if not isinstance(supports, dict):
                            raise TypeError("Copilot model capability supports must be an object")

                        # Only include models that are enabled in model picker
                        if not model_picker_enabled:
                            continue
                        if capability_type == "completion":
                            logger.debug(
                                "Skipping Copilot completion-only model without RM route: %s",
                                model_id,
                            )
                            continue
                        supported_endpoints = normalize_supported_endpoints(model)
                        reasoning_values = _normalize_reasoning_effort_values(supports)
                        tools_support = _normalize_catalog_boolean(supports, "tool_calls")
                        vision_support = _normalize_catalog_boolean(supports, "vision")
                        thinking_support = _normalize_catalog_boolean(supports, "thinking")
                        parallel_tools_support = _normalize_catalog_boolean(
                            supports,
                            "parallel_tool_calls",
                        )
                        models.append(
                            ModelInfo(
                                id=model_id,
                                name=name,
                                provider=self.name,
                                max_prompt_tokens=_normalize_catalog_limit(
                                    limits, "max_prompt_tokens"
                                ),
                                max_output_tokens=_normalize_catalog_limit(
                                    limits, "max_output_tokens"
                                ),
                                max_context_window_tokens=_normalize_catalog_limit(
                                    limits, "max_context_window_tokens"
                                ),
                                supports_thinking=thinking_support is True,
                                supports_vision=vision_support is True,
                                reasoning_effort_values=reasoning_values,
                                supported_endpoints=supported_endpoints,
                                operation_capabilities=copilot_operation_capabilities(model),
                                feature_capabilities={
                                    **(
                                        {Feature.TOOLS: tools_support}
                                        if tools_support is not None
                                        else {}
                                    ),
                                    **(
                                        {Feature.VISION: vision_support}
                                        if vision_support is not None
                                        else {}
                                    ),
                                    **(
                                        {
                                            Feature.REASONING: (
                                                thinking_support is True or bool(reasoning_values)
                                            )
                                        }
                                        if thinking_support is not None
                                        or reasoning_values is not None
                                        else {}
                                    ),
                                    **(
                                        {Feature.PARALLEL_TOOLS: parallel_tools_support}
                                        if parallel_tools_support is not None
                                        else {}
                                    ),
                                },
                            )
                        )
                except (TypeError, ValueError) as e:
                    self._raise_protocol_error(self.name, None, e)

            # Update cache
            self._models_ttl_cache.set(deepcopy(models))

            logger.info("Fetched %d Copilot models", len(models))
            return deepcopy(models)
        except (httpx.HTTPError, ProviderError) as e:
            # If cache exists, return stale cache on error — including the case
            # where the in-loop forced refresh raised a ProviderError because the
            # token is unrecoverable. Serving the last known catalog is better
            # than hard-failing the models endpoint.
            stale = self._models_ttl_cache._value
            if stale is not None:
                logger.warning(
                    "Failed to refresh Copilot models, using stale cache (%s)",
                    type(e).__name__,
                )
                # Router planning consumes this stale snapshot as the current
                # catalog. Renew it here so request-time endpoint and option
                # lookups observe the exact same model capabilities.
                self._models_ttl_cache.set(deepcopy(stale))
                return deepcopy(stale)
            logger.error("Failed to list Copilot models (%s)", type(e).__name__)
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                "Failed to list Copilot models",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.TRANSPORT,
                provider=self.name,
                cause=e,
            ) from e

    # Tools that are not supported by Copilot Responses API.
    # ``namespace`` items are conditionally allowed: if they carry an inner
    # ``tools`` array (Codex's MCP registry shape) they MUST pass through
    # so Copilot can resolve namespaced function_calls like
    # ``kusto/execute_query``. Explicitly unsupported or malformed tools are
    # client errors; silently dropping them changes the requested semantics.
    UNSUPPORTED_TOOL_TYPES = {
        "web_search",
        "web_search_preview",
        "code_interpreter",
    }

    def _filter_unsupported_tools(
        self,
        tools: list[dict] | None,
        *,
        model: str | None = None,
    ) -> list[dict] | None:
        """Validate and preserve tools supported by the Copilot Responses API.

        Args:
            tools: List of tool definitions

        Returns:
            Validated list of tools, or None if empty
        """
        if not tools:
            return None

        validated = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                validated.append(tool)
            elif tool_type == "namespace":
                # Codex's MCP discovery returns namespace items wrapping
                # the actual function tools. Pass through ONLY when the
                # inner ``tools`` array is present and non-empty —
                # otherwise Copilot 400s with
                # ``Missing required parameter: 'tools[N].tools'``.
                inner = tool.get("tools")
                if isinstance(inner, list) and inner:
                    validated.append(tool)
                else:
                    raise RequestOptionError(
                        "GitHub Copilot requires namespace tools to contain a non-empty tools list",
                        provider=self.name,
                        model=model,
                        parameter="tools",
                    )
            elif tool_type not in self.UNSUPPORTED_TOOL_TYPES:
                validated.append(tool)
            else:
                raise RequestOptionError(
                    f"GitHub Copilot does not support Responses tool type '{tool_type}'",
                    provider=self.name,
                    model=model,
                    parameter="tools",
                )

        return validated if validated else None

    def _build_responses_payload(self, request: ResponsesRequest) -> dict:
        """Build payload for Responses API request.

        Args:
            request: The responses request

        Returns:
            Payload dictionary for the API
        """
        self._validate_provider_extensions(request)
        if request.temperature is not None:
            raise RequestOptionError(
                "GitHub Copilot Responses does not support request option 'temperature'",
                provider=self.name,
                model=request.model,
                parameter="temperature",
            )
        payload: dict = {
            "model": request.model,
            "input": request.input,
            "stream": request.stream,
        }
        if request.instructions:
            payload["instructions"] = request.instructions
        if request.max_output_tokens:
            payload["max_output_tokens"] = request.max_output_tokens
        # Tool support - filter out unsupported tools
        filtered_tools = self._filter_unsupported_tools(request.tools, model=request.model)
        if filtered_tools:
            payload["tools"] = filtered_tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        for key, value in {
            "top_p": request.top_p,
            "metadata": request.metadata,
            "service_tier": request.service_tier,
        }.items():
            if value is not None:
                payload[key] = value
        # Catalog-driven path: trust whatever Copilot's
        # ``capabilities.supports.reasoning_effort`` advertises for this model.
        # Mirrors the chat path at copilot.py:147-164 — both paths must agree,
        # otherwise codex+gpt-5.x silently runs at ``high`` while chat runs at
        # ``xhigh``. Falls back to ``downgrade_for_upstream`` when the catalog
        # is cold (first request after restart) so we never block on a fetch.
        catalog = self._catalog_effort_values(request.model)
        if catalog is not None:
            if not catalog:
                if request.reasoning_effort is not None:
                    raise RequestOptionError(
                        "GitHub Copilot does not support reasoning for this model",
                        provider=self.name,
                        model=request.model,
                        parameter="reasoning_effort",
                    )
                upstream_effort: str | None = None
            else:
                desired = request.reasoning_effort
                if desired is None:
                    upstream_effort = None
                else:
                    upstream_effort = pick_closest_effort(desired, catalog)
                    if upstream_effort is None:
                        raise RequestOptionError(
                            "GitHub Copilot has no reasoning tier at or below the requested tier",
                            provider=self.name,
                            model=request.model,
                            parameter="reasoning_effort",
                        )
        else:
            if (
                request.reasoning_effort is not None
                and _known_reasoning_support(request.model) is False
            ):
                raise RequestOptionError(
                    "GitHub Copilot does not support reasoning for this model",
                    provider=self.name,
                    model=request.model,
                    parameter="reasoning_effort",
                )
            if _known_reasoning_support(request.model) is True:
                upstream_effort = request.reasoning_effort
            else:
                upstream_effort = downgrade_for_upstream(request.reasoning_effort)
            if (
                _known_reasoning_support(request.model) is None
                and request.reasoning_effort in ("xhigh", "max")
                and upstream_effort == "high"
            ):
                logger.warning(
                    "Copilot Responses catalog cold for %s; "
                    "downgrading reasoning_effort=%s to high as a precaution",
                    request.model,
                    request.reasoning_effort,
                )
        if upstream_effort is not None:
            # ``summary: auto`` opts in to reasoning_summary_text events so we
            # can forward chain-of-thought as Anthropic thinking blocks.
            payload["reasoning"] = {"effort": upstream_effort, "summary": "auto"}
            # Copilot CAPI doesn't stream reasoning_summary_text deltas for some
            # models; the summary instead arrives in output_item.done.item.summary[].
            # Asking for encrypted_content also lets us round-trip reasoning state
            # across turns (matches vscode-copilot-chat reference client).
            payload["include"] = ["reasoning.encrypted_content"]
        return payload

    def validate_responses_request(self, request: ResponsesRequest) -> None:
        """Exercise the Responses payload policy without upstream I/O."""
        self._build_responses_payload(request)

    def _extract_response_content(self, data: dict) -> tuple[str, str | None]:
        """Extract text and refusal content from a Responses API response.

        Args:
            data: The response JSON data

        Returns:
            The extracted text content
        """
        content = ""
        refusal_parts: list[str] = []
        for output in data.get("output", []):
            if output.get("type") == "message":
                for content_item in output.get("content", []):
                    if content_item.get("type") == "output_text":
                        content += content_item.get("text", "")
                    elif content_item.get("type") == "refusal":
                        refusal_parts.append(content_item.get("refusal", ""))
        return content, "".join(refusal_parts) or None

    def _extract_reasoning(self, data: dict) -> tuple[str | None, str | None, str | None]:
        """Extract one atomic reasoning item's summary, upstream id, and blob.

        Returns ``(thinking_text, thinking_id, thinking_signature)``. All
        three are ``None`` when the response has no reasoning output. The
        ``id`` and the ``encrypted_content`` blob must be round-tripped
        together — the blob is signed against its id, so pairing it with a
        locally-generated id 400s the next turn.
        """
        output = next(
            (item for item in data.get("output", []) if item.get("type") == "reasoning"),
            None,
        )
        if output is None:
            return None, None, None
        text = "".join(summary.get("text") or "" for summary in output.get("summary", []))
        return (
            text or None,
            output.get("id"),
            output.get("encrypted_content"),
        )

    def _extract_tool_calls(self, data: dict) -> list[ResponsesToolCall]:
        """Extract tool calls from Responses API response.

        Args:
            data: The response JSON data

        Returns:
            List of tool calls
        """
        tool_calls = []
        for output in data.get("output", []):
            if output.get("type") == "function_call":
                tool_calls.append(
                    ResponsesToolCall(
                        call_id=output.get("call_id", ""),
                        name=output.get("name", ""),
                        arguments=output.get("arguments", "{}"),
                        namespace=output.get("namespace"),
                    )
                )
            elif output.get("type") == "custom_tool_call":
                tool_calls.append(
                    ResponsesToolCall(
                        call_id=output.get("call_id", ""),
                        name=output.get("name", ""),
                        arguments=output.get("input", ""),
                        kind="custom",
                    )
                )
            elif output.get("type") == "tool_search_call":
                args = output.get("arguments")
                if isinstance(args, str):
                    args_str = args
                elif args is None:
                    args_str = "{}"
                else:
                    args_str = json.dumps(args)
                tool_calls.append(
                    ResponsesToolCall(
                        call_id=output.get("call_id", ""),
                        name="tool_search",
                        arguments=args_str,
                        kind="tool_search",
                    )
                )
        return tool_calls

    @staticmethod
    def _validate_responses_output(data: dict) -> bool:
        """Validate the structural fields consumed by the Responses adapter."""
        output = data.get("output")
        if not isinstance(output, list):
            raise TypeError("Responses output must be a list")
        has_deliverable = False
        reasoning_item_count = 0
        for item in output:
            if not isinstance(item, dict):
                raise TypeError("Responses output item must be an object")
            item_type = item.get("type")
            if not isinstance(item_type, str):
                raise TypeError("Responses output item type must be a string")
            if item_type == "message":
                content = item.get("content")
                if not isinstance(content, list):
                    raise TypeError("Responses message content must be a list")
                for part in content:
                    if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                        raise TypeError("Responses message content part must be typed object")
                    if part.get("type") == "output_text" and not isinstance(part.get("text"), str):
                        raise TypeError("Responses output_text part must contain text")
                    if part.get("type") == "output_text" and part.get("text"):
                        has_deliverable = True
                    if part.get("type") == "refusal":
                        refusal = part.get("refusal")
                        if not isinstance(refusal, str) or not refusal:
                            raise TypeError(
                                "Responses refusal part must contain non-empty refusal text"
                            )
                        has_deliverable = True
            elif item_type in {"function_call", "custom_tool_call", "tool_search_call"}:
                required_fields = {
                    "function_call": ("call_id", "name", "arguments"),
                    "custom_tool_call": ("call_id", "name", "input"),
                    "tool_search_call": ("call_id",),
                }[item_type]
                if any(not isinstance(item.get(field), str) for field in required_fields):
                    raise TypeError(f"Responses {item_type} item is malformed")
                has_deliverable = True
            elif item_type == "reasoning":
                reasoning_item_count += 1
                if reasoning_item_count > 1:
                    raise TypeError("Responses response contains multiple atomic reasoning items")
                summary = item.get("summary", [])
                if not isinstance(summary, list):
                    raise TypeError("Responses reasoning summary must be a list")
                for entry in summary:
                    if not isinstance(entry, dict):
                        raise TypeError("Responses reasoning summary entry must be an object")
                    if "type" in entry and not isinstance(entry["type"], str):
                        raise TypeError("Responses reasoning summary type must be a string")
                    if "text" in entry and not isinstance(entry["text"], str):
                        raise TypeError("Responses reasoning summary text must be a string")
                for field in ("id", "encrypted_content"):
                    value = item.get(field)
                    if value is not None and not isinstance(value, str):
                        raise TypeError(f"Responses reasoning {field} must be a string or null")
                if item.get("encrypted_content") and not item.get("id"):
                    raise TypeError("Responses encrypted reasoning is missing its upstream id")
                if any(
                    isinstance(entry.get("text"), str) and bool(entry["text"]) for entry in summary
                ) or bool(item.get("encrypted_content")):
                    has_deliverable = True
        return has_deliverable

    @staticmethod
    def _validate_responses_usage(usage: object) -> None:
        """Validate the Responses usage fields consumed by downstream adapters."""
        BaseProvider._validated_token_usage(
            usage,
            fields=("input_tokens", "output_tokens", "total_tokens"),
            label="Responses",
            detail_fields={
                "input_tokens_details": ("cached_tokens",),
                "output_tokens_details": ("reasoning_tokens",),
            },
        )

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        """Generate a Responses API completion via Copilot (for Codex models)."""
        await self.ensure_token()

        payload = self._build_responses_payload(request)

        logger.debug("Copilot responses completion: model=%s", request.model)
        try:
            response = await self._send_with_auth_retry(
                "POST",
                COPILOT_RESPONSES_PATH,
                json=payload,
                headers_kwargs={
                    "vision_request": self._responses_input_has_vision(request.input),
                    "response_input": (
                        request.input if isinstance(request.input, (str, list)) else None
                    ),
                },
                model=request.model,
            )
            response.raise_for_status()
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("Responses response must be an object")
                model = self._validated_response_model(data, request.model)
                self._validate_responses_usage(data.get("usage"))
                has_deliverable = self._validate_responses_output(data)
                terminal_outcome = _responses_terminal_outcome(data)
                if (
                    terminal_outcome.transport is TransportTermination.EXCEPTION
                    and terminal_outcome.error is not None
                    and terminal_outcome.error.code == "upstream_protocol_error"
                ):
                    raise ValueError(terminal_outcome.error.message)
                if (
                    terminal_outcome.response_status is ResponseStatus.COMPLETED
                    and not has_deliverable
                ):
                    raise ValueError("completed Responses response contains no deliverable output")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                self._raise_protocol_error(self.name, request.model, e)

            content, refusal = self._extract_response_content(data)
            tool_calls = self._extract_tool_calls(data)
            thinking, thinking_id, thinking_sig = self._extract_reasoning(data)

            usage = None
            if "usage" in data:
                usage = data["usage"]

            logger.debug("Copilot responses completion successful")
            return ResponsesResponse(
                content=content,
                model=model,
                usage=usage,
                tool_calls=tool_calls if tool_calls else None,
                thinking=thinking,
                thinking_id=thinking_id,
                thinking_signature=thinking_sig,
                terminal_outcome=terminal_outcome,
                refusal=refusal,
            )
        except httpx.HTTPStatusError as e:
            self._raise_unsupported_operation_if_applicable(
                e.response.status_code,
                e.response.content,
                model=request.model,
                cause=e,
            )
            self._raise_http_status_error(
                "Copilot",
                e,
                logger,
                include_body=True,
                provider=self.name,
                model=request.model,
            )
        except httpx.TimeoutException as e:
            self._raise_timeout_error("Copilot", e, logger, provider=self.name, model=request.model)
        except httpx.HTTPError as e:
            self._raise_http_error("Copilot", e, logger, provider=self.name, model=request.model)

    async def responses_completion_stream(
        self, request: ResponsesRequest
    ) -> AsyncIterator[ResponsesStreamChunk]:
        """Generate a streaming Responses API completion via Copilot (for Codex models)."""
        await self.ensure_token()

        payload = self._build_responses_payload(request)
        payload["stream"] = True

        logger.debug("Copilot streaming responses: model=%s", request.model)
        logger.debug(
            "Copilot responses payload metadata: model=%s stream=%s tools=%d input_type=%s",
            request.model,
            payload.get("stream"),
            len(payload.get("tools") or []),
            type(request.input).__name__,
        )
        try:
            async with self._stream_with_auth_retry(
                COPILOT_RESPONSES_PATH,
                json=payload,
                headers_kwargs={
                    "vision_request": self._responses_input_has_vision(request.input),
                    "response_input": (
                        request.input if isinstance(request.input, (str, list)) else None
                    ),
                },
                model=request.model,
            ) as response:
                # Check for errors before processing stream
                if response.status_code >= 400:
                    # Read the error body before the context closes
                    error_body = await response.aread()
                    logger.error(
                        "Copilot responses stream API error: %d (body_bytes=%d)",
                        response.status_code,
                        len(error_body),
                    )
                    self._raise_unsupported_operation_if_applicable(
                        response.status_code,
                        error_body,
                        model=request.model,
                    )
                    # 401/403 are handled by _stream_with_auth_retry (refresh +
                    # retry, then a retryable auth error), so they never reach
                    # here; only transient 429/5xx are retryable at this point.
                    status_code = response.status_code
                    retryable = status_code in (429, 500, 502, 503, 504, 529)
                    kind = (
                        ProviderFailureKind.RATE_LIMIT
                        if status_code in (429, 529)
                        else ProviderFailureKind.UPSTREAM_STATUS
                    )
                    raise ProviderError(
                        f"Copilot API error: {status_code}",
                        status_code=status_code,
                        retryable=retryable,
                        kind=kind,
                        upstream_status_code=status_code,
                        provider=self.name,
                        model=request.model,
                    )

                stream_finished = False
                final_usage = None
                emitted_tool_call = False
                text_parts: dict[tuple[int, int], str] = {}
                completed_text_parts: dict[tuple[int, int], str] = {}
                refusal_parts: dict[tuple[int, int], str] = {}
                completed_refusal_parts: dict[tuple[int, int], str] = {}
                reasoning_parts: dict[tuple[int, int], str] = {}
                completed_reasoning_parts: dict[tuple[int, int], str] = {}
                reasoning_item_ids: dict[int, str] = {}
                completed_reasoning_items: dict[int, tuple[tuple[str, ...], str | None]] = {}
                declared_output_item_types: dict[int, str] = {}
                completed_output_items: dict[int, tuple[str, str]] = {}
                emitted_reasoning_signatures: set[int] = set()
                # Track pending function calls being streamed, keyed by output_index
                # (Copilot obfuscates item IDs differently across events, so we can't match by ID)
                pending_fcs: dict[int, dict] = {}
                # Diagnostic: count any event types we don't explicitly handle
                # so we can spot custom_tool_call_input.* or other channels we
                # might be dropping.
                unknown_event_counts: dict[str, int] = {}

                def bind_declared_type(output_index: int, item_type: str) -> None:
                    declared_type = declared_output_item_types.get(output_index)
                    if declared_type is not None and declared_type != item_type:
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError("Responses output index changed item type"),
                        )
                    declared_output_item_types[output_index] = item_type

                async for line in response.aiter_lines():
                    if stream_finished:
                        break

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        # ``[DONE]`` terminates the SSE transport but carries no
                        # Responses semantic status. The route detects an
                        # unexpected EOF unless response.done/completed already
                        # produced an explicit terminal chunk.
                        break

                    try:
                        data = json.loads(data_str)
                        if not isinstance(data, dict):
                            raise TypeError("Responses stream event must be an object")
                        event_type = data.get("type", "")
                        if not isinstance(event_type, str):
                            raise TypeError("Responses stream event type must be a string")
                        if event_type in {
                            "response.output_item.added",
                            "response.output_item.done",
                        }:
                            item = data.get("item")
                            if not isinstance(item, dict):
                                raise TypeError("Responses stream output item must be an object")
                            item_type = item.get("type")
                            if not isinstance(item_type, str):
                                raise TypeError("Responses output item type must be a string")
                            output_index = data.get("output_index", 0)
                            if not isinstance(output_index, int) or isinstance(output_index, bool):
                                raise TypeError("Responses output_index must be an integer")
                            if item_type in {"function_call", "custom_tool_call"}:
                                for field in ("call_id", "name"):
                                    if not isinstance(item.get(field), str):
                                        raise TypeError(
                                            f"Responses {item_type} {field} must be a string"
                                        )
                            if item_type == "function_call":
                                namespace = item.get("namespace")
                                if namespace is not None and not isinstance(namespace, str):
                                    raise TypeError(
                                        "Responses function call namespace must be a string or null"
                                    )
                            if item_type == "tool_search_call" and not isinstance(
                                item.get("call_id"), str
                            ):
                                raise TypeError("Responses tool_search_call call_id must be string")
                            if item_type == "reasoning":
                                summary = item.get("summary", []) or []
                                if not isinstance(summary, list):
                                    raise TypeError("Responses reasoning summary must be a list")
                                for entry in summary:
                                    if not isinstance(entry, dict):
                                        raise TypeError(
                                            "Responses reasoning summary entry malformed"
                                        )
                                    text = entry.get("text")
                                    if text is not None and not isinstance(text, str):
                                        raise TypeError("Responses reasoning text must be string")
                                for field in ("id", "encrypted_content"):
                                    value = item.get(field)
                                    if value is not None and not isinstance(value, str):
                                        raise TypeError(
                                            f"Responses reasoning {field} must be a string"
                                        )
                            if item_type == "message":
                                content = item.get("content", [])
                                if not isinstance(content, list):
                                    raise TypeError("Responses message content must be a list")
                                for part in content:
                                    if not isinstance(part, dict):
                                        raise TypeError(
                                            "Responses message content part must be an object"
                                        )
                                    part_type = part.get("type")
                                    if part_type not in {"output_text", "refusal"}:
                                        raise TypeError(
                                            "Responses message content part type is unsupported"
                                        )
                                    if part_type == "output_text" and not isinstance(
                                        part.get("text"), str
                                    ):
                                        raise TypeError(
                                            "Responses message output text must be a string"
                                        )
                                    if part_type == "refusal" and not isinstance(
                                        part.get("refusal"), str
                                    ):
                                        raise TypeError(
                                            "Responses message refusal must be a string"
                                        )
                        if event_type in {
                            "response.done",
                            "response.completed",
                            "response.incomplete",
                            "response.failed",
                            "response.cancelled",
                        }:
                            terminal_response = data.get("response")
                            if not isinstance(terminal_response, dict):
                                raise TypeError("Responses terminal response must be an object")
                            self._validate_responses_usage(terminal_response.get("usage"))
                        if event_type in {
                            "response.output_text.delta",
                            "response.reasoning_summary_text.delta",
                            "response.refusal.delta",
                        } and not isinstance(data.get("delta"), str):
                            raise TypeError("Responses text delta must be a string")
                        if event_type == "response.output_text.done" and not isinstance(
                            data.get("text"), str
                        ):
                            raise TypeError("Responses output text must be a string")
                        if event_type in {
                            "response.output_text.delta",
                            "response.output_text.done",
                        }:
                            for field in ("output_index", "content_index"):
                                value = data.get(field, 0)
                                if not isinstance(value, int) or isinstance(value, bool):
                                    raise TypeError(
                                        f"Responses output text {field} must be an integer"
                                    )
                        if event_type == "response.refusal.done" and not isinstance(
                            data.get("refusal"), str
                        ):
                            raise TypeError("Responses refusal must be a string")
                        if event_type in {
                            "response.refusal.delta",
                            "response.refusal.done",
                        }:
                            for field in ("output_index", "content_index"):
                                value = data.get(field, 0)
                                if not isinstance(value, int) or isinstance(value, bool):
                                    raise TypeError(f"Responses refusal {field} must be an integer")
                        if event_type == "response.reasoning_summary_text.done" and not isinstance(
                            data.get("text"), str
                        ):
                            raise TypeError("Responses reasoning done text must be a string")
                        if event_type in {
                            "response.reasoning_summary_text.delta",
                            "response.reasoning_summary_text.done",
                        }:
                            item_id = data.get("item_id")
                            if item_id is not None and not isinstance(item_id, str):
                                raise TypeError(
                                    "Responses reasoning item_id must be a string or null"
                                )
                            for field in ("output_index", "summary_index"):
                                value = data.get(field, 0)
                                if not isinstance(value, int) or isinstance(value, bool):
                                    raise TypeError(
                                        f"Responses reasoning {field} must be an integer"
                                    )
                        if event_type in {
                            "response.reasoning_summary_part.added",
                            "response.reasoning_summary_part.done",
                        }:
                            item_id = data.get("item_id")
                            if item_id is not None and not isinstance(item_id, str):
                                raise TypeError(
                                    "Responses reasoning item_id must be a string or null"
                                )
                            for field in ("output_index", "summary_index"):
                                value = data.get(field, 0)
                                if (
                                    not isinstance(value, int)
                                    or isinstance(value, bool)
                                    or value < 0
                                ):
                                    raise TypeError(
                                        f"Responses reasoning {field} must be a non-negative "
                                        "integer"
                                    )
                            part = data.get("part")
                            if not isinstance(part, dict):
                                raise TypeError(
                                    "Responses reasoning summary part must be an object"
                                )
                            if part.get("type") != "summary_text":
                                raise TypeError(
                                    "Responses reasoning summary part type must be summary_text"
                                )
                            if not isinstance(part.get("text"), str):
                                raise TypeError(
                                    "Responses reasoning summary part text must be a string"
                                )
                        if event_type in {
                            "response.function_call_arguments.delta",
                            "response.custom_tool_call_input.delta",
                        } and not isinstance(data.get("delta"), str):
                            raise TypeError("Responses tool delta must be a string")
                        if event_type == "response.function_call_arguments.done" and not isinstance(
                            data.get("arguments"), str
                        ):
                            raise TypeError("Responses function arguments must be a string")
                        if event_type == "response.custom_tool_call_input.done" and not isinstance(
                            data.get("input"), str
                        ):
                            raise TypeError("Responses custom tool input must be a string")
                        if event_type in {
                            "response.function_call_arguments.delta",
                            "response.function_call_arguments.done",
                            "response.custom_tool_call_input.delta",
                            "response.custom_tool_call_input.done",
                        }:
                            output_index = data.get("output_index", 0)
                            if not isinstance(output_index, int) or isinstance(output_index, bool):
                                raise TypeError("Responses output_index must be an integer")
                        if event_type == "response.output_item.done":
                            item = data["item"]
                            item_type = item.get("type")
                            if item_type == "function_call":
                                for field in ("call_id", "name", "arguments"):
                                    if not isinstance(item.get(field), str):
                                        raise TypeError(
                                            f"Responses function call {field} must be a string"
                                        )
                            elif item_type == "custom_tool_call" and not isinstance(
                                item.get("input"), str
                            ):
                                raise TypeError("Responses custom tool input must be a string")
                    except (json.JSONDecodeError, TypeError) as e:
                        self._raise_protocol_error(self.name, request.model, e)

                    typed_item_type = {
                        "response.output_text.delta": "message",
                        "response.output_text.done": "message",
                        "response.refusal.delta": "message",
                        "response.refusal.done": "message",
                        "response.reasoning_summary_text.delta": "reasoning",
                        "response.reasoning_summary_text.done": "reasoning",
                        "response.reasoning_summary_part.added": "reasoning",
                        "response.reasoning_summary_part.done": "reasoning",
                        "response.function_call_arguments.delta": "function_call",
                        "response.function_call_arguments.done": "function_call",
                        "response.custom_tool_call_input.delta": "custom_tool_call",
                        "response.custom_tool_call_input.done": "custom_tool_call",
                    }.get(event_type)
                    if typed_item_type is not None:
                        bind_declared_type(data.get("output_index", 0), typed_item_type)

                    # Handle text delta events
                    if event_type == "response.output_text.delta":
                        delta_text = data.get("delta", "")
                        text_part = (
                            data.get("output_index", 0),
                            data.get("content_index", 0),
                        )
                        if text_part in completed_text_parts:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("Responses output text delta followed a completed part"),
                            )
                        if delta_text:
                            text_parts[text_part] = text_parts.get(text_part, "") + delta_text
                            yield ResponsesStreamChunk(
                                content=delta_text,
                                output_index=text_part[0],
                                content_index=text_part[1],
                                output_item_type="message",
                            )
                        else:
                            yield ResponsesStreamChunk(
                                content="",
                                output_index=text_part[0],
                                content_index=text_part[1],
                                provenance_only=True,
                                output_item_type="message",
                            )

                    elif event_type == "response.output_text.done":
                        snapshot = data.get("text", "")
                        text_part = (
                            data.get("output_index", 0),
                            data.get("content_index", 0),
                        )
                        completed = completed_text_parts.get(text_part)
                        if completed is not None:
                            if completed != snapshot:
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses output text part completed with conflicting "
                                        "snapshots"
                                    ),
                                )
                            continue
                        accumulated = text_parts.get(text_part, "")
                        if not snapshot.startswith(accumulated):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses output text done snapshot conflicts with its deltas"
                                ),
                            )
                        completed_text_parts[text_part] = snapshot
                        suffix = snapshot[len(accumulated) :]
                        if suffix:
                            text_parts[text_part] = snapshot
                            yield ResponsesStreamChunk(
                                content=suffix,
                                output_index=text_part[0],
                                content_index=text_part[1],
                                output_item_type="message",
                            )
                        elif not accumulated:
                            yield ResponsesStreamChunk(
                                content="",
                                output_index=text_part[0],
                                content_index=text_part[1],
                                provenance_only=True,
                                output_item_type="message",
                            )

                    elif event_type == "response.refusal.delta":
                        refusal_delta = data.get("delta", "")
                        refusal_part = (
                            data.get("output_index", 0),
                            data.get("content_index", 0),
                        )
                        if refusal_part in completed_refusal_parts:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("Responses refusal delta followed a completed part"),
                            )
                        if refusal_delta:
                            refusal_parts[refusal_part] = (
                                refusal_parts.get(refusal_part, "") + refusal_delta
                            )
                            yield ResponsesStreamChunk(
                                content="",
                                refusal=refusal_delta,
                                output_index=refusal_part[0],
                                content_index=refusal_part[1],
                                output_item_type="message",
                            )
                        else:
                            yield ResponsesStreamChunk(
                                content="",
                                output_index=refusal_part[0],
                                content_index=refusal_part[1],
                                provenance_only=True,
                                output_item_type="message",
                            )

                    elif event_type == "response.refusal.done":
                        snapshot = data.get("refusal", "")
                        refusal_part = (
                            data.get("output_index", 0),
                            data.get("content_index", 0),
                        )
                        if refusal_part in completed_refusal_parts:
                            if completed_refusal_parts[refusal_part] != snapshot:
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses refusal part completed with "
                                        "conflicting snapshots"
                                    ),
                                )
                            continue
                        accumulated = refusal_parts.get(refusal_part, "")
                        if not snapshot.startswith(accumulated):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses refusal done snapshot conflicts with its deltas"
                                ),
                            )
                        completed_refusal_parts[refusal_part] = snapshot
                        suffix = snapshot[len(accumulated) :]
                        if suffix:
                            refusal_parts[refusal_part] = snapshot
                            yield ResponsesStreamChunk(
                                content="",
                                refusal=suffix,
                                output_index=refusal_part[0],
                                content_index=refusal_part[1],
                                output_item_type="message",
                            )
                        elif not accumulated:
                            yield ResponsesStreamChunk(
                                content="",
                                output_index=refusal_part[0],
                                content_index=refusal_part[1],
                                provenance_only=True,
                                output_item_type="message",
                            )

                    # Reasoning summary (chain-of-thought) deltas — surfaced so
                    # entry routes (Anthropic, Gemini) can forward them as
                    # thinking blocks. Copilot may obfuscate ``item_id``
                    # independently on every summary event, so only
                    # ``output_index`` correlates these deltas. The canonical
                    # identity arrives with ``output_item.done`` and its
                    # encrypted blob.
                    elif event_type == "response.reasoning_summary_text.delta":
                        delta = data.get("delta", "")
                        output_index = data.get("output_index", 0)
                        reasoning_part = (output_index, data.get("summary_index", 0))
                        if output_index in completed_reasoning_items:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("Responses reasoning delta followed a completed item"),
                            )
                        if reasoning_part in completed_reasoning_parts:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses reasoning delta followed a completed summary part"
                                ),
                            )
                        if delta:
                            reasoning_parts[reasoning_part] = (
                                reasoning_parts.get(reasoning_part, "") + delta
                            )
                            yield ResponsesStreamChunk(
                                content="",
                                thinking=delta,
                                output_index=output_index,
                                reasoning_summary_index=reasoning_part[1],
                                output_item_type="reasoning",
                            )
                        else:
                            yield ResponsesStreamChunk(
                                content="",
                                output_index=output_index,
                                reasoning_summary_index=reasoning_part[1],
                                provenance_only=True,
                                output_item_type="reasoning",
                            )

                    elif event_type == "response.reasoning_summary_text.done":
                        # Don't yield ``thinking_signature=item_id`` here. The
                        # Codex path treats every signature as ``encrypted_content``
                        # and round-trips it to Copilot, which then 400s with
                        # ``Encrypted content could not be decrypted`` because
                        # ``item_id`` is just a local identifier, not the real
                        # encrypted blob. The real blob arrives later on
                        # ``output_item.done.item.encrypted_content``; emit the
                        # signature there so both Codex and Anthropic round-trips
                        # use the verifiable value.
                        output_index = data.get("output_index", 0)
                        known_item_id = reasoning_item_ids.get(output_index)
                        summary_index = data.get("summary_index", 0)
                        reasoning_part = (output_index, summary_index)
                        snapshot = data.get("text", "")
                        completed_item = completed_reasoning_items.get(output_index)
                        if completed_item is not None:
                            summary_snapshots = completed_item[0]
                            if (
                                summary_index >= len(summary_snapshots)
                                or summary_snapshots[summary_index] != snapshot
                            ):
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses reasoning done conflicts with a completed item"
                                    ),
                                )
                            encrypted_blob = completed_item[1]
                            if (
                                encrypted_blob
                                and known_item_id
                                and output_index not in emitted_reasoning_signatures
                            ):
                                emitted_reasoning_signatures.add(output_index)
                                yield ResponsesStreamChunk(
                                    content="",
                                    output_index=output_index,
                                    reasoning_summary_index=summary_index,
                                    output_item_type="reasoning",
                                    output_item_done=True,
                                    thinking_id=known_item_id,
                                    thinking_signature=encrypted_blob,
                                )
                            continue
                        if reasoning_part in completed_reasoning_parts:
                            if completed_reasoning_parts[reasoning_part] != snapshot:
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses reasoning summary completed with conflicting "
                                        "snapshots"
                                    ),
                                )
                            continue
                        accumulated = reasoning_parts.get(reasoning_part, "")
                        if not snapshot.startswith(accumulated):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses reasoning done snapshot conflicts with its deltas"
                                ),
                            )
                        completed_reasoning_parts[reasoning_part] = snapshot
                        suffix = snapshot[len(accumulated) :]
                        if suffix:
                            reasoning_parts[reasoning_part] = snapshot
                            yield ResponsesStreamChunk(
                                content="",
                                thinking=suffix,
                                output_index=output_index,
                                reasoning_summary_index=summary_index,
                                output_item_type="reasoning",
                            )
                        elif not accumulated:
                            yield ResponsesStreamChunk(
                                content="",
                                output_index=output_index,
                                reasoning_summary_index=summary_index,
                                provenance_only=True,
                                output_item_type="reasoning",
                            )

                    # Handle function call output_item.added - start of a new function call
                    elif event_type == "response.output_item.added":
                        item = data.get("item", {})
                        output_idx = data.get("output_index", 0)
                        item_type = item.get("type")
                        bind_declared_type(output_idx, item_type)
                        if item.get("type") == "function_call":
                            pending_fcs[output_idx] = {
                                "call_id": item.get("call_id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                                "kind": "function",
                                # MCP namespace (e.g. "kusto"). Required on
                                # round-trip or Copilot 400s the next turn.
                                "namespace": item.get("namespace"),
                            }
                        elif item.get("type") == "custom_tool_call":
                            # Custom tools (e.g. Codex's apply_patch) stream
                            # raw text via custom_tool_call_input.delta. Same
                            # bookkeeping as function_call but flagged so the
                            # route emits the right event shape downstream.
                            pending_fcs[output_idx] = {
                                "call_id": item.get("call_id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                                "kind": "custom",
                            }
                        elif item.get("type") == "tool_search_call":
                            # Codex CLI registers a `tool_search` tool
                            # (execution=client) so the model can dynamically
                            # discover MCP tools. Codex's dispatcher matches on
                            # ResponseItem::ToolSearchCall — wrapping this as a
                            # function_call(name="tool_search") makes the call
                            # silently abort (registry has no function tool of
                            # that name). Tag with kind="tool_search" so the
                            # route emits a real tool_search_call item.
                            # NOTE: arguments arrive whole on output_item.done;
                            # if Copilot ever streams them via a dedicated
                            # delta event we'll spot it via unknown_event_counts.
                            pending_fcs[output_idx] = {
                                "call_id": item.get("call_id", ""),
                                "name": "tool_search",
                                "arguments": "",
                                "kind": "tool_search",
                            }

                    # Handle function call arguments delta - accumulate silently
                    elif event_type == "response.function_call_arguments.delta":
                        delta = data.get("delta", "")
                        output_idx = data.get("output_index", 0)
                        fc = pending_fcs.get(output_idx)
                        if fc is None:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("Responses function call payload is missing its item"),
                            )
                        if delta:
                            fc["arguments"] += delta

                    elif event_type == "response.custom_tool_call_input.delta":
                        delta = data.get("delta", "")
                        output_idx = data.get("output_index", 0)
                        fc = pending_fcs.get(output_idx)
                        if fc is None:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("Responses custom tool payload is missing its item"),
                            )
                        if delta:
                            fc["arguments"] += delta

                    # Handle function call arguments done — finalize arguments
                    # but DON'T emit yet. Copilot CAPI sends the ``namespace``
                    # field (required for MCP-namespaced tools like
                    # ``kusto/execute_query``) on the *later* ``output_item.done``
                    # event, not on this one. Emitting here loses namespace and
                    # the next turn 400s with ``Missing namespace for
                    # function_call 'X'``. Defer to output_item.done.
                    elif event_type == "response.function_call_arguments.done":
                        output_idx = data.get("output_index", 0)
                        fc = pending_fcs.get(output_idx)
                        if fc is None:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("Responses function call payload is missing its item"),
                            )
                        fc["arguments"] = data.get("arguments", fc["arguments"])

                    elif event_type == "response.custom_tool_call_input.done":
                        output_idx = data.get("output_index", 0)
                        fc = pending_fcs.get(output_idx)
                        if fc is None:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("Responses custom tool payload is missing its item"),
                            )
                        fc["arguments"] = data.get("input", fc["arguments"])

                    # Handle output_item.done for function calls. Copilot
                    # delivers ``namespace`` (for MCP tools) on this event only.
                    elif event_type == "response.output_item.done":
                        item = data.get("item", {})
                        item_type = item.get("type")
                        output_idx = data.get("output_index", 0)
                        bind_declared_type(output_idx, item_type)
                        completed_type = (
                            "reasoning" if output_idx in completed_reasoning_items else None
                        )
                        if item_type in {
                            "message",
                            "function_call",
                            "custom_tool_call",
                            "tool_search_call",
                        }:
                            if completed_type is not None:
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses output index completed with "
                                        "conflicting item types"
                                    ),
                                )
                            item_snapshot = json.dumps(
                                item,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                            completed_snapshot = completed_output_items.get(output_idx)
                            if completed_snapshot is not None:
                                if completed_snapshot != (item_type, item_snapshot):
                                    self._raise_protocol_error(
                                        self.name,
                                        request.model,
                                        TypeError(
                                            "Responses output item completed with conflicting "
                                            "snapshots"
                                        ),
                                    )
                                continue
                            completed_output_items[output_idx] = (item_type, item_snapshot)
                        if item.get("type") == "function_call":
                            fc = pending_fcs.pop(output_idx, None) or {}
                            # Prefer the final item payload (carries namespace
                            # and the canonical arguments). Fall back to the
                            # bookkeeping dict if the item is sparse.
                            emitted_tool_call = True
                            yield ResponsesStreamChunk(
                                content="",
                                tool_call=ResponsesToolCall(
                                    call_id=item.get("call_id") or fc.get("call_id", ""),
                                    name=item.get("name") or fc.get("name", ""),
                                    arguments=item.get("arguments")
                                    or fc.get("arguments", "")
                                    or "{}",
                                    kind=fc.get("kind", "function"),
                                    namespace=item.get("namespace") or fc.get("namespace"),
                                ),
                                output_index=output_idx,
                                output_item_type="function_call",
                                output_item_done=True,
                            )
                        elif item.get("type") == "custom_tool_call":
                            # Fallback: if custom_tool_call_input.done didn't
                            # fire (or pending_fcs was already drained), emit
                            # from the final item payload.
                            fc = pending_fcs.pop(output_idx, None)
                            emitted_tool_call = True
                            yield ResponsesStreamChunk(
                                content="",
                                tool_call=ResponsesToolCall(
                                    call_id=item.get("call_id") or (fc or {}).get("call_id", ""),
                                    name=item.get("name") or (fc or {}).get("name", ""),
                                    arguments=item.get("input") or (fc or {}).get("arguments", ""),
                                    kind="custom",
                                ),
                                output_index=output_idx,
                                output_item_type="custom_tool_call",
                                output_item_done=True,
                            )
                        elif item.get("type") == "reasoning":
                            # Copilot CAPI delivers the reasoning summary here
                            # rather than via reasoning_summary_text.delta.
                            summary_list = item.get("summary", []) or []
                            upstream_id = item.get("id")
                            output_index = data.get("output_index", 0)
                            known_item_id = reasoning_item_ids.get(output_index)
                            if (
                                upstream_id is not None
                                and known_item_id is not None
                                and upstream_id != known_item_id
                            ):
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses reasoning item changed id for one output index"
                                    ),
                                )
                            if upstream_id is not None:
                                reasoning_item_ids[output_index] = upstream_id
                            effective_item_id = upstream_id or known_item_id
                            summary_snapshots = tuple(
                                (summary.get("text") or "")
                                for summary in summary_list
                                if isinstance(summary, dict)
                            )
                            encrypted_blob = item.get("encrypted_content")
                            item_snapshot = (summary_snapshots, encrypted_blob)
                            completed_non_reasoning = completed_output_items.get(output_index)
                            if completed_non_reasoning is not None:
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses output index completed with "
                                        "conflicting item types"
                                    ),
                                )
                            if output_index in completed_reasoning_items:
                                if completed_reasoning_items[output_index] != item_snapshot:
                                    self._raise_protocol_error(
                                        self.name,
                                        request.model,
                                        TypeError(
                                            "Responses reasoning item completed with conflicting "
                                            "snapshots"
                                        ),
                                    )
                                if (
                                    encrypted_blob
                                    and effective_item_id
                                    and output_index not in emitted_reasoning_signatures
                                ):
                                    emitted_reasoning_signatures.add(output_index)
                                    yield ResponsesStreamChunk(
                                        content="",
                                        output_index=output_index,
                                        reasoning_summary_index=(
                                            len(completed_reasoning_items[output_index][0]) - 1
                                            if completed_reasoning_items[output_index][0]
                                            else None
                                        ),
                                        output_item_type="reasoning",
                                        output_item_done=True,
                                        thinking_id=effective_item_id,
                                        thinking_signature=encrypted_blob,
                                    )
                                continue
                            tracked_summary_indices = {
                                summary_index
                                for part_output_index, summary_index in (
                                    set(reasoning_parts) | set(completed_reasoning_parts)
                                )
                                if part_output_index == output_index
                            }
                            final_summary_indices = set(range(len(summary_snapshots)))
                            if not tracked_summary_indices.issubset(final_summary_indices):
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses reasoning item omitted a streamed summary part"
                                    ),
                                )
                            logger.info(
                                "Copilot /responses reasoning item: "
                                "summary_segments=%d encrypted=%s id=%s",
                                len(summary_list),
                                bool(item.get("encrypted_content")),
                                upstream_id,
                            )
                            for summary_index, snapshot in enumerate(summary_snapshots):
                                reasoning_part = (output_index, summary_index)
                                if reasoning_part in completed_reasoning_parts:
                                    if completed_reasoning_parts[reasoning_part] != snapshot:
                                        self._raise_protocol_error(
                                            self.name,
                                            request.model,
                                            TypeError(
                                                "Responses reasoning item summary conflicts with "
                                                "its completed snapshot"
                                            ),
                                        )
                                    continue
                                accumulated = reasoning_parts.get(reasoning_part, "")
                                if not snapshot.startswith(accumulated):
                                    self._raise_protocol_error(
                                        self.name,
                                        request.model,
                                        TypeError(
                                            "Responses reasoning item summary conflicts with its "
                                            "deltas"
                                        ),
                                    )
                                completed_reasoning_parts[reasoning_part] = snapshot
                                suffix = snapshot[len(accumulated) :]
                                if suffix:
                                    reasoning_parts[reasoning_part] = snapshot
                                    yield ResponsesStreamChunk(
                                        content="",
                                        thinking=suffix,
                                        output_index=output_index,
                                        reasoning_summary_index=summary_index,
                                        output_item_type="reasoning",
                                        thinking_id=effective_item_id,
                                    )
                            completed_reasoning_items[output_index] = item_snapshot
                            # Emit the upstream id and the encrypted blob
                            # separately so the route can pair them on the
                            # reasoning output item it forwards to Codex. The
                            # blob is only valid against its own id; using a
                            # locally-generated id (or worse, treating the id
                            # as the signature) 400s the next turn with
                            # ``Encrypted content could not be decrypted``.
                            if encrypted_blob and effective_item_id:
                                emitted_reasoning_signatures.add(output_index)
                                yield ResponsesStreamChunk(
                                    content="",
                                    output_index=output_index,
                                    reasoning_summary_index=(
                                        len(summary_snapshots) - 1 if summary_snapshots else None
                                    ),
                                    output_item_type="reasoning",
                                    output_item_done=True,
                                    thinking_id=effective_item_id,
                                    thinking_signature=encrypted_blob,
                                )
                            elif effective_item_id:
                                yield ResponsesStreamChunk(
                                    content="",
                                    output_index=output_index,
                                    reasoning_summary_index=(
                                        len(summary_snapshots) - 1 if summary_snapshots else None
                                    ),
                                    output_item_type="reasoning",
                                    output_item_done=True,
                                    thinking_id=effective_item_id,
                                )
                            elif not encrypted_blob:
                                yield ResponsesStreamChunk(
                                    content="",
                                    output_index=output_index,
                                    reasoning_summary_index=(
                                        len(summary_snapshots) - 1 if summary_snapshots else None
                                    ),
                                    output_item_type="reasoning",
                                    output_item_done=True,
                                )
                        elif item.get("type") == "tool_search_call":
                            # Forward as kind="tool_search" so the route emits
                            # an actual tool_search_call wire item — codex's
                            # dispatcher refuses anything else (see v0.3.7
                            # changelog). Arguments arrive as a dict here;
                            # serialize to a JSON string for transport on the
                            # ResponsesToolCall dataclass; the route deserializes
                            # before re-emitting.
                            fc = pending_fcs.pop(output_idx, None)
                            args = item.get("arguments")
                            if isinstance(args, str):
                                args_str = args
                            elif args is None:
                                args_str = "{}"
                            else:
                                args_str = json.dumps(args)
                            call_id = item.get("call_id") or (fc and fc.get("call_id")) or ""
                            emitted_tool_call = True
                            yield ResponsesStreamChunk(
                                content="",
                                tool_call=ResponsesToolCall(
                                    call_id=call_id,
                                    name="tool_search",
                                    arguments=args_str,
                                    kind="tool_search",
                                ),
                                output_index=output_idx,
                                output_item_type="tool_search_call",
                                output_item_done=True,
                            )
                        else:
                            if item_type == "message":
                                streamed_content_indices = {
                                    content_index
                                    for part_output_index, content_index in (
                                        set(text_parts)
                                        | set(refusal_parts)
                                        | set(completed_text_parts)
                                        | set(completed_refusal_parts)
                                    )
                                    if part_output_index == output_idx
                                }
                                snapshot_content_indices = set(range(len(item.get("content", []))))
                                if not streamed_content_indices.issubset(snapshot_content_indices):
                                    self._raise_protocol_error(
                                        self.name,
                                        request.model,
                                        TypeError(
                                            "Responses message snapshot conflicts with streamed "
                                            "content indices"
                                        ),
                                    )
                                for content_index, part in enumerate(item.get("content", [])):
                                    part_type = part.get("type")
                                    if part_type == "output_text":
                                        snapshot = part.get("text", "")
                                        key = (output_idx, content_index)
                                        accumulated = text_parts.get(key, "")
                                        if not snapshot.startswith(accumulated):
                                            self._raise_protocol_error(
                                                self.name,
                                                request.model,
                                                TypeError(
                                                    "Responses message text snapshot conflicts "
                                                    "with streamed deltas"
                                                ),
                                            )
                                        suffix = snapshot[len(accumulated) :]
                                        if suffix:
                                            text_parts[key] = snapshot
                                            yield ResponsesStreamChunk(
                                                content=suffix,
                                                output_index=output_idx,
                                                content_index=content_index,
                                                output_item_type="message",
                                            )
                                    elif part_type == "refusal":
                                        snapshot = part.get("refusal", "")
                                        key = (output_idx, content_index)
                                        accumulated = refusal_parts.get(key, "")
                                        if not snapshot.startswith(accumulated):
                                            self._raise_protocol_error(
                                                self.name,
                                                request.model,
                                                TypeError(
                                                    "Responses message refusal snapshot conflicts "
                                                    "with streamed deltas"
                                                ),
                                            )
                                        suffix = snapshot[len(accumulated) :]
                                        if suffix:
                                            refusal_parts[key] = snapshot
                                            yield ResponsesStreamChunk(
                                                content="",
                                                refusal=suffix,
                                                output_index=output_idx,
                                                content_index=content_index,
                                                output_item_type="message",
                                            )
                                yield ResponsesStreamChunk(
                                    content="",
                                    output_index=output_idx,
                                    output_item_type="message",
                                    output_item_done=True,
                                )
                            elif item_type not in _BENIGN_DONE_ITEM_TYPES:
                                key = f"output_item.done:{item_type}"
                                unknown_event_counts[key] = unknown_event_counts.get(key, 0) + 1

                    # Preserve the inner response status even when an upstream
                    # gateway uses a mismatched outer event name (for example,
                    # response.completed carrying status=incomplete).
                    elif event_type in {
                        "response.done",
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                        "response.cancelled",
                    }:
                        resp = data.get("response", {})
                        if isinstance(resp, dict):
                            final_usage = resp.get("usage") or final_usage
                        terminal_outcome = _responses_terminal_outcome(resp)
                        if any(
                            encrypted_blob and output_index not in reasoning_item_ids
                            for output_index, (_, encrypted_blob) in (
                                completed_reasoning_items.items()
                            )
                        ):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses encrypted reasoning is missing its upstream id"
                                ),
                            )
                        if (
                            terminal_outcome.transport is TransportTermination.EXCEPTION
                            and terminal_outcome.error is not None
                            and terminal_outcome.error.code == "upstream_protocol_error"
                        ):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                ValueError(terminal_outcome.error.message),
                            )
                        logger.info(
                            "Copilot /responses stream terminal: model=%s outer_type=%s "
                            "status=%s emitted_tool_call=%s usage=%s",
                            request.model,
                            event_type,
                            terminal_outcome.response_status.value,
                            emitted_tool_call,
                            final_usage,
                        )
                        yield ResponsesStreamChunk(
                            content="",
                            usage=final_usage,
                            terminal_outcome=terminal_outcome,
                        )
                        stream_finished = True

                    # Catch-all: count any other event types so we can spot
                    # things we silently drop (e.g. custom_tool_call_input.*).
                    # ``_BENIGN_UPSTREAM_EVENTS`` are intentionally skipped —
                    # the route synthesizes equivalents from the deltas we
                    # already consume.
                    else:
                        if event_type not in _BENIGN_UPSTREAM_EVENTS:
                            unknown_event_counts[event_type] = (
                                unknown_event_counts.get(event_type, 0) + 1
                            )

                if not stream_finished:
                    logger.warning(
                        "Copilot /responses transport ended without an explicit terminal event: "
                        "model=%s",
                        request.model,
                    )

                if unknown_event_counts:
                    logger.warning(
                        "Copilot /responses unhandled event types: model=%s counts=%s",
                        request.model,
                        unknown_event_counts,
                    )

        except httpx.TimeoutException as e:
            self._raise_timeout_error(
                "Copilot",
                e,
                logger,
                stream=True,
                provider=self.name,
                model=request.model,
            )
        except httpx.HTTPError as e:
            self._raise_http_error(
                "Copilot",
                e,
                logger,
                stream=True,
                provider=self.name,
                model=request.model,
            )

    def _reject_option(self, request: ChatRequest, parameter: str) -> None:
        raise RequestOptionError(
            f"GitHub Copilot Chat does not support request option '{parameter}'",
            provider=self.name,
            model=request.model,
            parameter=parameter,
        )

    def _build_chat_payload(self, request: ChatRequest, *, stream: bool) -> dict:
        """Build a Copilot Chat payload with an explicit option policy."""
        self._validate_provider_extensions(request)
        messages, _has_images = self._build_messages_payload(request)
        payload: dict = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        for parameter in ("top_k", "candidate_count", "response_mime_type"):
            if getattr(request, parameter) is not None:
                self._reject_option(request, parameter)
        if request.stop is not None and request.stop_sequences is not None:
            self._reject_option(request, "stop")
        options = {
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop if request.stop is not None else request.stop_sequences,
            "user": request.user,
            "metadata": request.metadata,
            "service_tier": request.service_tier,
        }
        for key, value in options.items():
            if value is not None:
                payload[key] = value
        apply_copilot_chat_reasoning(
            payload,
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            catalog_effort_values=self._catalog_effort_values(request.model),
        )
        return payload

    def validate_chat_request(self, request: ChatRequest, *, stream: bool) -> None:
        """Exercise the same Chat or Responses policy as actual execution."""
        from router_maestro.utils.responses_bridge import (
            chat_request_to_responses_request,
            should_use_responses_for_chat,
        )

        operation = Operation.RESPONSES_STREAM if stream else Operation.RESPONSES
        if should_use_responses_for_chat(
            request,
            self.name,
            responses_supported=self._catalog_operation_support(request.model, operation),
        ):
            self.validate_responses_request(chat_request_to_responses_request(request))
            return
        self._build_chat_payload(request, stream=stream)
