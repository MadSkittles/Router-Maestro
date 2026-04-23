"""GitHub Copilot provider implementation."""

import json
import time
from collections.abc import AsyncIterator

import httpx

from router_maestro.auth import AuthManager, AuthType
from router_maestro.auth.github_oauth import get_copilot_token
from router_maestro.auth.storage import OAuthCredential
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponsesToolCall,
)
from router_maestro.providers.tool_parsing import recover_tool_calls_from_content
from router_maestro.utils import get_logger
from router_maestro.utils.cache import TTLCache
from router_maestro.utils.reasoning import budget_to_effort, downgrade_for_upstream

logger = get_logger("providers.copilot")

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_CHAT_URL = f"{COPILOT_BASE_URL}/chat/completions"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
COPILOT_RESPONSES_URL = f"{COPILOT_BASE_URL}/responses"


def _claude_supports_reasoning(bare_lower: str) -> bool:
    """Whether a Claude model on Copilot exposes any reasoning control.

    Per the Copilot model picker (vscode-copilot-chat), the 4.6+ generations
    accept ``reasoning_effort``. Older models (sonnet-4, sonnet-4.5,
    opus-4.5, haiku-4.5) have no reasoning control surface.
    """
    return (
        bare_lower.startswith("claude-opus-4.7")
        or bare_lower.startswith("claude-opus-4.6")
        or bare_lower.startswith("claude-sonnet-4.6")
    )


_EFFORT_ORDER = ("low", "medium", "high", "xhigh")


def _pick_closest_effort(desired: str, allowed: list[str]) -> str | None:
    """Pick the value from ``allowed`` closest to ``desired`` on the L/M/H/XH ladder.

    Strategy when an exact match is unavailable: prefer the next *higher* tier
    (the user asked for thinking — give them more, not less), and fall back to
    the next lower tier if no higher tier is offered. Returns ``None`` only if
    ``allowed`` is empty.
    """
    if not allowed:
        return None
    if desired in allowed:
        return desired
    try:
        target = _EFFORT_ORDER.index(desired)
    except ValueError:
        # Unknown tier — give them whatever the catalog ranks highest.
        return max(allowed, key=lambda v: _EFFORT_ORDER.index(v) if v in _EFFORT_ORDER else -1)
    higher = [v for v in allowed if v in _EFFORT_ORDER and _EFFORT_ORDER.index(v) > target]
    if higher:
        return min(higher, key=lambda v: _EFFORT_ORDER.index(v))
    lower = [v for v in allowed if v in _EFFORT_ORDER and _EFFORT_ORDER.index(v) < target]
    if lower:
        return max(lower, key=lambda v: _EFFORT_ORDER.index(v))
    return allowed[0]


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

    * ``claude-opus-4.6*`` / ``claude-sonnet-4.6`` accept ``low``/``medium``/``high``.
    * ``claude-opus-4.7`` only accepts ``medium`` (clamp).
    * Older Claudes (4.5 / sonnet-4 / haiku) take no reasoning field.
    * ``gpt-5*`` / ``o1`` / ``o3`` / ``o4`` accept ``low``/``medium``/``high``/``xhigh``.
    * ``gpt-4*``, ``gemini-*`` take no reasoning field.

    For ``gpt-5.4*`` the gateway also requires ``max_completion_tokens`` instead
    of ``max_tokens``; this function performs that rewrite when present.
    """
    bare = model.split("/", 1)[1] if "/" in model else model
    bare_lower = bare.lower()

    is_claude = bare_lower.startswith("claude-")
    is_openai_reasoning = (
        bare_lower.startswith("gpt-5")
        or bare_lower.startswith("o1")
        or bare_lower.startswith("o3")
        or bare_lower.startswith("o4")
    )

    # Catalog-driven path: trust whatever Copilot advertises.
    if catalog_effort_values is not None:
        if not catalog_effort_values:
            # Catalog explicitly says no reasoning supported — emit nothing.
            pass
        else:
            desired = reasoning_effort or budget_to_effort(thinking_budget)
            if desired is None and thinking_budget is not None:
                # Client asked for thinking without a clear effort — be
                # aggressive and aim for the top of the catalog.
                desired = "high"
            if desired is not None:
                picked = _pick_closest_effort(desired, catalog_effort_values)
                if picked is not None:
                    payload["reasoning_effort"] = picked
        if bare_lower.startswith("gpt-5.4") and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        return

    # Hardcoded fallback when the catalog hasn't been fetched yet.
    if is_claude:
        if not _claude_supports_reasoning(bare_lower):
            # Older Claudes (sonnet-4 / sonnet-4.5 / opus-4.5 / haiku-4.5)
            # have no reasoning control on Copilot.
            pass
        else:
            effort = reasoning_effort or budget_to_effort(thinking_budget)
            if effort == "xhigh":
                effort = "high"  # Claude effort enum tops out at high
            if effort is None and thinking_budget is not None:
                effort = "high"
            # opus-4.7's gateway only accepts "medium" today. Clamp anything
            # else to medium so the request doesn't fail.
            if bare_lower.startswith("claude-opus-4.7") and effort in (
                "low",
                "medium",
                "high",
            ):
                effort = "medium"
            if effort in ("low", "medium", "high"):
                payload["reasoning_effort"] = effort
    elif is_openai_reasoning:
        effort = reasoning_effort or budget_to_effort(thinking_budget)
        # Copilot's gpt-5* line natively accepts xhigh (verified against the
        # gateway's supported_values list), so don't downgrade here.
        if effort in ("low", "medium", "high", "xhigh"):
            payload["reasoning_effort"] = effort

    if bare_lower.startswith("gpt-5.4") and "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")


# Model cache TTL in seconds (5 minutes)
MODELS_CACHE_TTL = 300


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

    def __init__(self) -> None:
        self.auth_manager = AuthManager()
        self._cached_token: str | None = None
        self._token_expires: int = 0
        # Model cache
        self._models_ttl_cache: TTLCache[list[ModelInfo]] = TTLCache(MODELS_CACHE_TTL)
        # Reusable HTTP client
        self._client: httpx.AsyncClient | None = None

    def is_authenticated(self) -> bool:
        """Check if authenticated with GitHub Copilot."""
        cred = self.auth_manager.get_credential("github-copilot")
        return cred is not None and cred.type == AuthType.OAUTH

    async def ensure_token(self) -> None:
        """Ensure we have a valid Copilot token, refreshing if needed."""
        cred = self.auth_manager.get_credential("github-copilot")
        if not cred or not isinstance(cred, OAuthCredential):
            logger.error("Not authenticated with GitHub Copilot")
            raise ProviderError("Not authenticated with GitHub Copilot", status_code=401)

        current_time = int(time.time())

        # Check if we need to refresh (token expired or will expire soon)
        if self._cached_token and self._token_expires > current_time + 60:
            return  # Token still valid

        logger.debug("Refreshing Copilot token")
        # Use a fresh short-lived client for token refresh to avoid blocking
        # on the shared streaming HTTP/2 connection pool
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_NON_STREAMING) as client:
                copilot_token = await get_copilot_token(client, cred.refresh)
        except httpx.HTTPError as e:
            logger.error("Failed to refresh Copilot token: %s", e)
            raise ProviderError(f"Failed to refresh Copilot token: {e}", retryable=True)

        self._cached_token = copilot_token.token
        self._token_expires = copilot_token.expires_at

        # Update stored credential with new access token (immutable pattern)
        updated_cred = OAuthCredential(
            refresh=cred.refresh,
            access=copilot_token.token,
            expires=copilot_token.expires_at,
        )
        self.auth_manager.storage.set("github-copilot", updated_cred)
        self.auth_manager.save()
        logger.debug("Copilot token refreshed, expires at %d", copilot_token.expires_at)

    def _get_headers(self, vision_request: bool = False) -> dict[str, str]:
        """Get headers for Copilot API requests.

        Args:
            vision_request: Whether this request contains images (vision)
        """
        if not self._cached_token:
            raise ProviderError("No valid token available", status_code=401)

        headers = {
            "Authorization": f"Bearer {self._cached_token}",
            "Content-Type": "application/json",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot/1.0.0",
            "Copilot-Integration-Id": "vscode-chat",
        }

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

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create a reusable HTTP client."""
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
        return self._client

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
            messages.append(msg)

            # Check if this message contains images (multimodal content)
            if isinstance(m.content, list):
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_images = True
                        break

        return messages, has_images

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion via Copilot."""
        await self.ensure_token()

        messages, has_images = self._build_messages_payload(request)

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        apply_copilot_chat_reasoning(
            payload,
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            catalog_effort_values=self._catalog_effort_values(request.model),
        )

        logger.debug(
            "Copilot chat completion: model=%s thinking_budget=%s reasoning_effort=%s "
            "payload_thinking_budget=%s payload_reasoning_effort=%s",
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            payload.get("thinking_budget"),
            payload.get("reasoning_effort"),
        )
        client = self._get_client()
        try:
            response = await client.post(
                COPILOT_CHAT_URL,
                json=payload,
                headers=self._get_headers(vision_request=has_images),
                timeout=TIMEOUT_NON_STREAMING,
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens") or 0

            bare_lower = (
                request.model.split("/", 1)[1] if "/" in request.model else request.model
            ).lower()
            reasoning_capable = bare_lower.startswith("claude-") and _claude_supports_reasoning(
                bare_lower
            )

            if not choices:
                # Reasoning-capable Claude models can spend their whole budget
                # on hidden reasoning and finish without emitting visible text.
                # That comes back as choices=[] with completion_tokens>0. Only
                # absorb it as an empty success when the client explicitly
                # asked for thinking on a known reasoning model — otherwise a
                # malformed upstream response would silently look like a blank
                # assistant message.
                if completion_tokens > 0 and reasoning_capable and _thinking_requested(request):
                    logger.warning(
                        "Copilot returned empty choices but completion_tokens=%d; "
                        "treating as thinking-only response",
                        completion_tokens,
                    )
                    return ChatResponse(
                        content="",
                        model=data.get("model", request.model),
                        finish_reason="stop",
                        usage=usage or None,
                        tool_calls=None,
                    )
                logger.error("Copilot API returned empty choices: %s", json.dumps(data)[:500])
                raise ProviderError(
                    f"Copilot API returned empty choices: {json.dumps(data)[:500]}",
                    status_code=500,
                    retryable=True,
                )

            logger.debug("Copilot chat completion successful")

            # Copilot may return multiple choices: one with text content and
            # separate ones each containing a single tool_call. Merge them all.
            content = None
            tool_calls = []
            finish_reason = "stop"
            thinking_text = ""
            thinking_signature: str | None = None

            for choice in choices:
                msg = choice.get("message", {})
                # Take content from the first choice that has it
                if content is None and msg.get("content"):
                    content = msg["content"]
                # Collect tool_calls from all choices
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
                # Use finish_reason from any choice (they should all match)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                # Collect reasoning text/signature only if the client opted in
                if _thinking_requested(request):
                    t, sig = _extract_reasoning_from_chunk(msg)
                    if t:
                        thinking_text += t
                    if sig and thinking_signature is None:
                        thinking_signature = sig

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

            return ChatResponse(
                content=content,
                model=data.get("model", request.model),
                finish_reason=finish_reason,
                usage=data.get("usage"),
                tool_calls=tool_calls,
                thinking=thinking_text or None,
                thinking_signature=thinking_signature,
            )
        except httpx.HTTPStatusError as e:
            self._raise_http_status_error("Copilot", e, logger, include_body=True)
        except httpx.TimeoutException as e:
            self._raise_timeout_error("Copilot", e, logger)
        except httpx.HTTPError as e:
            self._raise_http_error("Copilot", e, logger)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion via Copilot."""
        await self.ensure_token()

        messages, has_images = self._build_messages_payload(request)

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        apply_copilot_chat_reasoning(
            payload,
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            catalog_effort_values=self._catalog_effort_values(request.model),
        )

        logger.debug(
            "Copilot streaming chat: model=%s thinking_budget=%s reasoning_effort=%s "
            "payload_thinking_budget=%s payload_reasoning_effort=%s",
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            payload.get("thinking_budget"),
            payload.get("reasoning_effort"),
        )
        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                COPILOT_CHAT_URL,
                json=payload,
                headers=self._get_headers(vision_request=has_images),
            ) as response:
                response.raise_for_status()

                stream_finished = False
                async for line in response.aiter_lines():
                    if stream_finished:
                        break

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    data = json.loads(data_str)

                    # Extract usage if present (may come in separate chunk)
                    usage = data.get("usage")

                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        finish_reason = data["choices"][0].get("finish_reason")
                        tool_calls = delta.get("tool_calls")
                        if _thinking_requested(request):
                            thinking_text, thinking_sig = _extract_reasoning_from_chunk(delta)
                        else:
                            thinking_text, thinking_sig = "", None

                        if (
                            content
                            or finish_reason
                            or usage
                            or tool_calls
                            or thinking_text
                            or thinking_sig
                        ):
                            yield ChatStreamChunk(
                                content=content,
                                finish_reason=finish_reason,
                                usage=usage,
                                tool_calls=tool_calls,
                                thinking=thinking_text or None,
                                thinking_signature=thinking_sig,
                            )

                        # Mark stream as finished after receiving finish_reason
                        if finish_reason:
                            stream_finished = True
                    elif usage:
                        # Handle usage-only chunks (no choices)
                        yield ChatStreamChunk(
                            content="",
                            finish_reason=None,
                            usage=usage,
                        )
        except httpx.HTTPStatusError as e:
            self._raise_http_status_error("Copilot", e, logger, stream=True, include_body=True)
        except httpx.TimeoutException as e:
            self._raise_timeout_error("Copilot", e, logger, stream=True)
        except httpx.HTTPError as e:
            # Verbose diagnostics for stream connection issues (PR #17/#18)
            resp = getattr(e, "response", None)
            resp_text = resp.text if resp is not None else "No response"
            headers = {
                k: v
                for k, v in self._get_headers(vision_request=has_images).items()
                if k != "Authorization"
            }
            logger.error(
                "Copilot stream HTTP error: type=%s error=%r\n"
                "Request payload: %s\nRequest URL: %s\n"
                "Request headers: %s\nResponse: %s",
                type(e).__name__,
                e,
                json.dumps(payload, default=str)[:2000],
                COPILOT_CHAT_URL,
                headers,
                resp_text,
            )
            raise ProviderError(f"HTTP error: {type(e).__name__}: {e}", retryable=True)

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
                return cached

        await self.ensure_token()

        logger.debug("Fetching Copilot models from API")
        # Use a fresh short-lived client to avoid blocking on the shared
        # streaming HTTP/2 connection pool
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_NON_STREAMING) as client:
                response = await client.get(
                    COPILOT_MODELS_URL,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()

            models = []
            for model in data.get("data", []):
                # Only include models that are enabled in model picker
                if model.get("model_picker_enabled", True):
                    caps = model.get("capabilities", {})
                    limits = caps.get("limits", {})
                    supports = caps.get("supports", {})
                    reasoning_values = supports.get("reasoning_effort")
                    # Catalog field is optional and shape-flexible — accept
                    # list/tuple of strings or treat anything else as "unset".
                    if isinstance(reasoning_values, (list, tuple)):
                        reasoning_values = [str(v) for v in reasoning_values if isinstance(v, str)]
                    else:
                        reasoning_values = None
                    models.append(
                        ModelInfo(
                            id=model["id"],
                            name=model.get("name", model["id"]),
                            provider=self.name,
                            max_prompt_tokens=limits.get("max_prompt_tokens"),
                            max_output_tokens=limits.get("max_output_tokens"),
                            max_context_window_tokens=limits.get("max_context_window_tokens"),
                            supports_thinking=bool(supports.get("thinking")),
                            supports_vision=bool(supports.get("vision")),
                            reasoning_effort_values=reasoning_values,
                        )
                    )

            # Update cache
            self._models_ttl_cache.set(models)

            logger.info("Fetched %d Copilot models", len(models))
            return models
        except httpx.HTTPError as e:
            # If cache exists, return stale cache on error
            stale = self._models_ttl_cache._value
            if stale is not None:
                logger.warning("Failed to refresh Copilot models, using stale cache: %s", e)
                return stale
            logger.error("Failed to list Copilot models: %s", e)
            raise ProviderError(f"Failed to list models: {e}", retryable=True)

    # Tools that are not supported by Copilot Responses API.
    # ``namespace`` is emitted by Codex CLI to group MCP servers behind a
    # tool-search facade; Copilot rejects it with
    # ``Missing required parameter: 'tools[N].tools'``.
    UNSUPPORTED_TOOL_TYPES = {
        "web_search",
        "web_search_preview",
        "code_interpreter",
        "namespace",
    }

    def _filter_unsupported_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Filter out tools that are not supported by Copilot API.

        Args:
            tools: List of tool definitions

        Returns:
            Filtered list of tools, or None if empty
        """
        if not tools:
            return None

        filtered = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            # Only include function tools, filter out unsupported built-in tools
            if tool_type == "function":
                filtered.append(tool)
            elif tool_type not in self.UNSUPPORTED_TOOL_TYPES:
                filtered.append(tool)
            else:
                logger.debug("Filtering out unsupported tool type: %s", tool_type)

        return filtered if filtered else None

    def _build_responses_payload(self, request: ResponsesRequest) -> dict:
        """Build payload for Responses API request.

        Args:
            request: The responses request

        Returns:
            Payload dictionary for the API
        """
        payload: dict = {
            "model": request.model,
            "input": request.input,
            "stream": request.stream,
        }
        if request.instructions:
            payload["instructions"] = request.instructions
        if request.temperature != 1.0:
            payload["temperature"] = request.temperature
        if request.max_output_tokens:
            payload["max_output_tokens"] = request.max_output_tokens
        # Tool support - filter out unsupported tools
        filtered_tools = self._filter_unsupported_tools(request.tools)
        if filtered_tools:
            payload["tools"] = filtered_tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        upstream_effort = downgrade_for_upstream(request.reasoning_effort)
        if upstream_effort is not None:
            if request.reasoning_effort == "xhigh":
                logger.warning(
                    "Copilot Responses does not accept reasoning_effort=xhigh; downgrading to high"
                )
            payload["reasoning"] = {"effort": upstream_effort}
        return payload

    def _extract_response_content(self, data: dict) -> str:
        """Extract text content from Responses API response.

        Args:
            data: The response JSON data

        Returns:
            The extracted text content
        """
        content = ""
        for output in data.get("output", []):
            if output.get("type") == "message":
                for content_item in output.get("content", []):
                    if content_item.get("type") == "output_text":
                        content += content_item.get("text", "")
        return content

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
                    )
                )
        return tool_calls

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        """Generate a Responses API completion via Copilot (for Codex models)."""
        await self.ensure_token()

        payload = self._build_responses_payload(request)

        logger.debug("Copilot responses completion: model=%s", request.model)
        client = self._get_client()
        try:
            response = await client.post(
                COPILOT_RESPONSES_URL,
                json=payload,
                headers=self._get_headers(),
                timeout=TIMEOUT_NON_STREAMING,
            )
            response.raise_for_status()
            data = response.json()

            content = self._extract_response_content(data)
            tool_calls = self._extract_tool_calls(data)

            usage = None
            if "usage" in data:
                usage = data["usage"]

            logger.debug("Copilot responses completion successful")
            return ResponsesResponse(
                content=content,
                model=data.get("model", request.model),
                usage=usage,
                tool_calls=tool_calls if tool_calls else None,
            )
        except httpx.HTTPStatusError as e:
            self._raise_http_status_error("Copilot", e, logger, include_body=True)
        except httpx.TimeoutException as e:
            self._raise_timeout_error("Copilot", e, logger)
        except httpx.HTTPError as e:
            self._raise_http_error("Copilot", e, logger)

    async def responses_completion_stream(
        self, request: ResponsesRequest
    ) -> AsyncIterator[ResponsesStreamChunk]:
        """Generate a streaming Responses API completion via Copilot (for Codex models)."""
        await self.ensure_token()

        payload = self._build_responses_payload(request)
        payload["stream"] = True

        logger.debug("Copilot streaming responses: model=%s", request.model)
        logger.debug("Copilot responses payload: %s", payload)
        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                COPILOT_RESPONSES_URL,
                json=payload,
                headers=self._get_headers(),
            ) as response:
                # Check for errors before processing stream
                if response.status_code >= 400:
                    # Read the error body before the context closes
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    logger.error(
                        "Copilot responses stream API error: %d - %s",
                        response.status_code,
                        error_text,
                    )
                    retryable = response.status_code in (429, 500, 502, 503, 504)
                    raise ProviderError(
                        f"Copilot API error: {response.status_code} - {error_text}",
                        status_code=response.status_code,
                        retryable=retryable,
                    )

                stream_finished = False
                final_usage = None
                # Track pending function calls being streamed, keyed by output_index
                # (Copilot obfuscates item IDs differently across events, so we can't match by ID)
                pending_fcs: dict[int, dict] = {}

                async for line in response.aiter_lines():
                    if stream_finished:
                        break

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        # Stream ended, emit final chunk if we haven't already
                        if not stream_finished:
                            yield ResponsesStreamChunk(
                                content="",
                                finish_reason="stop",
                                usage=final_usage,
                            )
                            stream_finished = True
                        break

                    data = json.loads(data_str)
                    event_type = data.get("type", "")

                    # Handle text delta events
                    if event_type == "response.output_text.delta":
                        delta_text = data.get("delta", "")
                        if delta_text:
                            yield ResponsesStreamChunk(content=delta_text)

                    # Handle function call output_item.added - start of a new function call
                    elif event_type == "response.output_item.added":
                        item = data.get("item", {})
                        if item.get("type") == "function_call":
                            output_idx = data.get("output_index", 0)
                            pending_fcs[output_idx] = {
                                "call_id": item.get("call_id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                            }

                    # Handle function call arguments delta - accumulate silently
                    elif event_type == "response.function_call_arguments.delta":
                        delta = data.get("delta", "")
                        output_idx = data.get("output_index", 0)
                        fc = pending_fcs.get(output_idx)
                        if fc and delta:
                            fc["arguments"] += delta

                    # Handle function call arguments done
                    elif event_type == "response.function_call_arguments.done":
                        output_idx = data.get("output_index", 0)
                        fc = pending_fcs.pop(output_idx, None)
                        if fc:
                            fc["arguments"] = data.get("arguments", fc["arguments"])
                            # Emit complete tool call
                            yield ResponsesStreamChunk(
                                content="",
                                tool_call=ResponsesToolCall(
                                    call_id=fc["call_id"],
                                    name=fc["name"],
                                    arguments=fc["arguments"],
                                ),
                            )

                    # Handle output_item.done for function calls
                    elif event_type == "response.output_item.done":
                        item = data.get("item", {})
                        if item.get("type") == "function_call":
                            output_idx = data.get("output_index", 0)
                            # Only emit if not already done via function_call_arguments.done
                            fc = pending_fcs.pop(output_idx, None)
                            if fc is not None:
                                yield ResponsesStreamChunk(
                                    content="",
                                    tool_call=ResponsesToolCall(
                                        call_id=item.get("call_id", ""),
                                        name=item.get("name", ""),
                                        arguments=item.get("arguments", "{}"),
                                    ),
                                )

                    # Handle done event to get final usage
                    elif event_type == "response.done":
                        resp = data.get("response", {})
                        final_usage = resp.get("usage")
                        yield ResponsesStreamChunk(
                            content="",
                            finish_reason="stop",
                            usage=final_usage,
                        )
                        stream_finished = True

                    # Handle completed events
                    elif event_type == "response.completed":
                        # Final response received - emit finish chunk
                        resp = data.get("response", {})
                        if not final_usage:
                            final_usage = resp.get("usage")
                        yield ResponsesStreamChunk(
                            content="",
                            finish_reason="stop",
                            usage=final_usage,
                        )
                        stream_finished = True

                # If stream ended without explicit completion event, emit final chunk
                if not stream_finished:
                    logger.debug("Stream ended without completion event, emitting final chunk")
                    yield ResponsesStreamChunk(
                        content="",
                        finish_reason="stop",
                        usage=final_usage,
                    )

        except httpx.TimeoutException as e:
            self._raise_timeout_error("Copilot", e, logger, stream=True)
        except httpx.HTTPError as e:
            self._raise_http_error("Copilot", e, logger, stream=True)
