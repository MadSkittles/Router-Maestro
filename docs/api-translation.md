# API Translation Layer - Router-Maestro

## Overview

Router-Maestro uses an internal hub-and-spoke architecture. OpenAI Chat,
Anthropic, and Gemini requests are normalized to a typed `ChatRequest`; OpenAI
Responses uses a typed `ResponsesRequest`. The router builds one immutable,
capability-aware route plan, providers execute the selected wire operation, and
the entry protocol encodes the result.

```
Anthropic Request ──► translate_anthropic_to_openai() ──► ChatRequest ──► Provider ──► ChatResponse ──► build_anthropic_response() ──► Anthropic Response
OpenAI Request ──────────────── (passthrough) ──────────► ChatRequest ──► Provider ──► ChatResponse ──────────── (passthrough) ──────────────► OpenAI Response
OpenAI Responses ───────────── typed normalization ─────► ResponsesRequest ──► Provider/bridge ──► Responses output
Gemini Request ─────────────── protocol translation ────► ChatRequest ──► Provider ──► ChatResponse ──► Gemini Response
```

Routing and translation are separate decisions. A `ModelRef` always identifies
one provider plus one upstream model. A provider may adapt an operation (for
example, Chat through its Responses transport), and the beta Anthropic route
may adapt a model lacking native Anthropic transport through the standard
translated path. Neither adaptation is model fallback. Selecting another
`ModelRef` is allowed only by the frozen route plan after a retryable execution
failure.

---

## Translation Paths

### Path 1: Anthropic → Internal (Request Inbound)

**Function:** `translate_anthropic_to_openai()` in
`src/router_maestro/server/translation.py`

| Anthropic Field | Internal (ChatRequest) Field | Transformation |
|---|---|---|
| `model` | `model` | Preserved exactly for capability-aware Router resolution |
| `messages` | `messages` | Content blocks → role-based messages (see below) |
| `system` | System `Message` | String or TextBlock list → single system message, sanitized |
| `max_tokens` | `max_tokens` | Passthrough |
| `temperature` | `temperature` | Passthrough |
| `stream` | `stream` | Passthrough |
| `tools` | `tools` | `input_schema` → `parameters`, wrapped in `{"type":"function","function":{...}}` |
| `tool_choice` | `tool_choice` | `auto`→`"auto"`, `any`→`"required"`, `tool`→`{"type":"function",...}` |
| `thinking.type` | `thinking_type` | Extracted directly |
| `thinking.budget_tokens` | `thinking_budget` | Extracted directly |
| `output_config.effort` | `reasoning_effort` | Extracted directly; replaces adaptive budgets only |
| `top_p`, `top_k` | Typed request options | Preserved for providers that support them |
| `stop_sequences` | `stop_sequences` | Preserved and translated to provider-native stop syntax |
| `metadata` | `metadata` | Preserved when the target provider accepts metadata |

When `output_config.effort` is present with adaptive thinking, Router-Maestro
clears the internal `thinking_budget` before provider routing. Manual
`thinking.type="enabled"` retains its required `budget_tokens` alongside effort
when sufficient output-token headroom exists. Without effort, the existing budget
fallback remains: client `budget_tokens`, per-model server budget, then the server
default for an explicit thinking request.
Adaptive budgets may be used internally for provider effort mapping, but
Anthropic-native payloads always serialize adaptive thinking as
`{"type":"adaptive"}` without `budget_tokens`.

#### Message Translation Details

**User messages** (`_handle_user_message`):
- String content → `Message(role="user", content=text)`
- Content blocks are split:
  - `tool_result` → `Message(role="tool", tool_call_id=...)`
  - `text` → included in user message
  - `image` (base64) → OpenAI `image_url` format
  - `document` → preserved as an OpenAI-compatible file/image content part
  - `tool_reference` → skipped (logged)

`thinking` is not a valid Anthropic user content block and is rejected by the
request schema rather than being folded into user text. Images and documents
inside `tool_result.content` cannot remain on an OpenAI `tool` message, so they
are preserved in a follow-up multimodal user message after the tool result.

**Assistant messages** (`_handle_assistant_message`):
- `text` blocks → concatenated into `content`
- `thinking` blocks → omitted from legacy OpenAI assistant history; replaying
  hidden reasoning as visible content would corrupt the conversation
- `tool_use` blocks → OpenAI `tool_calls` format:
  ```
  Anthropic: {"type":"tool_use","id":"...","name":"...","input":{...}}
  OpenAI:    {"id":"...","type":"function","function":{"name":"...","arguments":"JSON string"}}
  ```

**System prompt** (`_sanitize_system_prompt`):
- Removes `x-anthropic-billing-header` lines (Claude Code adds these; Copilot rejects them)

#### Model Identity Preservation (`_translate_model_name`)

| Input | Output | Rule |
|---|---|---|
| `claude-sonnet-4-20250514` | `claude-sonnet-4-20250514` | Preserve concrete dated identity |
| `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` | Preserve punctuation and date |
| `anthropic/claude-opus-4.6` | `anthropic/claude-opus-4.6` | Preserve provider scope |

Undated names can still be resolved as aliases by the Router. Translation does
not collapse a concrete dated/versioned ID before the Router selects a catalog
entry and snapshots its capabilities.

---

### Path 2: Internal → Anthropic (Response Outbound, Non-Streaming)

**Function:** `build_anthropic_response()` in
`src/router_maestro/server/protocols/anthropic_reducer.py`

| Internal / OpenAI Field | Anthropic Field | Transformation |
|---|---|---|
| `choices[0].message.content` | `content[]` (TextBlock) | Wrapped in `{"type":"text","text":"..."}` |
| `choices[0].message.tool_calls` | `content[]` (ToolUseBlock) | Each becomes `{"type":"tool_use","id":"...","name":"...","input":{}}` |
| `choices[0].finish_reason` | `stop_reason` | `stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use` |
| `usage.prompt_tokens` | `usage.input_tokens` | Renamed |
| `usage.completion_tokens` | `usage.output_tokens` | Renamed |

The returned `model` is the provider-qualified model that actually executed the
request. An internal refusal is mapped to a text block here because Anthropic
Messages has no refusal content-block type; it remains typed on OpenAI Chat and
Responses boundaries.

---

### Path 3: OpenAI → Internal (Request Inbound, Passthrough)

**File:** `src/router_maestro/server/routes/chat.py`

OpenAI requests map almost directly to `ChatRequest`. Core and semantic options
are explicitly typed rather than merged into an unrestricted payload map:

```python
ChatRequest(
    model=request.model,
    messages=[Message(role=m.role, content=m.content) for m in request.messages],
    temperature=request.temperature,
    max_tokens=request.max_tokens,
    stream=request.stream,
    tools=request.tools,
    tool_choice=request.tool_choice,
    top_p=request.top_p,
    frequency_penalty=request.frequency_penalty,
    presence_penalty=request.presence_penalty,
    stop=request.stop,
    user=request.user,
    reasoning_effort=request.reasoning_effort,
)
```

Providers validate the complete request before execution. An accepted option is
forwarded or translated. A semantic option the selected model/provider cannot
represent is rejected with an OpenAI `invalid_request_error` and HTTP 400; it is
not silently omitted and does not trigger model fallback.

Omission is also part of the contract. If Chat, Responses, or Gemini omits
`temperature`, the internal value remains `None` and provider payload builders
leave the key out. An explicit value such as `1.0` remains present. Copilot Chat
forwards it, but Copilot Responses rejects any explicit temperature with a typed
OpenAI HTTP 400 before opening either a non-streaming or streaming response.

OpenAI Responses accepts only `reasoning.effort` in Phase 1. An empty reasoning
object is a no-op. `reasoning.summary` and any unknown sibling are rejected with
an OpenAI `invalid_request_error` whose parameter identifies the exact field;
streaming validation occurs before SSE commitment.

### Terminal and HTTP Semantics

Transport termination and response semantics are independent. Router-Maestro
records whether a provider ended with an explicit terminal, unexpected EOF,
client cancellation, or exception, separately from whether the response was
completed, incomplete, failed, cancelled, or unknown. Only an explicit
`completed` terminal is success; a clean EOF without a terminal is
`unexpected_eof`.

| Boundary | HTTP behavior | Protocol behavior |
|---|---|---|
| Validation, routing, provider open, or primed first-chunk failure before the route returns SSE | Entry-protocol non-2xx JSON error | No success SSE terminal is emitted |
| Failure after the response has started | HTTP status remains committed (normally `200`) | Exactly one protocol-native error/incomplete terminal; no fallback or replay |
| Explicit incomplete terminal | HTTP `200` | Preserve the protocol's incomplete/length terminal |
| Clean provider EOF without an explicit terminal | HTTP `200` after commitment | Encode `unexpected_eof` as a non-success in-stream terminal |
| Downstream disconnect | Already-committed status if any | Record cancellation and finalize resources; do not manufacture success |

The standard Anthropic stream emits a protocol `ping` while the Router opens
and primes the provider stream. Consequently, an open/first-chunk failure after
that ping is already post-commit and is encoded in-stream. OpenAI Chat,
Responses, Gemini, and beta-native Anthropic open/prime before returning their
stream response, so their corresponding early failures can still be non-2xx.

Native OpenAI Responses status is part of the response body, not a replacement
HTTP status. Non-stream `status: "incomplete"`, `"failed"`, or `"cancelled"`
remains HTTP `200`; streaming uses `response.incomplete` or `response.failed`
(with response status `failed`/`cancelled`) and also remains HTTP `200` once
started. The Chat/Anthropic/Gemini non-stream bridges cannot represent a native
failed/cancelled Responses result, so they return their own protocol-native
error envelope rather than a false success.

---

### Path 4: Internal → Anthropic (Response Outbound, Streaming)

**Reducer:** `AnthropicReducer` in
`src/router_maestro/server/protocols/anthropic_reducer.py`

The reducer owns an `AnthropicStreamState` and translates canonical
`ChatStreamChunk` values into Anthropic SSE events:

```
OpenAI chunk: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
                ↓
Anthropic events:
  event: message_start        (first chunk only)
  event: content_block_start  (when new block begins)
  event: content_block_delta  (text_delta or input_json_delta)
  event: content_block_stop   (when block ends)
  event: message_delta        (on finish_reason, includes usage)
  event: message_stop         (terminal event)
```

**State tracking:**
- `message_start_sent` — ensures `message_start` is sent exactly once
- `content_block_index` — tracks current Anthropic content block index
- `content_block_open` — whether a block is currently open
- `tool_calls` — maps OpenAI tool_call index to Anthropic block index
- `estimated_input_tokens` — pre-calculated token estimate for `message_start`
- `last_usage` — tracks latest usage from any chunk for accurate final reporting

**Token accuracy:** Prefers actual API tokens when available; falls back to estimated tokens for `message_start` event.

---

### Path 5: Internal → Provider (Outbound to Upstream API)

Each provider translates `ChatRequest` differently:

#### GitHub Copilot (`providers/copilot.py`)
```python
payload = {
    "model": request.model,
    "messages": messages,
    "stream": True/False,
}
# Conditional fields:
if request.temperature is not None: payload["temperature"] = request.temperature
if request.max_tokens is not None: payload["max_tokens"] = ...
if request.tools: payload["tools"] = ...
if request.tool_choice: payload["tool_choice"] = ...
# Reasoning effort is resolved by the provider's outbound contract
# (CopilotOutboundContract.resolve_reasoning). See "Reasoning-Effort
# Resolution" below.
resolution = provider.outbound_contract.resolve_reasoning(
    model=request.model,
    reasoning_effort=request.reasoning_effort,
    thinking_budget=request.thinking_budget,
    catalog_effort_values=provider._catalog_effort_values(request.model),
    operation=Operation.CHAT,
)
if resolution.effort is not None:
    payload["reasoning_effort"] = resolution.effort
```

#### Anthropic Native (`providers/anthropic.py`)
- Converts OpenAI-format messages back to Anthropic format
- Extracts system message → `system` parameter
- Adds `thinking` config if `thinking_type` is enabled/adaptive
- Sends `output_config.effort` when explicit effort is present, omitting an
  adaptive budget while retaining the budget required by manual enabled thinking
- Otherwise uses the budget fallback format:
  `{"type":"enabled","budget_tokens":16000}`

The beta native Copilot route preserves a valid `output_config.effort` and
rejects unsupported siblings such as `output_config.format` with an Anthropic
`invalid_request_error` and HTTP 400 before provider or SSE commitment. Explicit
effort removes only an adaptive `thinking.budget_tokens`; manual enabled thinking
keeps its required budget before forwarding when `budget_tokens < max_tokens`.
If enabled thinking omits `budget_tokens`, the configured server default is used.
Present `max_tokens` and `thinking.budget_tokens` values must be positive,
non-boolean integers; malformed requests receive an Anthropic
`invalid_request_error` before native completion token refresh or completion I/O.
When the Copilot model catalog advertises an effort allowlist, an unavailable
request is mapped only to the highest supported tier below it. Router-Maestro
never substitutes a higher tier; if no supported tier is at or below the
request, validation returns an Anthropic HTTP 400 before provider I/O.
Effort ordering is `minimal < low < medium < high < xhigh < max`. `minimal` is
an explicit request/catalog tier with no implicit `thinking.budget_tokens`
equivalent: budget conversion still begins at `low=1024`, and smaller positive
budgets are rejected instead of being guessed as `minimal`. Unknown tier names
remain invalid rather than being treated as a request for the largest catalog
tier. Copilot catalogs may additionally advertise the provider-owned sentinel
`none`; Router-Maestro preserves it in model capability metadata so the catalog
remains usable, but `none` is not a public request tier, has no budget mapping,
and is ignored by exact/downward tier selection. An allowlist containing only
`none` therefore cannot satisfy a positive reasoning-effort request. Other
unknown catalog strings remain strict upstream-protocol errors.

##### Reasoning-Effort Resolution: catalog-first, hardcode fallback

The Copilot provider's outbound contract
(`CopilotOutboundContract.resolve_reasoning` in `providers/copilot.py`) owns the
single reasoning-effort rule for Chat, Responses, and — via the shared
`pick_closest_effort` primitive — the native beta route. It chooses one of two
paths per request, decided solely by whether `catalog_effort_values` is `None`.

**`catalog_effort_values` comes from the in-memory model cache only**
(`CopilotCatalog.effort_values`); it never blocks the request on a network
fetch. It returns a tier list when the model advertises `reasoning_effort` in
its catalog `capabilities.supports`, and returns `None` in **two** distinct
cases:

1. the catalog cache is cold (the model list has not been fetched yet), or
2. the cache is warm but that model's `supports` block carries no
   `reasoning_effort` field at all.

Both cases fall back to the hardcoded family heuristic — so "the cache is warm"
is *not* sufficient to guarantee the catalog path; the model must also advertise
the field.

**Path A — catalog advertises tiers (`effort_values` returns a list):**
`pick_closest_effort(desired, tiers)` selects the highest advertised tier not
exceeding the request. An exact match is preserved; an above-catalog request
(e.g. `max` when the catalog tops out at `high`) is downgraded to the highest
tier at or below it; if no tier is at or below the request, it is rejected with
HTTP 400 (never substituted upward). An empty advertised list means the model
declares no reasoning support, so any explicit effort/budget is rejected. This
path uses **no** family heuristic — Copilot's advertisement is authoritative, so
newly opened tiers are adopted without a code change.

**Path B — no catalog reasoning info (`effort_values` returns `None`):** the
static family heuristic `_known_reasoning_support(model)` decides support
(`claude-opus-4.{6,7,8}` / `claude-sonnet-4.6` and `gpt-5*`/`o1`/`o3`/`o4` →
supported; `claude-*-4.5` and older, `gpt-4*`, `gemini-2.5` → unsupported →
reject; otherwise unknown). Chat then applies family-specific downgrades
(`claude-*` `xhigh`/`max` → `high`; other models `max` → `xhigh`; unknown-family
`xhigh` → `high`), while Responses applies the generic `downgrade_for_upstream`
(native tiers pass through, `xhigh`/`max` → `high`). This path is the
startup-window and no-advertised-field safety net, not the steady-state path.

**Observed catalog state (live probe, 2026-07-16, 21 Copilot models):** every
reasoning-capable model advertises tiers and therefore uses Path A — for
example `claude-opus-4.6`/`sonnet-4.6` advertise `[low, medium, high, max]`,
`claude-opus-4.7/4.8`/`sonnet-5` add `xhigh`, the `gpt-5.4/5.5` family advertises
`[none, low, medium, high, xhigh]`, and `gpt-5.6-*` adds `max`. Only four models
lack the field and fall to Path B — `claude-haiku-4.5`, `claude-opus-4.5`,
`claude-sonnet-4.5`, and `gemini-2.5-pro` — and all four resolve to
`known=False`, so Path B rejects reasoning for them rather than downgrading.
Concrete downgrade examples: `claude-opus-4.6` (no `xhigh` advertised) maps a
requested `xhigh` down to `high`; `gpt-5-mini` (tops at `high`) maps `max`/`xhigh`
down to `high`; `gpt-5.6-*` (advertises `max`) preserves a requested `max`
unchanged.

Standard Anthropic routing first freezes a `RoutePlan`, then resolves thinking
budget and reasoning support separately from every candidate's immutable
capability and `max_output_tokens` snapshot. The candidate-specific transformed
requests are rebound to that same ordered pool for validation and execution, so
runtime fallback uses the limits of the model that actually receives the
request. No priority, retry limit, or provider/model identity is recomputed.

#### OpenAI / OpenAI-Compatible (`providers/openai_base.py`)
- Direct passthrough of OpenAI format
- Adds `stream_options: {"include_usage": true}` for streaming
- Converts a representable thinking budget to `reasoning_effort`
- Rejects an explicit semantic option it cannot represent instead of dropping it

At the public OpenAI Chat boundary, client `stream_options.include_usage` is a
response-encoding preference rather than a provider capability. With explicit
`true`, Router-Maestro collects upstream usage and emits one final
`choices: []` usage chunk immediately before `[DONE]`; explicit `false`
suppresses usage on the downstream stream. Omitting `stream_options` preserves
the pre-existing Router-Maestro wire shape. `stream_options` on a non-streaming
request, non-boolean `include_usage`, and unsupported nested members receive an
OpenAI-native HTTP 400 before provider selection or I/O.

---

## Option and Error Policy

| Request value | Behavior |
|---|---|---|
| Provider supports the option natively | Forward the typed value |
| Entry and provider use different names/shapes | Translate without changing semantics |
| Requested reasoning tier is unavailable | On `minimal < low < medium < high < xhigh < max`, use the highest supported tier no greater than requested, or reject if none exists |
| Unknown reasoning tier or sub-`low` token budget | Reject; do not guess a tier or increase effort |
| Semantic option cannot be represented | Return the entry protocol's native HTTP 400 error |
| Unknown provider extension | Reject; extensions cannot overwrite core payload fields |

OpenAI errors use `invalid_request_error`, Anthropic errors use an
`invalid_request_error` envelope, and Gemini errors use `INVALID_ARGUMENT`.
Static capability/option failures are non-retryable and never authorize a model
switch.

The concrete provider policy is intentionally conservative:

| Provider transport | Representative supported translations | Explicit rejection examples |
|---|---|---|
| Copilot Chat | temperature, tools/tool choice, `top_p`, penalties, stop, user, metadata, service tier, downward reasoning-tier selection | `top_k`, Gemini candidate count/response MIME type, conflicting stop fields, unknown extensions |
| Copilot Responses | tools/tool choice, parallel tools, `top_p`, metadata, service tier, downward reasoning-tier selection | any explicit temperature, unsupported tool types, unknown extensions |
| OpenAI/custom Chat Completions | OpenAI-native temperature, tools/tool choice, `top_p`, penalties, stop, user, metadata, service tier; extended reasoning tiers are downgraded to upstream `high` | `top_k`, Gemini candidate count/response MIME type, conflicting stop fields, unknown extensions |
| Anthropic Messages | temperature, tools/tool choice, `top_p`, `top_k`, stop sequences, metadata/user, service tier, thinking and effort | penalties, Gemini candidate count/response MIME type, conflicting stop fields, unknown extensions |

Model operation/feature capabilities are checked during route planning before
this payload policy runs. A fallback candidate is eligible only when its
provider transport and selected model can support the requested operation and
required tools, vision, reasoning, or parallel-tool feature (or support is
explicitly unknown and the runtime contract permits probing). An explicitly
selected model remains primary; configured priorities can follow it only after
a retryable execution failure, even when the primary was absent from the
priority list. Static option/capability errors never trigger fallback.

When a Chat-shaped request is adapted to the Copilot Responses transport, the
Responses policy also rejects `frequency_penalty`, `presence_penalty`, `stop`,
`user`, `top_k`, `stop_sequences`, Gemini candidate count, and Gemini response
MIME type. This applies whether the entry protocol was OpenAI Chat, Anthropic,
or Gemini: the rejection is encoded as that entry protocol's HTTP 400 and is
resolved before provider I/O or model fallback.

Some protocol information still has no lossless counterpart:

| Field | Boundary behavior | Reason |
|---|---|---|
| OpenAI `refusal` → Anthropic/Gemini | Mapped to text at the final protocol boundary | Neither target protocol has an equivalent refusal wire type |
| Assistant `thinking` blocks in Anthropic → legacy OpenAI history | Omitted from visible assistant content | OpenAI assistant history has no lossless slot for Anthropic replay signatures |
| Images/documents in `tool_result` | Preserved as a follow-up multimodal user message | OpenAI `tool` messages accept text only |
| `tool_reference` blocks | User message translation | Logged and skipped |

OpenAI refusal is otherwise kept distinct end to end: assistant history,
provider payloads, Chat-to-Responses conversion, Responses parsing, non-stream
output, and `response.refusal.delta` streaming events all retain the typed
value.

---

## Model Listing Endpoints

Both API flavors expose model metadata from all registered providers. Every
public `id` is `provider/model-id`, so a listed model can be sent back without
losing provider identity. Bare IDs remain accepted as convenience aliases, but
they do not select a provider.

The public encoder uses explicit catalog provenance. It always retains a raw
upstream namespace, even when that namespace equals the provider name. For
example, provider `openrouter` and raw upstream ID `openrouter/auto` produce
`openrouter/openrouter/auto`. A catalog adapter must explicitly mark a value as
already public before Router-Maestro removes one provider prefix. An unknown
provider prefix is a scoped lookup failure (`404`), not an invitation to fuzzy
match across providers.

| Endpoint | Format | Key Fields |
|---|---|---|
| `GET /api/openai/v1/models` | OpenAI `ModelList` | `id`, `owned_by`, `max_prompt_tokens`, `supports_thinking` |
| `GET /api/anthropic/v1/models` | Anthropic `ModelList` | `id`, `display_name`, `supports_thinking`, `supports_vision` |

Both call `router.list_models()`, which aggregates and de-duplicates `ModelInfo`
from all providers. Successful OpenAI Chat/Responses, Anthropic, Gemini, and
beta-native responses likewise report the qualified model that actually served
the request. If execution falls back, the response reports the fallback
candidate rather than the originally requested alias.
