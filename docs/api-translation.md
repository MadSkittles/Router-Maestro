# API Translation Layer - Router-Maestro

## Overview

Router-Maestro uses an internal hub-and-spoke architecture. OpenAI Chat,
Anthropic, and Gemini requests are normalized to a typed `ChatRequest`; OpenAI
Responses uses a typed `ResponsesRequest`. The router builds one immutable,
capability-aware route plan, providers execute the selected wire operation, and
the entry protocol encodes the result.

```
Anthropic Request ──► translate_anthropic_to_openai() ──► ChatRequest ──► Provider ──► ChatResponse ──► translate_openai_to_anthropic() ──► Anthropic Response
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

**Function:** `translate_openai_to_anthropic()` in `translation.py`

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

---

### Path 4: Internal → Anthropic (Response Outbound, Streaming)

**Function:** `translate_openai_chunk_to_anthropic_events()` in `translation.py`

Uses a state machine (`AnthropicStreamState`) to translate OpenAI streaming chunks into Anthropic SSE events:

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
apply_copilot_chat_reasoning(
    payload,
    request.model,
    request.thinking_budget,
    request.reasoning_effort,
    catalog_effort_values=...,
)
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
