# API Translation Layer - Router-Maestro

## Overview

Router-Maestro uses an internal "hub-and-spoke" architecture: all incoming requests (Anthropic or OpenAI format) are translated to an internal `ChatRequest` format, routed to providers, and responses are translated back if needed.

```
Anthropic Request ──► translate_anthropic_to_openai() ──► ChatRequest ──► Provider ──► ChatResponse ──► translate_openai_to_anthropic() ──► Anthropic Response
OpenAI Request ──────────────── (passthrough) ──────────► ChatRequest ──► Provider ──► ChatResponse ──────────── (passthrough) ──────────────► OpenAI Response
```

---

## Translation Paths

### Path 1: Anthropic → Internal (Request Inbound)

**Function:** `translate_anthropic_to_openai()` in `src/router_maestro/server/translation.py:35-65`

| Anthropic Field | Internal (ChatRequest) Field | Transformation |
|---|---|---|
| `model` | `model` | Date suffixes stripped, hyphens→dots (`claude-haiku-4-5-20251001` → `claude-haiku-4.5`) |
| `messages` | `messages` | Content blocks → role-based messages (see below) |
| `system` | System `Message` | String or TextBlock list → single system message, sanitized |
| `max_tokens` | `max_tokens` | Passthrough |
| `temperature` | `temperature` | Passthrough |
| `stream` | `stream` | Passthrough |
| `tools` | `tools` | `input_schema` → `parameters`, wrapped in `{"type":"function","function":{...}}` |
| `tool_choice` | `tool_choice` | `auto`→`"auto"`, `any`→`"required"`, `tool`→`{"type":"function",...}` |
| `thinking.type` | `thinking_type` | Extracted directly |
| `thinking.budget_tokens` | `thinking_budget` | Extracted directly |

#### Message Translation Details

**User messages** (`_handle_user_message`, line 203):
- String content → `Message(role="user", content=text)`
- Content blocks are split:
  - `tool_result` → `Message(role="tool", tool_call_id=...)`
  - `text` → included in user message
  - `image` (base64) → OpenAI `image_url` format
  - `thinking` → treated as text
  - `tool_reference` → skipped (logged)

**Assistant messages** (`_handle_assistant_message`, line 267):
- `text` + `thinking` blocks → concatenated into `content`
- `tool_use` blocks → OpenAI `tool_calls` format:
  ```
  Anthropic: {"type":"tool_use","id":"...","name":"...","input":{...}}
  OpenAI:    {"id":"...","type":"function","function":{"name":"...","arguments":"JSON string"}}
  ```

**System prompt** (`_sanitize_system_prompt`, line 163):
- Removes `x-anthropic-billing-header` lines (Claude Code adds these; Copilot rejects them)

#### Model Name Translation (`_translate_model_name`, line 68)

| Input | Output | Rule |
|---|---|---|
| `claude-sonnet-4-20250514` | `claude-sonnet-4` | Strip date suffix |
| `claude-haiku-4-5-20251001` | `claude-haiku-4.5` | Hyphen version → dot, strip date |
| `claude-opus-4.6` | `claude-opus-4.6` | No change |

---

### Path 2: Internal → Anthropic (Response Outbound, Non-Streaming)

**Function:** `translate_openai_to_anthropic()` in `translation.py:368-417`

| Internal / OpenAI Field | Anthropic Field | Transformation |
|---|---|---|
| `choices[0].message.content` | `content[]` (TextBlock) | Wrapped in `{"type":"text","text":"..."}` |
| `choices[0].message.tool_calls` | `content[]` (ToolUseBlock) | Each becomes `{"type":"tool_use","id":"...","name":"...","input":{}}` |
| `choices[0].finish_reason` | `stop_reason` | `stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use` |
| `usage.prompt_tokens` | `usage.input_tokens` | Renamed |
| `usage.completion_tokens` | `usage.output_tokens` | Renamed |

---

### Path 3: OpenAI → Internal (Request Inbound, Passthrough)

**File:** `src/router_maestro/server/routes/chat.py:31-79`

OpenAI requests map almost directly to `ChatRequest`. Basic field mapping only:

```python
ChatRequest(
    model=request.model,
    messages=[Message(role=m.role, content=m.content) for m in request.messages],
    temperature=request.temperature,
    max_tokens=request.max_tokens,
    stream=request.stream,
)
```

**Fields NOT forwarded:** tools, tool_choice, thinking config, top_p, frequency/presence_penalty, stop sequences.

---

### Path 4: Internal → Anthropic (Response Outbound, Streaming)

**Function:** `translate_openai_chunk_to_anthropic_events()` in `translation.py:420-606`

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
    "temperature": request.temperature,
    "stream": True/False,
}
# Conditional fields:
if request.max_tokens: payload["max_tokens"] = ...
if request.tools: payload["tools"] = ...
if request.tool_choice: payload["tool_choice"] = ...
if request.thinking_budget is not None: payload["thinking_budget"] = ...
```

#### Anthropic Native (`providers/anthropic.py`)
- Converts OpenAI-format messages back to Anthropic format
- Extracts system message → `system` parameter
- Adds `thinking` config if `thinking_type` is enabled/adaptive
- Uses Anthropic API format: `{"type":"enabled","budget_tokens":16000}`

#### OpenAI / OpenAI-Compatible (`providers/openai_base.py`)
- Direct passthrough of OpenAI format
- Adds `stream_options: {"include_usage": true}` for streaming
- Does NOT forward `thinking_budget` (not supported by OpenAI API)

---

## What Gets Lost in Translation

| Field | Where Lost | Reason |
|---|---|---|
| `top_p`, `top_k` | Anthropic → Internal | Not in `ChatRequest` |
| `stop_sequences` | Anthropic → Internal | Not in `ChatRequest` |
| `metadata` | Anthropic → Internal | Not passed through |
| `thinking` block content | Streaming response | Merged into text |
| Images in `tool_result` | User message translation | Logged and skipped |
| `tool_reference` blocks | User message translation | Logged and skipped |

---

## Model Listing Endpoints

Both API flavors expose model metadata from all registered providers:

| Endpoint | Format | Key Fields |
|---|---|---|
| `GET /api/openai/v1/models` | OpenAI `ModelList` | `id`, `owned_by`, `max_prompt_tokens`, `supports_thinking` |
| `GET /api/anthropic/v1/models` | Anthropic `ModelList` | `id`, `display_name`, `supports_thinking`, `supports_vision` |

Both call `router.list_models()` which aggregates `ModelInfo` from all providers.
