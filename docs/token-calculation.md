# Token Calculation - Router-Maestro vs VS Code Copilot Chat

## Overview

Router-Maestro's token calculation is modeled after VS Code Copilot Chat's approach, with the same core formulas and constants. The main difference is Router-Maestro adds **provider-aware configuration** — different safety multipliers for Copilot, Anthropic, and OpenAI backends.

---

## Context Window Budget Formula

Both implementations use the same formula. Copilot Chat's original is in `modelMetadataFetcher.ts:146-149`; Router-Maestro's copy is in `utils/context_window.py:19-46`.

```
effective_output = min(max_output_tokens OR 4096, floor(max_prompt_tokens × 0.15))
context_window   = max_context_window_tokens OR (effective_output + max_prompt_tokens)
usable_prompt    = max(0, min(max_prompt_tokens, context_window - effective_output))
```

### Example: Claude Opus 4.6 (1M context)

| Parameter | Value |
|---|---|
| max_prompt_tokens | 1,000,000 |
| max_output_tokens | 32,000 |
| max_context_window_tokens | 1,000,000 |
| **effective_output** | min(32000, 1000000 × 0.15 = 150000) = **32,000** |
| **usable_prompt** | min(1000000, 1000000 - 32000) = **968,000** |

### Example: GPT-4o (128k context)

| Parameter | Value |
|---|---|
| max_prompt_tokens | 128,000 |
| max_output_tokens | 16,384 |
| max_context_window_tokens | 128,000 |
| **effective_output** | min(16384, 128000 × 0.15 = 19200) = **16,384** |
| **usable_prompt** | min(128000, 128000 - 16384) = **111,616** |

---

## Shared Constants

| Constant | Router-Maestro | Copilot Chat | Match |
|---|---|---|---|
| Tokens per message | 3 | 3 | ✅ |
| Tokens per name field | 1 | 1 | ✅ |
| Tokens per completion priming | 3 | 3 | ✅ |
| Base tool tokens | 16 | 16 | ✅ |
| Per-tool overhead | 8 | 8 | ✅ |
| Tool definition multiplier | 1.1× | 1.1× | ✅ |
| Tool calls multiplier | 1.5× | 1.5× | ✅ |
| Tiktoken encoding (default) | cl100k_base | cl100k_base | ✅ |
| Tiktoken encoding (o-series) | o200k_base | o200k_base | ✅ |
| Output cap | 15% of max_prompt | 15% of max_prompt | ✅ |
| Default max_output fallback | 4096 | 4096 | ✅ |

---

## Token Counting Implementation

### Router-Maestro (`utils/tokens.py`)

```python
def count_anthropic_request_tokens(system, messages, tools, model, config) -> int:
    # 1. System prompt tokens
    # 2. Per-message overhead (3 tokens each)
    # 3. Message content tokens (tiktoken encoding)
    # 4. Name field (+1 token)
    # 5. Tool calls content (× 1.5 multiplier)
    # 6. Completion priming (3 tokens)
    # 7. Tool definitions (base 16 + 8/tool + content × 1.1)
    # 8. Image tokens (OpenAI vision formula)
```

### Copilot Chat (`platform/tokenizer/node/tokenizer.ts`)

```typescript
async countMessageTokens(message): Promise<number> {
    return this.baseTokensPerMessage +
           await this.countMessageObjectTokens(toMode(OutputMode.OpenAI, message));
}

async countToolTokens(tools): Promise<number> {
    // base 16 + 8/tool + content × 1.1
}
```

**Core algorithm is identical.** Both:
1. Use tiktoken BPE tokenization
2. Recursively count nested object tokens
3. Apply same overhead constants
4. Use same safety multipliers

---

## Thinking Budget Normalization

Both clamp thinking budget to `[1024, min(32000, max_output_tokens - 1)]`.

### Router-Maestro (`utils/context_window.py:49-63`)

```python
def normalize_thinking_budget(budget, max_output_tokens, min_budget=1024, max_budget=32000):
    if budget is None: return None
    if max_output_tokens <= 1: return min_budget
    return max(min_budget, min(budget, min(max_budget, max_output_tokens - 1)))
```

### Copilot Chat (`chatEndpoint.ts:207-215`)

```typescript
_getThinkingBudget(): number | undefined {
    const configuredBudget = this._configurationService.getExperimentBasedConfig(...);
    if (!configuredBudget || configuredBudget <= 0) return undefined;
    const normalizedBudget = configuredBudget < 1024 ? 1024 : configuredBudget;
    return Math.min(32000, this._maxOutputTokens - 1, normalizedBudget);
}
```

**Same clamping logic.** Difference: Copilot Chat reads from experiment config; Router-Maestro accepts as function parameter.

---

## Key Differences

### 1. Provider-Aware Configuration (Router-Maestro only)

Router-Maestro uses different configs per provider in `utils/token_config.py`:

| Config | base_tool_tokens | tool_def_multiplier | tool_calls_multiplier |
|---|---|---|---|
| **COPILOT_CONFIG** | 16 | 1.1× | 1.5× |
| **ANTHROPIC_CONFIG** | 0 | 1.0× | 1.0× |
| **OPENAI_CONFIG** | 8 | 1.0× | 1.0× |

Copilot Chat has a single monolithic approach (always uses Copilot-inflated counts).

### 2. Anthropic Upstream Token API (Router-Maestro only)

Router-Maestro can call Anthropic's `/messages/count_tokens` endpoint for exact counts:

```python
async def count_tokens_via_anthropic_api(base_url, api_key, model, messages, system, tools) -> int:
```

Copilot Chat does not have this capability — it always uses local tiktoken estimation.

### 3. Token Reporting in Streaming

| Aspect | Router-Maestro | Copilot Chat |
|---|---|---|
| **Pre-stream estimate** | `_estimate_input_tokens()` at stream start | N/A (client-side) |
| **In-stream tracking** | `AnthropicStreamState` accumulation via `max()` | SSEProcessor accumulation |
| **Final report** | Prefers actual API tokens, falls back to estimate | Client-side only |

Router-Maestro inserts estimated `input_tokens` in the `message_start` SSE event, then reports actual tokens in the final `message_delta` event. This lets Claude Code show accurate context percentage.

### 4. O-Series Model Detection

| Router-Maestro | Copilot Chat |
|---|---|
| Regex pattern matching (`o1-`, `o3-`) | Family-based checks (`family.startsWith('o1')`) |
| Dynamic per model string | Explicit model enum (CHAT_MODEL.O1, O1MINI) |

Both select `o200k_base` encoding for O-series models.

### 5. Parity with Copilot Chat (Previously Missing, Now Implemented)

| Feature | In Copilot Chat | In Router-Maestro |
|---|---|---|
| Token counting LRU cache | ✅ 5000 entries | ✅ `@lru_cache(maxsize=5000)` on `count_tokens()` |
| Per-model token overrides (`cloneWithTokenOverride`) | ✅ | ✅ `ModelInfo.with_overrides()` + `model_overrides` config |
| Streaming chunk accumulation | ✅ SSEProcessor | ✅ `AnthropicStreamState` accumulated fields with `max()` |
| Configurable thinking budget | ✅ Config service | ✅ `ThinkingBudgetConfig` in `priorities.json` |

### 6. Added by Router-Maestro

| Feature | In Copilot Chat | In Router-Maestro |
|---|---|---|
| Provider-specific token configs | ❌ | ✅ |
| Anthropic API token counting | ❌ | ✅ |
| `max_output_tokens ≤ 1` edge case guard | ❌ | ✅ |
| Negative usable_prompt clamp (`max(0, ...)`) | ❌ | ✅ |
| Extracted reusable `normalize_thinking_budget()` | ❌ (embedded) | ✅ |

---

## Summary

Router-Maestro faithfully replicates Copilot Chat's token calculation formulas (15% cap, same constants, same multipliers). The strategic additions are:

1. **Provider-aware configs** — different multipliers for Copilot vs Anthropic vs OpenAI
2. **Anthropic upstream API** — exact token counts when available
3. **Edge case hardening** — guards for `max_output_tokens ≤ 1` and negative usable_prompt
4. **Extracted utilities** — thinking budget normalization as a standalone function

The two systems are **token-compatible** for the same models — given identical inputs and the Copilot config, they produce the same token estimates.
