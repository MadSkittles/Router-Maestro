# Changelog

All notable changes to Router-Maestro are documented here.

---

## v0.1.36 (2026-04-23)

### Features

- **Catalog-driven `reasoning_effort` dispatch on Copilot.** Read each model's `capabilities.supports.reasoning_effort` allowlist from Copilot's `/models` response and use it as the source of truth when picking which effort tier to send. When the desired tier isn't offered, step to the nearest available (prefer next higher, else next lower). The old hardcoded heuristic stays as a fallback when the catalog is silent. As a side effect, `gemini-3*` / `gpt-5-mini` / `gpt-5.4-mini` get correct reasoning routing without further code changes — and any future tier opened upstream (e.g. `high` on opus-4.7) will be picked up automatically.

---

## v0.1.35 (2026-04-23)

### Features

- **Claude reasoning passthrough on Copilot.** Surface upstream chain-of-thought back to Anthropic-compatible clients as `thinking` content blocks, with `thinking_delta` + `signature_delta` SSE events for streaming. Mirrors vscode-copilot-chat's field discovery (`reasoning_text` / `cot_summary` / `thinking` for text; `reasoning_opaque` / `cot_id` / `signature` for the signature). Gated behind explicit `thinking={type:"enabled"|"adaptive"}` so traces never leak to clients that didn't ask for them.

### Fixes

- **Per-Claude-family reasoning dispatch on Copilot.** `apply_copilot_chat_reasoning` now sends `reasoning_effort` (not `thinking_budget`) for `claude-opus-4.7` / `claude-opus-4.6*` / `claude-sonnet-4.6` — the Copilot gateway's actual control surface for those models. `opus-4.7` only accepts `medium` so we clamp; older models (`4.5`, `sonnet-4`, `haiku-4.5`) send neither field.
- Lower effort thresholds (`low=1024`, `medium=4096`, `high=8192`, `xhigh=16384`) so it's easier to reach the higher tiers — Copilot is free unlimited.
- Tolerate `choices=[]` with `completion_tokens>0` as a thinking-only success — but only on a reasoning-capable Claude AND when the client opted into thinking. Otherwise keep the 500 path so malformed upstream responses stay visible.
- Bump non-streaming HTTP read timeout 120s → 240s. `claude-opus-4.6` / `claude-sonnet-4.6` at high effort routinely take >2min on Copilot's side.

### Docs

- Drop hardcoded production domain from `CLAUDE.md`.

---

## v0.1.34 (2026-04-23)

### Fixes

- Per-model reasoning dispatch on Copilot's `/chat/completions` endpoint
  - Previously the chat path blindly forwarded `thinking_budget`, which the Copilot gateway rejects with `400 invalid_thinking_budget` for OpenAI reasoning models (`gpt-5*`, `o1`, `o3`, `o4`). This broke clients that send Anthropic-style `thinking` (e.g. Cherry Studio → `gpt-5.x`).
  - New `apply_copilot_chat_reasoning()` routes by model family: Claude keeps `thinking_budget`, GPT-5 / o-series get `reasoning_effort` (with `xhigh` preserved natively — Copilot accepts it), GPT-4 / Gemini omit both fields.
  - `gpt-5.4*` requires `max_completion_tokens` instead of `max_tokens`; the helper rewrites the field automatically.

### Logging

- Copilot chat / streaming now log `thinking_budget` and `reasoning_effort` (both incoming `ChatRequest` values and the resolved outbound payload values) at DEBUG, so operators can verify what was actually sent to the gateway.

---

## v0.1.33 (2026-04-20)

### Chores

- Re-release of v0.1.32 after a PyPI upload conflict left v0.1.32 unpublishable (no functional change)
- Apply `ruff format` to `providers/copilot.py` and `tests/test_reasoning_effort.py` ([#46](https://github.com/MadSkittles/Router-Maestro/pull/46))

---

## v0.1.32 (2026-04-20)

### Features

- End-to-end `reasoning_effort` / `thinking` passthrough across all entrypoints ([#44](https://github.com/MadSkittles/Router-Maestro/pull/44))
  - OpenAI Chat (`/api/openai/v1/chat/completions`), Responses (`/api/openai/v1/responses`), and Anthropic Messages now all accept and forward reasoning intensity
  - New `xhigh` extension level (24000 budget); auto-downgrades to `high` for OpenAI/Copilot upstreams that reject it
  - New shared `utils/reasoning.py` module with `effort_to_budget` / `budget_to_effort` mapping
  - OpenAI native provider now writes `reasoning_effort` into the upstream payload (previously dropped)

### Fixes

- Drop Codex CLI `namespace` tools before sending to Copilot ([#45](https://github.com/MadSkittles/Router-Maestro/pull/45))
  - Codex CLI groups MCP servers as `tools[].type="namespace"` entries; Copilot's Responses API rejects these with `Missing required parameter: 'tools[N].tools'`, breaking every Codex turn
  - Add `"namespace"` to `CopilotProvider.UNSUPPORTED_TOOL_TYPES` so the existing filter strips them
  - Side effect: Codex's MCP servers are not exposed through Copilot until a real MCP proxy is added

---

## v0.1.31 (2026-04-20)

### Fixes

- Accept Anthropic `document` content blocks (e.g. PDF attachments) in user messages and `tool_result` content ([#43](https://github.com/MadSkittles/Router-Maestro/pull/43))
  - Add `AnthropicDocumentBlock` / `AnthropicDocumentSource` to the user-content and tool_result-content unions, fixing a 422 `body.messages.*.content.str: Input should be a valid string` when Claude Code sent PDFs
  - Translate document blocks through `_extract_multimodal_content` in Anthropic-native shape so `AnthropicProvider` forwards them upstream verbatim
  - Documents nested inside `tool_result.content` are injected as a follow-up user message, mirroring the existing image behaviour

---

## v0.1.30 (2026-04-08)

### Fixes

- Suppress `KeyboardInterrupt` traceback when pressing Ctrl+C during CLI startup ([#42](https://github.com/MadSkittles/Router-Maestro/pull/42))
  - Add lightweight `cli/entry.py` wrapper with lazy import inside `try/except KeyboardInterrupt`
  - Ctrl+C during module import now exits cleanly with code 130 instead of dumping a full stack trace

---

## v0.1.29 (2026-04-02)

### Fixes

- Passthrough images from Anthropic `tool_result` to OpenAI format instead of silently dropping them ([#41](https://github.com/MadSkittles/Router-Maestro/pull/41))
  - Images in tool results are now extracted and injected as a follow-up user message with OpenAI multimodal `image_url` format
  - All images from multiple tool results are collected and appended after all tool messages to avoid interleaved `tool`/`user` message sequences that OpenAI rejects

---

## v0.1.28 (2026-03-31)

### Fixes

- Rename 1M option display name to "Opus 4.6 1M (Auto-activated)" to better distinguish from the internal provider model key ([#40](https://github.com/MadSkittles/Router-Maestro/pull/40))

---

## v0.1.27 (2026-03-31)

### Fixes

- Prepend `claude-opus-4-6[1m]` option at the top of the model list instead of the bottom ([#39](https://github.com/MadSkittles/Router-Maestro/pull/39))

---

## v0.1.26 (2026-03-31)

### Features

- Add Opus 4.6 1M context option to `config claude-code` wizard ([#38](https://github.com/MadSkittles/Router-Maestro/pull/38))
  - When `github-copilot/claude-opus-4.6-1m` is available, offers `claude-opus-4-6[1m]` as a selectable model that activates Claude Code's native 1M context window
  - Extracted `_fetch_models`, `_display_models`, and `_maybe_inject_opus_1m` for better testability
  - Out-of-range model selection now warns instead of silently falling back to auto-routing

### Documentation

- Highlight 1M context support and fuzzy model matching in README features
- Add ripgrep preference note to CLAUDE.md

---

## v0.1.25 (2026-03-31)

### Bug Fixes

- Fix 500 Internal Server Error caused by lone UTF-16 surrogate characters in request messages ([#37](https://github.com/MadSkittles/Router-Maestro/pull/37))
  - Sanitize message content in CopilotProvider to replace lone surrogates (e.g. `\udc8d`) before httpx JSON serialization

### Documentation

- Replace ASCII architecture diagram with Mermaid ([#36](https://github.com/MadSkittles/Router-Maestro/pull/36))
- Rewrite README deployment section for clarity ([#35](https://github.com/MadSkittles/Router-Maestro/pull/35))

---

## v0.1.24 (2026-03-15)

### Bug Fixes

- Fix streaming tool call matching to use `output_index` instead of `item_id`
  - Copilot obfuscates/encrypts item IDs differently across SSE events in the same stream, making ID-based matching impossible
  - `output_index` is consistent across `output_item.added`, `arguments.delta`, `arguments.done`, and `output_item.done` events

---

## v0.1.23 (2026-03-15)

### Bug Fixes

- Fix parallel tool call state corruption in Responses API streaming ([#34](https://github.com/MadSkittles/Router-Maestro/pull/34))
  - Replace single-state `current_fc` tracker with `pending_fcs` dict keyed by item ID, so concurrent tool calls are tracked independently
- Fix duplicate/orphaned delta events with mismatched IDs in streaming tool calls ([#34](https://github.com/MadSkittles/Router-Maestro/pull/34))
  - Remove `tool_call_delta` emission from copilot provider and dead handler in responses route; the complete tool call path already reconstructs the full SSE event sequence

---

## v0.1.22 (2026-03-15)

### Features

- Add `CLAUDE_CODE_ENABLE_LSP` environment variable to `config claude-code` generator ([#33](https://github.com/MadSkittles/Router-Maestro/pull/33))
  - Enables LSP support by default in generated Claude Code settings

---

## v0.1.21 (2026-03-15)

### Bug Fixes

- Add SOCKS proxy support to httpx dependency ([#32](https://github.com/MadSkittles/Router-Maestro/pull/32))
  - Changed `httpx` to `httpx[socks]` to include the `socksio` package, fixing CLI failures when a SOCKS proxy is configured via `ALL_PROXY` environment variable

### Documentation

- Add `tool_choice` finish_reason behavior analysis and diagnostic script ([#31](https://github.com/MadSkittles/Router-Maestro/pull/31))
- Enforce branch check before making changes in CLAUDE.md

---

## v0.1.20 (2026-03-09)

### Bug Fixes

- Fix Gemini tool call translation — non-streaming args + streaming premature emit ([#30](https://github.com/MadSkittles/Router-Maestro/pull/30))

---

## v0.1.19 (2026-03-09)

### Features

- Add Gemini-compatible API routes ([#26](https://github.com/MadSkittles/Router-Maestro/pull/26))
- Add tool calling support to OpenAI-compatible endpoint
- Add Gemini CLI config command with model selection

### Bug Fixes

- Merge tool_calls from all Copilot response choices (fixes multi-choice handling)
- Use multiarch builder for multi-platform Docker builds
- Rename gemini-cli to gemini, strip provider prefix from model
- Ruff format fixes for Gemini route files

### Documentation & Tests

- Expand CLAUDE.md with full project structure and key concepts
- Update README with Gemini support
- Add auth and tool parameter tests

---

## v0.1.18 (2026-03-06)

### Bug Fixes

- Recover tool calls from XML content when provider misplaces them ([#28](https://github.com/MadSkittles/Router-Maestro/pull/28))
  - GitHub Copilot API sometimes returns `finish_reason="tool_calls"` but embeds tool calls as `<tool_call>` XML in `message.content` instead of the proper `message.tool_calls` field
  - Add shared recovery utility (`tool_parsing.py`) that detects and extracts structured tool calls from XML content
  - Integrate into `CopilotProvider` and `OpenAIChatProvider` non-streaming paths
  - Add debug logging in Anthropic route for production diagnosis

---

## v0.1.17 (2026-03-06)

### Bug Fixes

- Fix 500 error when Anthropic API requests include `tools` parameter ([#27](https://github.com/MadSkittles/Router-Maestro/pull/27))
  - Add `tool_calls` field to `ChatResponse` and allow `content` to be `None`
  - Forward `tools`/`tool_choice` in OpenAI base and Anthropic native provider payloads
  - Convert OpenAI-format `tool_calls` back to Anthropic `tool_use` blocks in both streaming and non-streaming responses
  - Fix JSON string arguments not being parsed in `translate_openai_to_anthropic`

### Documentation

- Rename `RELEASE_NOTES.md` to `CHANGELOG.md`

---

## v0.1.16 (2026-03-05)

### Features

- Auto-route to 1m model variant when `context-1m` beta header is detected ([#25](https://github.com/MadSkittles/Router-Maestro/pull/25))
  - Claude CLI sends `anthropic-beta: context-1m-*` when user selects `[1m]` model variant
  - Automatically rewrites model ID to the `-1m` variant (e.g. `claude-opus-4-6` → `claude-opus-4.6-1m`) when available in provider cache
  - Add `find_extended_context_variant()` utility with normalized matching for dot/hyphen differences

---

## v0.1.15 (2026-03-02)

### Bug Fixes

- Use fresh httpx client for CopilotProvider non-streaming operations, resolving admin endpoint hangs under concurrent streaming load ([#23](https://github.com/MadSkittles/Router-Maestro/pull/23))

### Refactoring

- Migrate to `StrEnum` instead of `(str, Enum)` pattern for cleaner enum definitions

### Documentation

- Add API translation layer documentation with detailed translation paths and message handling

---

## v0.1.14 (2026-02-13)

### Features

- Add thinking passthrough and model metadata for Opus 4.6 ([#22](https://github.com/MadSkittles/Router-Maestro/pull/22))
- Add token counting cache, model overrides, streaming accumulation, and thinking budget config
- Add `docker-compose.dev.yml` for local source builds
- Centralize httpx timeout handling with shared constants ([#20](https://github.com/MadSkittles/Router-Maestro/pull/20))

### Bug Fixes

- Enable HTTP/2 and optimize connection pool for Copilot provider ([#18](https://github.com/MadSkittles/Router-Maestro/pull/18))
- Add SSE keepalive heartbeats to prevent silent streaming timeouts
- Use fine-grained httpx timeout for Copilot HTTP client ([#17](https://github.com/MadSkittles/Router-Maestro/pull/17))
- Increase Copilot timeout to 600s and improve stream error logging ([#16](https://github.com/MadSkittles/Router-Maestro/pull/16))
- Address code review findings from refactoring

### Refactoring

- Extract TTLCache utility to deduplicate cache patterns ([#19](https://github.com/MadSkittles/Router-Maestro/pull/19))
- Consolidate HTTP error handling in providers
- Remove dead code and legacy aliases from tokens module
- Extract block field helpers to simplify `translation.py`
- Extract message close helper in responses streaming
- Deduplicate CLI config commands
- Simplify `tool_result` content parsing in translation
- Simplify streaming code and deduplicate SSE error handling

---

## v0.1.13 (2026-02-11)

### Features

- Add fuzzy model ID matching with `rapidfuzz` ([#15](https://github.com/MadSkittles/Router-Maestro/pull/15))
- Add provider-aware token counting configuration ([#11](https://github.com/MadSkittles/Router-Maestro/pull/11))

### Bug Fixes

- Expand `AnthropicThinkingConfig` type to include `adaptive` and `disabled` variants

### Documentation

- Document fuzzy model ID matching in README
- Embed quick start demo video in README

---

## v0.1.12 (2026-02-07)

### Features

- Align token counting with VS Code Copilot Chat for accurate estimation ([#10](https://github.com/MadSkittles/Router-Maestro/pull/10))
- Sort non-priority models in `list_models` by provider, family, and version ([#9](https://github.com/MadSkittles/Router-Maestro/pull/9))

### Refactoring

- Replace character-based token estimation with tiktoken ([#6](https://github.com/MadSkittles/Router-Maestro/pull/6))
- Support dict inputs in token counting functions ([#8](https://github.com/MadSkittles/Router-Maestro/pull/8))

### CI

- Ensure releases only happen from master branch

---

## v0.1.11 (2026-02-05)

### Features

- Integrate tiktoken for accurate token counting ([#5](https://github.com/MadSkittles/Router-Maestro/pull/5))
- Add model-specific token calibration from agent-maestro

### Bug Fixes

- Improve token estimation accuracy with centralized function ([#3](https://github.com/MadSkittles/Router-Maestro/pull/3))

### Documentation

- Add GitHub MCP tools preference to CLAUDE.md

---

## v0.1.10 (2026-02-05)

### Bug Fixes

- Handle `tool_use` and `tool_result` blocks in token counting ([#2](https://github.com/MadSkittles/Router-Maestro/pull/2))

---

## v0.1.9 (2026-02-04)

### Bug Fixes

- Handle `tool_reference` content blocks in Anthropic API
- Merge env config instead of replacing to preserve user variables ([#1](https://github.com/MadSkittles/Router-Maestro/pull/1))

### Refactoring

- Extract shared OpenAI-compatible chat logic into base class
- Move inline imports to module level and improve test coverage

---

## v0.1.8 (2026-02-03)

### Bug Fixes

- Prevent duplicate tool call emissions in Responses API streaming

### Documentation

- Add version update instructions to CLAUDE.md

---

## v0.1.7 (2026-02-03)

### Features

- Add OpenAI Responses API support for Codex models

### Bug Fixes

- Use `isinstance` for type narrowing in credential check

### Documentation

- Add Codex CLI configuration to README

---

## v0.1.6 (2026-02-02)

### Features

- Add Anthropic-compatible `/api/anthropic/v1/models` endpoint
- Add test model to Anthropic Messages API
- Add GitHub Actions CI and release workflows

### Bug Fixes

- Fix config command to preserve existing settings
- Fix Pylance type warnings in Anthropic routes
- Fix lint errors in tests: remove unused imports, sort imports

---

## v0.1.5 (2026-02-02)

_Note: v0.1.5 changes were included as part of the v0.1.6 tag._

### Features

- Add GitHub Actions CI and release workflows
- Add interactive config command

### Bug Fixes

- Preserve existing settings in config command
- Fix lint and type errors

---

## v0.1.4 (2026-01-29)

### Features

- Add Docker quick start guide
- Document contexts concept for remote VPS Docker deployments
- Add interactive config command

### Bug Fixes

- Fix streaming type error and improve markdown formatting
- Fix Docker run command to match `docker-compose.yml`
- Fix markdown lint issues in README

---

## v0.1.3 (2026-01-29)

### Breaking Changes

- Remove stats tracking feature

---

## v0.1.2 (2026-01-29)

_Initial public release._

### Features

- Multi-provider routing with priority-based model selection
- OpenAI-compatible and Anthropic-compatible API endpoints
- GitHub Copilot, OpenAI, Anthropic, and custom provider support
- Typer-based CLI with subcommands: `server`, `auth`, `model`, `context`, `config`
- Docker support with multi-arch builds
- Config hot-reload for live configuration updates
- Fallback CLI commands for improved routing documentation
- Context-aware server status
- Comprehensive VPS deployment and Claude Code integration docs
