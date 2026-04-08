# Changelog

All notable changes to Router-Maestro are documented here.

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
