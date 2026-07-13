# Router-Maestro Maintainability Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed cross-protocol correctness defects first, then make routing, provider capabilities, request contracts, streaming state, credentials, configuration, and request lifecycle explicit enough that future protocol and model changes have one authoritative implementation.

**Architecture:** Preserve the existing route/translation → Router → provider structure, but strengthen the seams. Introduce small typed domain concepts and capability-aware route planning before splitting large modules. Move protocol state machines and lifecycle ownership behind explicit interfaces; do not rewrite the service, introduce microservices, or replace FastAPI/Pydantic.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, dataclasses, httpx, pytest/pytest-asyncio, Ruff, uv.

---

## Status and review gate

This document is the approved implementation plan. Decisions A–C were approved on 2026-07-11; Decision D and Task 19 remain deferred pending a separate compatibility discussion.

- Baseline reviewed: `6cfcb6658f9d60494708db4b1f78a819677ca765` (`v0.5.4`).
- Phase 0 working branch: `fix/codebase-review-phase-0`.
- Phase 0 implementation and independent final review completed on 2026-07-11.
- Phase 0 verification: `1330 passed`; Ruff, formatting, and `git diff --check` pass.
- Phase 0 unbounded live Copilot integration: `66 passed, 1 skipped` in 15:04.
- Phase 1 Tasks 6–10 implementation and final independent review completed on
  2026-07-14. The final specification/scope gate reported
  `0 Critical / 0 Important / 0 Minor`; the engineering-quality gate reported
  `0 Critical / 0 Important` and retained three non-blocking consolidation
  observations for Phase 2 (shared stream priming, prepared Chat snapshot
  binding, and Chat-to-Responses transport-policy ownership).
- Phase 1 full unit verification: `3002 passed` in 7.08s. Ruff lint and format
  checks pass for `src/`, `tests/`, and `integration_tests/`; `git diff --check`
  passes.
- Phase 1 unbounded live Copilot integration: `73 passed, 3 skipped` in 24:06.
  No model-matrix limiting environment variable was set.
- Phase 1 local acceptance gates are complete. Landing through the Phase 1
  pull request is pending.
- Tasks may proceed phase-by-phase except deferred Task 19.
- Each phase lands through its own pull request and leaves the full unit suite green.
- Before creating every phase PR, run `uv run pytest tests/ -v`, both Ruff checks, and the unbounded `make integration-test`. A bounded model matrix is useful during development but never satisfies the phase gate.
- Create the PR only after all local gates pass. Merge only after the PR checks pass, then branch the next phase from the updated `master`.
- Do not tag, publish, or otherwise create a release as part of this remediation.

## Guiding constraints

1. Preserve the current public API unless a decision is explicitly called out below.
2. Add characterization or failing tests before behavior changes.
3. Prefer one authoritative business rule over route-specific copies.
4. Do not create a universal protocol AST in one step. Add only the typed concepts required by confirmed defects.
5. Do not split a file because it is long. Split only when responsibilities and tests can move behind a stable interface.
6. Keep Copilot compatibility workarounds at the Copilot boundary.
7. Do not add Redis, a database, a file watcher, or multi-worker coordination in this plan.
8. Do not commit automatically during execution unless the user explicitly authorizes commits. The commit boundaries below are recommendations only.

## Decisions required before implementation

Decisions A–C use the recommended policies below. Decision D is unresolved and blocks Task 19 only; no admin/inference authentication behavior may change until the user approves a transition strategy.

### Decision A: explicit-model priority fallback (APPROVED)

**Recommended:** If the current explicit `provider/model` is absent from `priorities`, use the complete priority list as fallback candidates after the explicit primary fails. Remove duplicates of the primary.

Alternative: retain the current behavior and document that priority fallback only applies when the primary appears in the list. This is less consistent with the current automatic-fallback promise.

### Decision B: public model identifiers (APPROVED)

**Recommended:** Return `provider/model-id` as the unique public ID from model-list endpoints. Continue accepting bare model IDs as convenience aliases, but document that aliases do not select a provider.

Alternative: retain duplicate bare IDs and add a separate provider-qualified field. This is more backward-compatible but leaves clients unable to round-trip an unambiguous selection using the standard `id` field.

### Decision C: unsupported request options (APPROVED)

**Recommended:** Reject an explicitly requested, semantically important unsupported option with a protocol-native 400. Permit an observable reasoning-tier substitution only when the exact tier is unavailable: choose the highest supported tier that does not exceed the requested tier. If no such tier exists, reject the request. Never silently increase reasoning effort, cost, or latency.

This intentionally changes the current `pick_closest_effort()` higher-first behavior, which can map a request upward. The compatibility alternative is to retain nearest-tier selection, including a higher tier, but surface the substitution in the response metadata/logs and document the cost/latency consequence. Silently omitting an unsupported option is not an accepted policy.

### Decision D: administrator authentication compatibility (PENDING; TASK 19 DEFERRED)

**Recommended:** Add `ROUTER_MAESTRO_ADMIN_API_KEY`. When absent, temporarily fall back to the inference API key and log a startup warning; document a later removal window.

Alternative: require the new key immediately. This is safer but breaking for existing remote administration setups.

## Target module boundaries

The final shape should evolve toward the following responsibilities. Names may be adjusted during implementation, but responsibilities must remain separate.

```text
server/routes/*
  Parse protocol request, invoke Router, encode protocol response/error.

server/protocols/*
  Protocol-specific encoders/reducers for OpenAI Chat, Responses,
  Anthropic, and Gemini. No provider selection or credentials.

routing/model_ref.py
  Unique provider/model identity.

routing/capabilities.py
  Provider-level operation support plus model-level feature/operation support.

routing/route_plan.py
  Primary and fallback candidates selected before execution.

routing/router.py
  Public facade coordinating registry, catalog, planning, and execution.

providers/*
  Provider transport and wire codecs. Provider-specific compatibility stays here.

runtime/request_context.py
  Request ID, config snapshot, audit lifecycle, stream commit state, and terminal outcome.

config/repository.py
  Immutable runtime snapshots, content revisions, and compare-and-swap mutation.

auth/repository.py
  Read-latest, patch-one-provider, atomic credential persistence.
```

---

## Phase 0: Correctness and contract characterization

### Task 1: Reject malformed beta-native thinking values instead of returning 500

**Why:** The v0.5.4 enabled-thinking fix now passes raw beta JSON budgets into numeric normalization. A string/list/dict budget raises `TypeError` and becomes an internal 500.

**Files:**
- Modify: `src/router_maestro/server/routes/anthropic_beta.py`
- Reuse or modify: `src/router_maestro/server/schemas/anthropic.py`
- Test: `tests/test_anthropic_beta.py`

- [x] **Step 1: Add endpoint-level failing tests**

  Parameterize an explicitly supplied `thinking.budget_tokens` with a string, list, dictionary, boolean, fractional float, zero, and negative integer. Use `TestClient(..., raise_server_exceptions=False)` and the real budget resolver. Assert a protocol-native client error and that `ensure_token`/upstream send are not called. Add a positive characterization case for `thinking.type="enabled"` with no `budget_tokens`; it must continue using the configured server default.

- [x] **Step 2: Run the focused test and confirm current failure**

  Run: `uv run pytest tests/test_anthropic_beta.py -k 'invalid_budget or invalid_max_tokens' -v`

  Expected before implementation: the string/list/dict cases return 500; some other invalid numeric shapes are accepted.

- [x] **Step 3: Add strict boundary validation**

  Validate `max_tokens` and the `thinking` discriminated union before `_apply_thinking_budget_native()`. When present, require `budget_tokens` and `max_tokens` to be positive non-boolean integers; omission of `budget_tokens` remains valid and selects the server default. Return an Anthropic error envelope such as:

  ```json
  {
    "type": "error",
    "error": {
      "type": "invalid_request_error",
      "message": "thinking.budget_tokens must be a positive integer"
    }
  }
  ```

- [x] **Step 4: Add the `max_tokens=1024/1025` boundary test**

  Confirm that 1024 cannot carry an enabled 1024-token budget and 1025 can. Keep the implementation rule `budget_tokens < max_tokens` explicit.

- [x] **Step 5: Add a corrective unreleased changelog entry and update current docs**

  Modify `README.md`, `docs/api-translation.md`, and `docs/token-calculation.md` to state that enabled budgets are retained when sufficient output-token headroom exists. Add the clarification under a new unreleased/next-release section in `CHANGELOG.md`; do not rewrite the historical v0.5.4 entry unless the user explicitly approves that editorial change.

- [x] **Step 6: Verify**

  Run:

  ```bash
  uv run pytest tests/test_anthropic_beta.py tests/test_thinking_budget_config.py tests/test_thinking_passthrough.py -v
  uv run ruff check src/ tests/
  ```

**Suggested commit boundary:** `fix: validate beta-native thinking budgets`

### Task 2: Establish one terminal invariant for all streaming protocols

**Why:** An upstream EOF without an explicit terminal chunk is treated as success by Chat/Responses and as an unterminated stream by Anthropic/Gemini.

**Files:**
- Modify: `src/router_maestro/providers/base.py`
- Modify: `src/router_maestro/server/routes/chat.py`
- Modify: `src/router_maestro/server/routes/responses.py`
- Modify: `src/router_maestro/server/routes/anthropic.py`
- Modify: `src/router_maestro/server/routes/gemini.py`
- Modify as needed: `src/router_maestro/server/translation.py`
- Modify as needed: `src/router_maestro/server/translation_gemini.py`
- Test: `tests/test_chat_stream_done_sentinel.py`
- Test: `tests/test_responses_route_wire_shape.py`
- Test: `tests/test_translation_advanced.py`
- Test: `tests/test_gemini_routes.py`
- Create: `tests/test_stream_terminal_invariants.py`

- [x] **Step 1: Add a shared characterization matrix**

  For each protocol, feed one content chunk followed by EOF with no `finish_reason`. Assert that the result is not a successful terminal event. Cover explicit success, explicit incomplete/failure, upstream exception, and client cancellation separately from EOF. For a post-commit failure, assert the HTTP status remains 200, the protocol emits an in-stream error/incomplete terminal, and the internal request outcome is non-success.

- [x] **Step 2: Run the matrix and record current behavior**

  Run: `uv run pytest tests/test_stream_terminal_invariants.py -v`

  Expected before implementation: Chat emits `[DONE]`; Responses emits `response.completed`; Anthropic/Gemini lack their normal terminal events while their pipeline records success.

- [x] **Step 3: Separate transport termination from response semantics**

  Introduce orthogonal typed fields in `providers/base.py` rather than one overloaded enum:

  - `TransportTermination`: `EXPLICIT_TERMINAL`, `UNEXPECTED_EOF`, `CLIENT_CANCELLED`, `EXCEPTION`.
  - `ResponseStatus`: `COMPLETED`, `INCOMPLETE`, `FAILED`, `CANCELLED`, `UNKNOWN`.
  - A terminal outcome carrying both fields plus optional `finish_reason`, `incomplete_details`, and typed error.

  Do not yet redesign content blocks. A legal Responses `incomplete` terminal is therefore distinct from an in-progress stream that ends at EOF.

- [x] **Step 4: Make each route encode unexpected EOF as a protocol error/incomplete result**

  Preserve each protocol's framing, but share the success invariant. This task's characterization emits content before EOF, so the stream is already committed: the wire HTTP status necessarily remains 200; emit only the protocol's in-stream error/incomplete terminal and record `unexpected_eof` in the internal outcome/metrics label. Do not store the internal result in a field named HTTP status. Pre-commit provider-open/first-chunk priming and native non-2xx responses are deliberately implemented once in Task 13; do not add a temporary priming layer here.

- [x] **Step 5: Verify focused and full suites**

  Run:

  ```bash
  uv run pytest tests/test_stream_terminal_invariants.py tests/test_chat_stream_done_sentinel.py tests/test_responses_route_wire_shape.py tests/test_gemini_routes.py tests/test_translation_advanced.py -v
  uv run pytest tests/ -q
  ```

**Suggested commit boundary:** `fix: reject unterminated provider streams`

### Task 3: Fix Responses item-state isolation

**Why:** Closing a message does not reset `accumulated_content`, so `text A → reasoning R → text B` emits the second message as `AB`.

**Files:**
- Modify: `src/router_maestro/server/routes/responses.py`
- Test: `tests/test_responses_route_wire_shape.py`
- Test: `tests/test_streaming_accumulation.py`

- [x] **Step 1: Add a failing interleaving test**

  Exercise `text → reasoning → text → terminal` and assert output items `message("A"), reasoning("R"), message("B")`.

- [x] **Step 2: Run and confirm the second message is currently `AB`**

  Run: `uv run pytest tests/test_responses_route_wire_shape.py -k text_reasoning_text -v`

- [x] **Step 3: Reset per-message state on close**

  Clear message-specific accumulator and content index when `close_open_message()` finishes. Do not clear reasoning state there.

- [x] **Step 4: Add adjacent interleavings**

  Cover `reasoning → text`, `text → tool → text`, empty reasoning, and consecutive messages to prevent another shared-accumulator regression.

- [x] **Step 5: Verify**

  Run: `uv run pytest tests/test_responses_route_wire_shape.py tests/test_streaming_accumulation.py -v`

**Suggested commit boundary:** `fix: isolate Responses streaming item state`

### Task 4: Make Anthropic parallel tool streaming legal

**Why:** Interleaved OpenAI tool deltas can target an Anthropic block after that block has already emitted `content_block_stop`.

**Files:**
- Modify: `src/router_maestro/server/translation.py`
- Test: `tests/test_translation_advanced.py`
- Test: `tests/test_anthropic_provider_stream.py`

- [x] **Step 1: Add the failing interleaved two-tool sequence**

  Assert that no `content_block_delta` is emitted for an index after its `content_block_stop`.

- [x] **Step 2: Run and capture the illegal sequence**

  Run: `uv run pytest tests/test_translation_advanced.py -k interleaved_parallel_tools -v`

- [x] **Step 3: Buffer tool arguments until an explicit flush point with a hard ceiling**

  Because the upstream deltas have no reliable per-tool completion marker, accumulate arguments by upstream tool index and flush complete `tool_use` blocks in stable index order only at the overall explicit terminal. Count buffered argument bytes toward the existing runaway `max_bytes` ceiling (or an equally strict per-request ceiling sourced from the same config) so the new state cannot grow without bound. Keep the existing text and thinking block ordering rules.

- [x] **Step 4: Add malformed/incomplete arguments coverage**

  Treat partial JSON at explicit terminal or unexpected EOF as a protocol error rather than silently manufacturing `{}`. Cover A/B/A delta interleaving, tool/text/tool ordering, oversized arguments, and EOF before terminal; after an error, emit neither `content_block_stop` for a fabricated tool nor `message_stop` success.

- [x] **Step 5: Verify**

  Run: `uv run pytest tests/test_translation_advanced.py tests/test_anthropic_provider_stream.py tests/test_anthropic_stream_invoke_recovery.py -v`

**Suggested commit boundary:** `fix: serialize interleaved Anthropic tool blocks`

### Task 5: Preserve native Responses terminal status

**Why:** Provider results collapse `incomplete` into Chat finish reasons, and the native Responses route re-emits them as `completed` with no incomplete details.

**Files:**
- Modify: `src/router_maestro/providers/base.py`
- Modify: `src/router_maestro/providers/copilot.py`
- Modify: `src/router_maestro/utils/responses_bridge.py`
- Modify: `src/router_maestro/server/routes/responses.py`
- Test: `tests/test_responses_bridge.py`
- Test: `tests/test_copilot_responses_stream.py`
- Test: `tests/test_responses_route_wire_shape.py`

- [x] **Step 1: Add non-stream and stream failing tests**

  Cover `incomplete + max_output_tokens`, `incomplete + content_filter`, `failed`, and `cancelled`. Assert native Responses wire status/details, then separately assert the lossy Chat mapping.

- [x] **Step 2: Run and confirm native output is currently completed**

- [x] **Step 3: Extend internal Responses results/chunks**

  Carry typed native status, incomplete reason/details, and error. Retain `finish_reason` only as an adapter output, not the canonical source.

- [x] **Step 4: Move status mapping to protocol adapters**

  The native Responses route preserves status. Chat/Anthropic/Gemini bridges map it to their finish/stop semantics.

- [x] **Step 5: Verify**

  Run:

  ```bash
  uv run pytest tests/test_responses_bridge.py tests/test_copilot_responses_stream.py tests/test_responses_route_wire_shape.py -v
  uv run pytest tests/ -q
  ```

**Suggested commit boundary:** `fix: preserve Responses incomplete status`

---

## Phase 1: Routing semantics and provider capabilities

### Task 6: Introduce model identity, operation, features, and capabilities

**Files:**
- Create: `src/router_maestro/routing/model_ref.py`
- Create: `src/router_maestro/routing/capabilities.py`
- Create: `src/router_maestro/routing/route_plan.py`
- Modify: `src/router_maestro/providers/base.py`
- Modify: `src/router_maestro/routing/router.py`
- Modify: `src/router_maestro/server/routes/anthropic_beta.py`
- Test: `tests/test_router_advanced.py`
- Test: `tests/test_anthropic_beta.py`
- Create: `tests/test_provider_capabilities.py`

- [x] **Step 1: Add capability selection tests**

  Cover Chat, Chat stream, Responses, Responses stream, native Anthropic, tools, vision, parallel tools, and reasoning. Include two models from the same provider with different support (for example, a Copilot Claude model that can use native Anthropic and a Copilot GPT model that cannot). The beta-native messages route must request `Operation.NATIVE_ANTHROPIC` through `RoutePlan`; replace its local `_is_native_eligible()` authority with the same model capability result. The key regression is an auto-routed Responses request whose first priority cannot perform Responses but the second can.

  Token counting is deliberately not a completion-routing operation in this task: local estimates and exact provider token APIs have protocol-specific inputs and must not silently fall back to a different tokenizer/model. Count-token routes may use public `ModelRef` resolution, and Task 14 replaces the beta route's private Copilot transport call with a public facade method.

- [x] **Step 2: Define the minimal types**

  ```python
  class Operation(StrEnum):
      CHAT = "chat"
      CHAT_STREAM = "chat_stream"
      RESPONSES = "responses"
      RESPONSES_STREAM = "responses_stream"
      NATIVE_ANTHROPIC = "native_anthropic"

  @dataclass(frozen=True)
  class ModelRef:
      provider: str
      upstream_id: str

  @dataclass(frozen=True)
  class RequestFeatures:
      tools: bool = False
      vision: bool = False
      reasoning: bool = False
      parallel_tools: bool = False
  ```

  Define separate `ProviderCapabilities` (transport-level operations implemented by a provider) and tri-state `ModelCapabilities` (supported/unsupported/unknown operations and features for one catalog model). Bind `ModelCapabilities` to a `ModelRef`/catalog entry; do not infer that every model supports an operation because one model from that provider does.

- [x] **Step 3: Add public provider capability declarations**

  Default `BaseProvider` to its actually implemented operations. Override only where a provider differs. Populate model capabilities from `ModelInfo`, catalog metadata, and existing explicit compatibility rules, preserving the distinction between unknown support and explicit `False`. Lock this decision matrix in tests:

  | Selection | Capability | Planning/execution behavior |
  |---|---|---|
  | Automatic | supported | Keep candidate in configured priority order. |
  | Automatic | unsupported | Filter candidate before execution. |
  | Automatic | unknown | Rank after all known-supported candidates while preserving relative priority among unknowns; attempt only as a compatibility fallback. A typed runtime unsupported failure may advance to the next automatic candidate. |
  | Explicit `provider/model` | unsupported | Return the entry protocol's native 400 and do not switch models. |
  | Explicit `provider/model` | unknown | Permit the attempt for backward compatibility; a typed unsupported result is non-retryable and does not switch models. |

  Erratum: `Operation.NATIVE_ANTHROPIC` describes provider transport eligibility; on beta Messages, adapt an explicitly selected `ModelRef` without that transport through the standard translated handler for the same model (not model fallback), while explicitly unsupported request features or options—including tools, vision, or reasoning—remain native 400s and Decision A may switch models only after a retryable execution failure.

  Decision A applies to retryable provider/transport failures after a valid explicit primary is selected. It does not turn an explicit static capability mismatch or invalid request into permission to substitute a different model.

- [x] **Step 4: Return a `RoutePlan` from resolution**

  Resolve Decision A in these tests because candidate enumeration is part of the first `RoutePlan` contract. The plan contains the primary and filtered/ranked fallback candidates, each with its `ModelRef` and effective tri-state capabilities. Keep `Router` as the public facade during migration. Make beta-native messages consume this plan for `NATIVE_ANTHROPIC`; leave count-token behavior scoped as described above.

- [x] **Step 5: Verify**

  Run: `uv run pytest tests/test_provider_capabilities.py tests/test_router_advanced.py tests/test_anthropic_beta.py tests/test_providers.py -v`

**Suggested commit boundary:** `refactor: add capability-aware route planning`

### Task 7: Define and normalize provider failure contracts

**Files:**
- Modify: `src/router_maestro/providers/base.py`
- Modify: `src/router_maestro/providers/openai_base.py`
- Modify: `src/router_maestro/providers/anthropic.py`
- Modify: `src/router_maestro/providers/copilot.py`
- Test: `tests/test_providers.py`
- Test: `tests/test_providers_advanced.py`
- Create: `tests/test_provider_protocol_errors.py`

- [x] **Step 1: Add malformed 2xx tests**

  Cover invalid JSON, HTML, empty choices/content, bad SSE JSON before first chunk, and bad SSE JSON after first chunk.

- [x] **Step 2: Add a typed provider failure contract**

  At minimum distinguish transport, authentication, rate limit, upstream status, upstream protocol, unsupported operation, and client request. Carry HTTP status, safe client message, retryability, provider/model context, and original cause. This is the sole error-kind vocabulary used by the later attempt ledger and protocol encoders.

- [x] **Step 3: Catch adapter-boundary parser failures**

  Convert `JSONDecodeError`, `KeyError`, `IndexError`, and schema violations into a 502 upstream-protocol failure. Mark it retryable before stream commitment.

- [x] **Step 4: Preserve the stream commit distinction**

  Expose malformed events before the primed first chunk as retryable provider failures. After the first chunk is committed, surface the same typed failure to the protocol stream encoder without replay or provider switching.

- [x] **Step 5: Verify**

  Run: `uv run pytest tests/test_provider_protocol_errors.py tests/test_providers.py tests/test_providers_advanced.py -v`

**Suggested commit boundary:** `fix: normalize upstream protocol failures`

### Task 8: Unify fallback attempt semantics

**Files:**
- Modify: `src/router_maestro/routing/router.py`
- Modify: `src/router_maestro/providers/base.py`
- Test: `tests/test_router_advanced.py`
- Test: `tests/test_provider_protocol_errors.py`
- Create: `tests/test_fallback_attempts.py`

- [x] **Step 1: Add a shared stream/non-stream attempt matrix**

  Cover primary retryable failure, secondary fatal failure, all retryable failures, auth failure, malformed upstream protocol failure from Task 7, and max retry count.

- [x] **Step 2: Verify the RoutePlan candidate contract**

  Reuse the Decision A behavior fixed by Task 6 and assert execution consumes candidates in exactly that order without recomputing priorities mid-request.

- [x] **Step 3: Implement a shared attempt ledger**

  Record provider, `ModelRef`, operation, HTTP status (when one exists), typed failure kind, and retryability. Stop on non-retryable errors. On exhaustion, raise the last meaningful failure or a typed `AllProvidersFailed` that preserves all attempts.

- [x] **Step 4: Keep the stream commit point**

  Continue priming the first chunk before exposing the iterator. A failure before the first chunk may select the next planned candidate; once the first chunk is committed, never switch providers.

- [x] **Step 5: Add structured fallback logging/metrics**

  Reuse the existing observability mechanism; do not add a second metrics registry.

- [x] **Step 6: Verify**

  Run: `uv run pytest tests/test_fallback_attempts.py tests/test_provider_protocol_errors.py tests/test_router_advanced.py tests/test_observability_metrics.py -v`

**Suggested commit boundary:** `refactor: unify provider fallback decisions`

### Task 9: Make model listing and fuzzy resolution unambiguous

**Files:**
- Modify: `src/router_maestro/routing/router.py`
- Modify: `src/router_maestro/utils/model_match.py`
- Modify: `src/router_maestro/server/routes/models.py`
- Modify: `src/router_maestro/server/routes/anthropic.py`
- Modify: `src/router_maestro/cli/model.py`
- Test: `tests/test_model_match.py`
- Test: `tests/test_models_route.py`
- Test: `tests/test_anthropic_models.py`
- Test: `tests/test_router_advanced.py`

- [x] **Step 1: Resolve Decision B and add round-trip tests**

  A model selected from a public list must route back to the same provider. Cover two providers exposing the same upstream ID.

- [x] **Step 2: Return unique public identifiers**

  Keep `ModelRef` internally and encode the agreed public ID at the route boundary.

- [x] **Step 3: Make fuzzy score authoritative**

  Choose the highest score first. Use model date/version only to break ties within the same normalized family. Return an ambiguity error when confidence is too low or top candidates are effectively tied across families.

- [x] **Step 4: Verify aliases and provider scoping**

  Preserve dot/hyphen aliases and ensure `provider/query` never crosses providers.

- [x] **Step 5: Verify**

  Run: `uv run pytest tests/test_model_match.py tests/test_models_route.py tests/test_anthropic_models.py tests/test_router_advanced.py -v`

**Suggested commit boundary:** `fix: make model identity and fuzzy matching deterministic`

### Task 10: Replace `extra` with typed request options and an explicit support policy

**Files:**
- Modify: `src/router_maestro/providers/base.py`
- Modify: `src/router_maestro/server/routes/chat.py`
- Create: `src/router_maestro/server/protocols/__init__.py`
- Create: `src/router_maestro/server/protocols/errors.py`
- Modify: `src/router_maestro/server/translation.py`
- Modify: `src/router_maestro/server/translation_gemini.py`
- Modify: `src/router_maestro/utils/responses_bridge.py`
- Modify: `src/router_maestro/providers/openai_base.py`
- Modify: `src/router_maestro/providers/anthropic.py`
- Modify: `src/router_maestro/providers/copilot.py`
- Test: `tests/test_chat_route_options.py`
- Test: `tests/test_gemini_routes.py`
- Test: `tests/test_thinking_passthrough.py`
- Create: `tests/test_provider_option_fidelity.py`
- Create: `tests/test_protocol_error_envelopes.py`

- [x] **Step 1: Create an option-fidelity matrix**

  Cover `top_p`, frequency/presence penalties, stop, user, top_k, stop sequences, metadata, service tier, temperature, and reasoning effort across each provider/operation.

- [x] **Step 2: Resolve Decision C**

  Encode each provider's `supported`, `translated`, `substituted-down`, and `rejected` behavior in tests. Add explicit coverage showing the recommended policy never maps a requested reasoning tier upward; if the compatibility alternative is chosen, assert that every upward substitution is surfaced rather than silently called a downgrade.

- [x] **Step 3: Add typed fields**

  Move supported cross-protocol options out of `extra`. Retain a narrowly scoped provider-extension map only if a real use case remains; it must not overwrite core payload fields with `dict.update()`.

- [x] **Step 4: Make bridges preserve or reject options**

  Chat → Responses and Anthropic/Gemini translations must not silently lose accepted options. Introduce the minimal shared `CLIENT_REQUEST` failure mapping and protocol error encoder here so rejection produces the correct native 400; Task 13 later expands the same encoder to all error sources and stream phases rather than migrating route-specific `HTTPException(detail=...)` bodies.

- [x] **Step 5: Verify**

  Run: `uv run pytest tests/test_provider_option_fidelity.py tests/test_chat_route_options.py tests/test_gemini_routes.py tests/test_thinking_passthrough.py tests/test_protocol_error_envelopes.py -v`

**Suggested commit boundary:** `refactor: type provider request options`

---

## Phase 2: Protocol reducers and error envelopes

### Task 11: Extract Responses streaming into a pure reducer

**Files:**
- Modify: `src/router_maestro/server/protocols/__init__.py`
- Create: `src/router_maestro/server/protocols/responses_reducer.py`
- Modify: `src/router_maestro/server/routes/responses.py`
- Test: `tests/test_responses_route_wire_shape.py`
- Test: `tests/test_streaming_accumulation.py`
- Create: `tests/test_responses_reducer.py`

- [ ] **Step 1: Characterize current valid event sequences**

  Cover text, reasoning, function/custom/tool-search calls, namespace, usage, incomplete, failure, and interleaving.

- [ ] **Step 2: Define reducer input/output**

  The reducer consumes canonical provider chunks and emits typed Responses events plus a final snapshot. It owns item indexes, per-item accumulators, and terminal state; it does not select providers or write SSE strings.

- [ ] **Step 3: Move behavior without changing wire output**

  Keep route orchestration thin: parse request, call Router, feed reducer, encode events.

- [ ] **Step 4: Make non-stream and stream share output-item construction**

  Eliminate duplicate rules for tool item shapes, reasoning IDs/signatures, usage, and terminal status.

- [ ] **Step 5: Verify**

  Run: `uv run pytest tests/test_responses_reducer.py tests/test_responses_route_wire_shape.py tests/test_streaming_accumulation.py tests/test_responses_usage.py -v`

**Suggested commit boundary:** `refactor: extract Responses event reducer`

### Task 12: Extract Anthropic stream/content normalization

**Files:**
- Create: `src/router_maestro/server/protocols/anthropic_reducer.py`
- Create: `src/router_maestro/providers/anthropic_codec.py`
- Modify: `src/router_maestro/server/translation.py`
- Modify: `src/router_maestro/server/routes/anthropic.py`
- Modify: `src/router_maestro/providers/anthropic.py`
- Test: `tests/test_translation_advanced.py`
- Test: `tests/test_anthropic_provider_stream.py`
- Create: `tests/test_anthropic_provider_codec.py`
- Create: `tests/test_anthropic_reducer.py`

- [ ] **Step 1: Characterize text, thinking/signature, tool, usage, and terminal mapping**

- [ ] **Step 2: Define typed content blocks and canonical stop causes**

  Add only `Text`, `Reasoning`, `ToolCall`, `ToolResult`, and required signature metadata. Avoid copying complete vendor schemas into the core.

- [ ] **Step 3: Keep upstream codec and downstream reducer separate**

  `providers/anthropic_codec.py` converts Anthropic upstream JSON/SSE into canonical provider responses/chunks and is imported only by the provider layer. `server/protocols/anthropic_reducer.py` converts those canonical chunks into downstream Anthropic blocks/events and never parses upstream transport frames. The provider must not import `server.*`, and the server reducer must not call provider-private HTTP/codec helpers.

- [ ] **Step 4: Make each layer internally consistent**

  Make the Anthropic upstream codec's stream/non-stream paths preserve the same reasoning/signature and canonical stop causes. Make the downstream Anthropic reducer's stream/non-stream construction share content-block rules. Prove both equivalences independently in provider-codec and reducer tests.

- [ ] **Step 5: Remove the unused duplicate non-stream translator**

  Either make `translate_openai_to_anthropic()` the route's single implementation or delete it after moving its tested behavior into the reducer. Do not leave two authorities.

- [ ] **Step 6: Verify**

  Run: `uv run pytest tests/test_anthropic_provider_codec.py tests/test_anthropic_reducer.py tests/test_translation.py tests/test_translation_advanced.py tests/test_anthropic_provider_stream.py -v`

**Suggested commit boundary:** `refactor: unify Anthropic content normalization`

### Task 13: Add protocol-native error encoders

**Files:**
- Modify: `src/router_maestro/server/protocols/errors.py`
- Modify: `src/router_maestro/server/app.py`
- Modify: `src/router_maestro/server/routes/chat.py`
- Modify: `src/router_maestro/server/routes/responses.py`
- Modify: `src/router_maestro/server/routes/anthropic.py`
- Modify: `src/router_maestro/server/routes/anthropic_beta.py`
- Modify: `src/router_maestro/server/routes/gemini.py`
- Test: `tests/test_streaming_error_body.py`
- Test: `tests/test_protocol_error_envelopes.py`

- [ ] **Step 1: Add SDK-facing wire tests**

  Cover Pydantic `RequestValidationError`, authentication `HTTPException`, route-level 404/405, typed 400/401/429/502 provider failures, overload, and unexpected internal failure for all four protocols. For streaming endpoints, split every applicable case into pre-commit/provider-open and post-commit cases.

- [ ] **Step 2: Map typed provider failures to protocol errors**

  Before commit, return a native non-2xx JSON response: OpenAI/Responses use an OpenAI `error` object where required, Anthropic uses its `type=error` envelope, and Gemini uses a Google-style error without FastAPI `detail` nesting. After SSE commit, encode the same typed failure as the protocol's in-stream error/failed/incomplete event while the HTTP status remains 200.

- [ ] **Step 3: Prime stream opening before constructing `StreamingResponse`**

  Move provider resolution/opening and the first upstream chunk (or explicit eager protocol prelude decision) ahead of response commitment where possible. This makes model/auth/provider-open failures return non-2xx JSON instead of a misleading 200 from a lazy generator. Once any body bytes are emitted, switch permanently to the post-commit encoder.

- [ ] **Step 4: Route framework and unexpected exceptions through the same protocol layer**

  Dispatch `RequestValidationError`, `HTTPException`, 404/405, and unexpected exceptions by matched protocol path. Preserve request IDs and avoid plain-text 500s or generic `{"detail": ...}` bodies for protocol endpoints.

- [ ] **Step 5: Enforce one terminal event**

  After an in-stream error/failed/incomplete terminal, emit no success terminal such as `[DONE]`, `response.completed`, `message_stop`, or Gemini `STOP`. Add assertions for mutually exclusive terminal sequences.

- [ ] **Step 6: Verify**

  Run: `uv run pytest tests/test_protocol_error_envelopes.py tests/test_streaming_error_body.py tests/test_responses_stream_early_error.py -v`

**Suggested commit boundary:** `fix: emit protocol-native error envelopes`

### Task 14: Split Copilot internals by responsibility after contracts stabilize

**Files:**
- Create: `src/router_maestro/providers/copilot_support/__init__.py`
- Create: `src/router_maestro/providers/copilot_support/auth_session.py`
- Create: `src/router_maestro/providers/copilot_support/transport.py`
- Create: `src/router_maestro/providers/copilot_support/catalog.py`
- Create: `src/router_maestro/providers/copilot_support/chat_codec.py`
- Create: `src/router_maestro/providers/copilot_support/responses_codec.py`
- Modify: `src/router_maestro/providers/copilot.py`
- Move/modify tests from `tests/test_providers.py`, `tests/test_copilot_chat_reasoning.py`, `tests/test_copilot_responses_stream.py`, and `tests/test_copilot_tool_filter.py`

- [ ] **Step 1: Freeze public `CopilotProvider` contract with tests**

- [ ] **Step 2: Extract transport and auth without behavior changes**

  Keep token mint, 401/403 retry, HTTP/2 client recycle, and headers in this boundary.

- [ ] **Step 3: Extract catalog parsing**

  Keep catalog freshness and capability parsing independent of completion codecs.

- [ ] **Step 4: Add a public native token-count method**

  Move the beta Anthropic route's direct `_send_with_auth_retry()` call behind a public Copilot facade/transport method that accepts native count-token payloads and returns the typed upstream result/failure. Keep standard Anthropic/Gemini local estimation at the protocol boundary; do not route token counting through a different completion model.

- [ ] **Step 5: Extract Chat and Responses codecs/reducers**

  Preserve tool XML recovery, custom/tool-search/namespace handling, and reasoning ID/signature pairing in provider-boundary tests.

- [ ] **Step 6: Keep `CopilotProvider` as a facade**

  Keep the existing `providers/copilot.py` module as the stable facade and import its internal collaborators from `copilot_support/`. Do not create a `providers/copilot/__init__.py` package alongside `copilot.py`: that module/package name collision makes import resolution and intermediate commits ambiguous. Router and routes continue importing `CopilotProvider` from the existing public module/package export.

- [ ] **Step 7: Verify**

  Run:

  ```bash
  uv run pytest tests/test_providers.py tests/test_providers_advanced.py tests/test_copilot_chat_reasoning.py tests/test_copilot_responses_stream.py tests/test_copilot_tool_filter.py tests/test_tool_parsing.py -v
  uv run pytest tests/ -q
  ```

**Suggested commit boundaries:** one extraction commit per responsibility; never combine all five moves with behavior changes.

---

## Phase 3: Request lifecycle, credentials, configuration, and resources

### Task 15: Establish versioned runtime configuration snapshots

**Files:**
- Create: `src/router_maestro/config/repository.py`
- Modify: `src/router_maestro/config/settings.py`
- Modify: `src/router_maestro/server/routes/admin.py`
- Modify: `src/router_maestro/server/schemas/admin.py`
- Test: `tests/test_config_advanced.py`
- Test: `tests/test_admin_routes.py`
- Create: `tests/test_runtime_config_snapshot.py`

- [ ] **Step 1: Add repository and lost-update tests**

  Read one immutable snapshot, mutate the backing file through a second repository instance, and prove the first snapshot does not change. Add two Admin PATCH operations carrying the same revision and assert exactly one succeeds while the stale writer receives a conflict. Derive the revision from canonical validated content rather than file modification time alone.

- [ ] **Step 2: Implement immutable snapshot + revision**

  The repository validates `PrioritiesConfig`, returns an immutable/deep-copied snapshot with a content revision, and writes through the existing atomic persistence primitive. Expose compare-and-swap update; do not add a file watcher or multi-process coordination.

- [ ] **Step 3: Add typed Admin GET/PATCH**

  Expose all supported runtime fields and require a revision/ETag for mutation to prevent lost updates. Keep current whole-file helpers as compatibility wrappers until Task 17 migrates request consumers.

- [ ] **Step 4: Verify the foundation**

  Run: `uv run pytest tests/test_runtime_config_snapshot.py tests/test_config_advanced.py tests/test_admin_routes.py -v`

**Suggested commit boundary:** `refactor: add versioned runtime config snapshots`

### Task 16: Give Router generations and model catalogs explicit async ownership

**Files:**
- Modify: `src/router_maestro/routing/router.py`
- Modify: `src/router_maestro/server/app.py`
- Modify: `src/router_maestro/server/routes/admin.py`
- Modify: `src/router_maestro/utils/cache.py`
- Modify: `src/router_maestro/providers/copilot_support/catalog.py`
- Test: `tests/test_router_advanced.py`
- Test: `tests/test_providers.py`
- Create: `tests/test_router_lifecycle.py`

- [ ] **Step 1: Add generation-lease and concurrent-refresh tests**

  Start a stream on generation A, swap to generation B, and assert A's provider client stays open until the stream terminates or is cancelled; then it closes exactly once. Assert new requests use B. Also assert concurrent cold-cache requests share one catalog refresh.

- [ ] **Step 2: Move the Router owner to `app.state`**

  Supply the active generation through FastAPI dependencies. Build a new generation before an atomic swap; keep a compatibility `get_router()` facade only while callers migrate.

- [ ] **Step 3: Add an async generation lease/refcount**

  Request/stream owners acquire a lease before selecting a provider. A retired generation rejects new leases and closes its provider resources only after all existing leases release. Release must be idempotent and run on success, exception, and cancellation; never close a client still serving a response body.

- [ ] **Step 4: Make catalog refresh single-flight**

  Build a new immutable snapshot off to the side and atomically replace the old one. Continue serving stale data until the new snapshot is ready. Use monotonic time for TTL and max-age decisions.

- [ ] **Step 5: Verify**

  Run: `uv run pytest tests/test_router_lifecycle.py tests/test_router_advanced.py tests/test_providers.py tests/test_timeout.py -v`

**Suggested commit boundary:** `refactor: own router generations and catalog lifecycles`

### Task 17: Put pipeline and audit ownership in a request context

**Files:**
- Create: `src/router_maestro/runtime/__init__.py`
- Create: `src/router_maestro/runtime/request_context.py`
- Modify: `src/router_maestro/server/app.py`
- Modify: `src/router_maestro/pipeline/request_pipeline.py`
- Modify: `src/router_maestro/utils/audit.py`
- Modify: `src/router_maestro/routing/router.py`
- Modify: `src/router_maestro/server/routes/chat.py`
- Modify: `src/router_maestro/server/routes/responses.py`
- Modify: `src/router_maestro/server/routes/anthropic.py`
- Modify: `src/router_maestro/server/routes/anthropic_beta.py`
- Modify: `src/router_maestro/server/routes/gemini.py`
- Modify: `src/router_maestro/providers/openai_base.py`
- Modify: `src/router_maestro/providers/anthropic.py`
- Modify: `src/router_maestro/providers/copilot_support/transport.py`
- Test: `tests/test_audit.py`
- Test: `tests/test_stream_pipeline_finish.py`
- Test: `tests/test_runtime_config_snapshot.py`
- Test: `tests/test_router_lifecycle.py`
- Create: `tests/test_request_lifecycle.py`

- [ ] **Step 1: Add end-to-end trace lifecycle tests**

  With tracing enabled, assert unique request IDs and the intended inbound/upstream/upstream-response/outbound records for stream and non-stream success, early open failure, provider error, unexpected EOF, post-commit error, cancellation, and beta-native requests. First add a characterization proving a single request can currently observe cached fallback config and freshly reloaded guard/thinking config. Then start a stream, mutate config and swap Router generations mid-stream, and assert the request keeps its initial config revision and generation lease until the body finishes.

- [ ] **Step 2: Create request context before route execution**

  Acquire exactly one immutable config snapshot from Task 15 and one Router generation lease from Task 16. Carry their revisions/IDs with request ID, audit trace, stream-commit state, `TransportTermination`, and `ResponseStatus`. Router planning, thinking resolution, guards, audit, and beta transformations must read this context rather than independently reloading configuration.

- [ ] **Step 3: Bind finalization to ASGI body completion**

  A route returning `StreamingResponse` must not finalize or release its generation lease merely because the endpoint function returned. Wrap the body iterator so success terminal, exception, unexpected EOF, and `CancelledError` each finalize the request and release the lease exactly once after the last ASGI body event. Non-streaming responses finalize after response construction.

- [ ] **Step 4: Record upstream at the transport boundary**

  Routes must not reconstruct provider URLs, headers, or payloads for audit.

- [ ] **Step 5: Integrate all stream guards, including beta-native**

  Respect the captured snapshot's enable flags and include leak/runaway guard behavior. Wire `beta_strip` from that snapshot or remove the dead option and correct the documentation after user confirmation.

- [ ] **Step 6: Move audit file writes off the event loop**

  Use `asyncio.to_thread` or a bounded async writer; redact sensitive headers and payload fields before scheduling writes.

- [ ] **Step 7: Verify**

  Run: `uv run pytest tests/test_request_lifecycle.py tests/test_runtime_config_snapshot.py tests/test_router_lifecycle.py tests/test_audit.py tests/test_stream_pipeline_finish.py tests/test_pipeline_guards.py -v`

**Suggested commit boundary:** `refactor: centralize request pipeline lifecycle`

### Task 18: Introduce atomic credential mutation

**Files:**
- Create: `src/router_maestro/auth/repository.py`
- Modify: `src/router_maestro/auth/manager.py`
- Modify: `src/router_maestro/auth/storage.py`
- Modify: `src/router_maestro/providers/copilot_support/auth_session.py`
- Modify: `src/router_maestro/server/routes/admin.py`
- Test: `tests/test_auth.py`
- Test: `tests/test_auth_advanced.py`
- Test: `tests/test_providers.py`
- Create: `tests/test_auth_concurrency.py`

- [ ] **Step 1: Add the lost-update reproduction**

  Hold an old Copilot credential snapshot, add an OpenAI key through another manager, then persist a refreshed Copilot credential. Assert both credentials survive.

- [ ] **Step 2: Implement `update_provider` and `remove_provider`**

  Under a process lock, read the latest file, patch one provider, and call the existing owner-only atomic writer.

- [ ] **Step 3: Remove whole-file persistence from providers**

  Copilot refresh calls the repository operation; it must not save an old `AuthManager` map.

- [ ] **Step 4: Verify concurrent login/logout/refresh**

  Run: `uv run pytest tests/test_auth_concurrency.py tests/test_auth.py tests/test_auth_advanced.py tests/test_providers.py -v`

**Suggested commit boundary:** `fix: prevent credential lost updates`

### Task 19: Separate administrator and inference authentication

**Files:**
- Modify: `src/router_maestro/server/middleware/auth.py`
- Modify: `src/router_maestro/server/app.py`
- Modify: `src/router_maestro/config/contexts.py`
- Modify: `src/router_maestro/config/server.py`
- Modify: `src/router_maestro/cli/client.py`
- Modify: `src/router_maestro/cli/context.py`
- Modify: `README.md`
- Modify: `docs/deployment.md`
- Test: `tests/test_auth_middleware.py`
- Test: `tests/test_admin_routes.py`
- Test: `tests/test_config.py`
- Test: `tests/test_config_advanced.py`
- Create: `tests/test_admin_client_auth.py`

- [ ] **Step 1: Resolve Decision D and add server compatibility tests**

  Cover a configured admin key, a wrong inference key, the temporary no-admin-key fallback from Decision D, startup warning, and remote-admin-disabled mode. In admin-key mode, prove an inference token cannot reach `/api/admin/*`.

- [ ] **Step 2: Add an admin-key verifier**

  Keep constant-time comparison and existing header compatibility. Allow an explicit setting to disable remote admin routes.

- [ ] **Step 3: Give CLI contexts two explicit credentials**

  Add `admin_api_key` alongside the existing inference `api_key` in `ContextConfig`, plus `--admin-api-key` input/display/update support. `AdminClient` uses `admin_api_key`; during the compatibility window only, it may fall back to the existing `api_key` when no admin credential is stored. Inference helpers continue returning only `api_key` and must never read `admin_api_key`. Preserve owner-only file permissions and do not log either secret.

- [ ] **Step 4: Document migration and removal window**

  Document local server setup, remote context setup, the compatibility fallback, how to rotate each credential independently, and the planned release in which inference-key admin fallback will be removed.

- [ ] **Step 5: Verify**

  Run: `uv run pytest tests/test_auth_middleware.py tests/test_admin_routes.py tests/test_admin_client_auth.py tests/test_config.py tests/test_config_advanced.py tests/test_cli_entry.py -v`

**Suggested commit boundary:** `feat: separate admin API credentials`

### Task 20: Repair the custom-provider credential workflow

**Files:**
- Modify: `src/router_maestro/routing/router.py`
- Modify: `src/router_maestro/cli/auth.py`
- Modify: `src/router_maestro/server/routes/admin.py`
- Modify: `src/router_maestro/config/providers.py`
- Modify: `README.md`
- Test: `tests/test_config.py`
- Test: `tests/test_admin_routes.py`
- Create: `tests/test_custom_provider_auth.py`

- [ ] **Step 1: Choose and document the credential source priority**

  Recommended: explicit environment variable → auth repository → unauthenticated only when provider config explicitly permits it.

- [ ] **Step 2: Add tests for documented `OLLAMA_API_KEY` behavior**

  Also cover normalized hyphenated provider names and a custom key written through the Admin API.

- [ ] **Step 3: Make provider discovery data-driven**

  Obtain custom provider definitions/auth requirements from the server for remote contexts. When the server is unavailable in a local bootstrap flow, read the same typed local provider config instead of falling back to a three-provider constant.

- [ ] **Step 4: Either type and consume `options` or remove it**

  Do not retain an unrestricted documented config field with no runtime effect.

- [ ] **Step 5: Verify**

  Run: `uv run pytest tests/test_custom_provider_auth.py tests/test_config.py tests/test_admin_routes.py -v`

**Suggested commit boundary:** `fix: make custom provider authentication usable`

### Phase 3 gate: Verify lifecycle and state changes together

- [ ] **Step 6: Verify Phase 3 as a system**

  After Tasks 15–20, run:

  ```bash
  uv run pytest tests/ -q
  uv run ruff check src/ tests/
  uv run ruff format --check src/ tests/
  RM_INTEGRATION_MAX_MODELS=8 make integration-test
  ```

  The bounded live run is required here because this phase changes every inference route's request lifetime, authentication, Router ownership, and provider resource lifetime.

---

## Phase 4: Documentation, live validation, and cleanup

### Task 21: Align documentation with actual contracts

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md` only for new release notes, not historical rewriting unless explicitly approved
- Modify: `docs/api-translation.md`
- Modify: `docs/token-calculation.md`
- Modify: `docs/observability.md`
- Modify: `docs/deployment.md`
- Modify: `docs/tool-choice-behavior.md`

- [ ] **Step 1: Document terminal and error semantics**

- [ ] **Step 2: Document capability-aware fallback and explicit-model behavior**

- [ ] **Step 3: Document unique model IDs and bare aliases**

- [ ] **Step 4: Document option support/rejection by provider**

- [ ] **Step 5: Document real audit artifacts and administrator authentication**

- [ ] **Step 6: Remove stale claims**

  In particular, verify existing claims about five-route pipeline unification, four audit files per request, beta header stripping, custom-provider environment keys, 1M variants, and reasoning-suffix rewrites against production code.

### Task 22: Expand live integration coverage at the boundaries changed by this plan

**Files:**
- Modify: `integration_tests/conftest.py`
- Modify: `integration_tests/test_live_reasoning_matrix.py`
- Modify: `integration_tests/test_live_anthropic_beta.py`
- Add or modify integration tests for fallback/terminal behavior where a controllable fake upstream is required

- [ ] **Step 1: Add `enabled + budget + effort` to stream/non-stream live coverage**

- [ ] **Step 2: Add operation/capability routing coverage**

- [ ] **Step 3: Add request-option fidelity coverage for supported options**

- [ ] **Step 4: Add protocol-native error envelope checks**

- [ ] **Step 5: Run the bounded matrix**

  Run: `RM_INTEGRATION_MAX_MODELS=8 make integration-test`

- [ ] **Step 6: Reserve the full matrix for the final post-cleanup gate**

  Do not claim full live validation yet: Task 23 still changes production code, dependencies, the lock file, and packaging. Record the bounded result and run the full matrix only after that cleanup.

### Task 23: Remove dead abstractions and unused dependencies only after migration

**Files:**
- Modify after evidence: `src/router_maestro/pipeline/guards.py`
- Modify after evidence: `src/router_maestro/pipeline/beta_strip.py`
- Modify after evidence: `src/router_maestro/server/translation.py`
- Modify after evidence: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Search production references after all migrations**

  Use `rg` to prove whether `guarded_stream`, `build_guards`, beta-strip helpers, duplicate translators, and candidate dependencies are unused.

- [ ] **Step 2: Delete only confirmed dead code/dependencies**

- [ ] **Step 3: Re-lock and verify imports/build**

  Run:

  ```bash
  uv lock
  uv run python -m compileall -q src/router_maestro
  uv run router-maestro --help
  uv build
  uv run pytest tests/ -q
  uv run ruff check src/ tests/
  uv run ruff format --check src/ tests/
  ```

  Inspect the wheel/sdist contents for the new `providers/copilot_support/` and `server/protocols/` packages, then install the wheel into a temporary uv environment and run `router-maestro --help` as the packaging smoke test.

- [ ] **Step 4: Run the final live matrix on the exact acceptance tree**

  Run: `make integration-test`

  This is the last behavior-changing verification step. Record the commit/tree hash and matrix result so the final acceptance claim applies to the same code that was built and packaged.

**Suggested commit boundary:** `chore: remove superseded compatibility code`

---

## Final acceptance criteria

The remediation is complete only when all of the following are true:

1. Every protocol requires an explicit successful terminal outcome. Pre-commit failures return native non-2xx JSON; post-commit unexpected EOF keeps HTTP 200 but emits a native in-stream error/incomplete terminal and records a non-success internal outcome.
2. Responses native status and incomplete details survive provider normalization.
3. Responses and Anthropic streaming reducers pass interleaved text/reasoning/tool sequence tests.
4. A fallback candidate is selected only if both its provider transport and selected model support the requested operation and required features.
5. Stream and non-stream fallback use the same retry/fatal-error policy.
6. A model selected from a public list routes back to the same provider.
7. Accepted request options are supported, explicitly translated/substituted according to Decision C, or rejected—never silently dropped or silently increased in cost/effort.
8. Malformed upstream 2xx payloads become typed provider protocol errors and participate in safe fallback.
9. Validation, authentication, routing, provider, framework, and internal failures use protocol-native envelopes, with mutually exclusive success/error terminal events before and after stream commit.
10. Audit/request pipeline covers stream and non-stream success, error, unexpected EOF, cancellation, early failure, and beta-native paths with unique request IDs; stream finalization occurs at ASGI body termination, not endpoint return.
11. Concurrent credential updates cannot overwrite unrelated providers.
12. Inference credentials do not implicitly grant administration when the new admin-key mode is enabled.
13. The documented custom-provider authentication workflow works end to end.
14. Each request consumes one immutable runtime configuration revision.
15. Router generations use leases so old provider resources close exactly once only after in-flight request bodies finish, and catalog refresh is single-flight.
16. `uv run pytest tests/ -q`, Ruff, format check, compile check, `uv build` plus wheel smoke installation, and the full live integration matrix pass.

## Explicitly deferred

- Microservices or service decomposition.
- Redis/database-backed configuration or credentials.
- Multi-worker/HA synchronization.
- A universal, lossless AST for every vendor protocol.
- Replacing FastAPI, Pydantic, httpx, uv, or pytest.
- File-watch-based hot reload before snapshot/revision semantics exist.
- Performance optimization not supported by profiling or production metrics.

## Recommended execution order

1. Complete Phase 0 in order.
2. Resolve Decisions A–C, then complete Phase 1 in order; provider failure normalization (Task 7) precedes fallback execution unification (Task 8).
3. Complete reducers before splitting Copilot.
4. In Phase 3, establish config snapshots (Task 15) and Router generation leases (Task 16) before request-context adoption (Task 17).
5. Complete atomic credential mutation (Task 18), resolve Decision D before administrator-auth work (Task 19), then repair custom-provider auth (Task 20).
6. Run the Phase 3 system gate before documentation, full live validation, and dead-code cleanup.

Phase 0 and most isolated tests can use parallel subagents, but tasks that modify `providers/base.py`, `routing/router.py`, or shared protocol contracts should be serialized to avoid conflicting contract definitions.
