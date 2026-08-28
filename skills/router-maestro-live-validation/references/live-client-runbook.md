# Live Client Runbook

## Inputs and safety boundary

Resolve these before testing:

- test instance URL and the configured Claude environment that targets it;
- Codex provider name, such as `router-maestro-hk`;
- deployed image tag and source commit;
- representative Claude and Codex models;
- audit state and trace location.

Use a test instance. Rebuilding, deploying, changing audit settings, or editing persistent client
configuration requires authorization from the user or an already-authorized deployment workflow.

When operating an existing Otty tab, use the available Computer Use interface. Re-query the app
state after each action and wait for the client to show `Ready` before entering the next prompt.

## Phase 1: one-shot smoke sweep

Obtain the current model list from Router-Maestro instead of relying on a stale hard-coded catalog.
Select the user-requested models; for a representative Copilot sweep, cover available Gemini,
Grok, and MAI families.

Prefer the bundled runner for the repeatable matrix:

```bash
uv run python skills/router-maestro-live-validation/scripts/live_model_matrix.py \
  --context remote-vm-hk --client all --phase all
```

`--phase all` first runs one-shot smoke cases and then starts a fresh persisted session per
`(client, model)` for a two-request recall check. It dynamically obtains
`/api/openai/v1/models`, gives Codex an isolated version-matched `model_catalog_json`, retries only
an explicit 500/502/503/504 once, and treats any Codex fallback-metadata warning as a failure.
Temporary state and sanitized logs are deleted by default; use `--keep-logs` or `--output-dir` when
diagnostics must survive. The commands below remain useful for narrowing down an individual case.

Claude example:

```bash
claude -p --model "github-copilot/MODEL" --no-session-persistence \
  --prompt-suggestions=false --tools="" --output-format json \
  "Reply with exactly: OPAQUE_UNIQUE_TOKEN" | jq -r '.result'
```

Keep the empty tools value attached with `=`. Current Claude Code versions parse `--tools` as a
variadic option, so `--tools ""` before the prompt can consume the prompt and fail without sending
a request. Prefer an opaque alphanumeric token and inspect `.result`; model/family names inside the
requested phrase can trigger harmless rewriting that makes an exact-match smoke test noisy.

Codex example:

```bash
codex exec \
  -c 'model_provider="router-maestro-hk"' \
  -c 'web_search="disabled"' \
  -m "github-copilot/MODEL" \
  --ephemeral --skip-git-repo-check -s read-only \
  "Reply with exactly: OK MODEL"
```

For a qualified model slug that is not in Codex's bundled catalog, first verify that the active
config supplies a `model_catalog_json` entry generated for the installed Codex version. Treat
`Model metadata for ... not found. Defaulting to fallback metadata` as a failed smoke test. After
catalog or config changes, force one ordinary shell-tool call and confirm Codex emits the standard
shell tool rather than a nested custom tool:

```bash
codex exec \
  -c 'model_provider="router-maestro-hk"' \
  -c 'web_search="disabled"' \
  -m "github-copilot/grok-4.6" \
  --ephemeral --skip-git-repo-check -s read-only \
  "Use the shell tool once to run printf METADATA_TOOL_OK, then reply exactly: METADATA PASS"
```

Record one PASS/FAIL result per `(client, model)`. Retry one isolated transient 5xx once. Do not
retry deterministic 400s into a pass; capture their request ID and investigate them.

## Phase 2: Claude Code multi-turn session

Choose a model that exercises the desired cross-protocol path. Anthropic ingress to a Chat-only
Gemini model is the default representative for Anthropic-to-Chat semantic conversion.

Start a real interactive session. Use the user's authorized permission mode; `bypassPermissions`
is appropriate only when the user has explicitly authorized it.

Use the model string produced by Router-Maestro configuration. If the selected server-advertised
context is 1M, retain the `[1m]` suffix; Claude Code uses that client-side hint to avoid applying
fallback context metadata to qualified custom model IDs.

```bash
claude --model "github-copilot/gemini-3.6-flash[1m]" \
  --permission-mode bypassPermissions
```

Wait for MCP initialization, then run all rounds in the same session:

1. `Remember the verification token CLAUDE-UNIQUE for this session. Reply with exactly: TURN1 ACK`
2. `What verification token did I ask you to remember? Reply with only the token.`
3. Ask Bash to create `/tmp/rm-claude-UNIQUE.txt` with a unique marker, read it, delete it, and
   return an exact success phrase.
4. Ask for exactly one harmless read-only MCP call, preferably `mcp__qmd.status`, followed by an
   exact success phrase.
5. Ask for the original verification token again after both tool calls.
6. Exit with `/exit`.

Pass only if the exact token survives both tool-result round trips and the temporary file is
deleted. Split adjacent assistant text blocks may be joined when they visibly form the exact
requested phrase; missing or reordered content is a failure.

For automated multi-turn checks, prefer `--output-format stream-json
--include-partial-messages --verbose` and concatenate every
`stream_event.event.delta.text` whose event is `content_block_delta` and delta type is
`text_delta`. Claude Code's aggregate JSON `result` can contain only the final assistant text
block when a cross-protocol response is split across several blocks; treating that field alone
as the complete answer creates false missing-prefix failures.

## Phase 3: Codex multi-turn session

Use a Responses-capable model to exercise the identity path. Disable hosted web search when the
selected model has no equivalent hosted-search transport.

```bash
codex \
  -c 'model_provider="router-maestro-hk"' \
  -c 'web_search="disabled"' \
  -m "github-copilot/grok-4.6" \
  -s workspace-write -a never
```

Use the user's authorized sandbox and approval policy. Wait for the client to become `Ready`, then
run the same five logical rounds with a distinct `CODEX-UNIQUE` token and `/tmp` filename:

1. remember and acknowledge;
2. recall;
3. `exec_command` write/read/delete;
4. one `mcp__qmd.status` call;
5. recall after tools;
6. `/exit`.

The interactive `codex` entry point may not accept flags that are valid only for `codex exec`, such
as `--skip-git-repo-check` on some versions.

## Audit expectations

For the representative paths:

| Client path | Expected entry | Expected upstream | Mode | IR |
|---|---|---|---|---|
| Claude → Chat-only Gemini | `anthropic_messages` | `openai_chat` | `semantic_ir` | `true` |
| Claude → Responses Grok/MAI | `anthropic_messages` | `openai_responses` | `semantic_ir` | `true` |
| Codex → Chat-only Gemini | `openai_responses` | `openai_chat` | `semantic_ir` | `true` |
| Codex → Responses Grok | `openai_responses` | `openai_responses` | `identity` | `false` |
| Codex → Responses MAI | `openai_responses` | `openai_responses` | `identity` | `false` |

Every final request must have HTTP 200 and `outcome=selected`. Confirm that no post-frame replay or
unexpected model/transport switch occurred. Correlate by request ID and selected model: clients may
launch auxiliary title, memory, or summarization requests with a different model while the tested
turn is running. Record those separately instead of attributing them to the requested path.

Claude Code may perform one deliberate `mid-conversation-system-2026-04-07` beta negotiation at
the start of a session. Router-Maestro returns a recognizable pre-attempt 400, after which Claude
removes that beta and retries with `<system-reminder>` blocks. Classify it as expected negotiation
only when the response names that exact beta, no provider attempt occurred, the retry succeeds, and
it does not repeat on later turns. Other 400s remain failures to investigate.

## Failure investigation

1. Keep the client session open until the request ID and failure phase are known.
2. Inspect bounded container logs and the matching audit trace. Filter credential-bearing startup
   lines and summarize structures rather than dumping request content.
3. Separate pre-I/O representability failures, provider preparation failures, upstream errors, and
   post-commit stream failures.
4. Add a focused regression test, run the relevant subset, deploy a new uniquely tagged image, and
   restart the affected client sequence from round 1.

Known regression signatures worth checking:

- `Model metadata for ... not found. Defaulting to fallback metadata`: the selected qualified slug
  is missing from `model_catalog_json`, so Codex may use incorrect context and tool capabilities.
- `Grok namespace members must be functions` immediately after a custom model catalog change: the
  generated entry inherited a model-specific code-mode tool surface instead of neutral metadata.
- `OutputTextDelta without active item` on Codex → Chat: the Responses encoder emitted a text delta
  before `response.output_item.added` and `response.content_part.added`; verify the full
  `added → delta → done → completed` lifecycle.
- `namespaced tool name exceeds the function-name limit`: function-only transport projection is
  expanding MCP namespace/name pairs beyond the provider limit.
- second-turn Copilot 400 after a successful Codex response: an unverifiable `encrypted_content`
  blob may have been replayed without its upstream reasoning ID.
- `No provider transport is available for anthropic_messages` only on turn 2: empty Anthropic
  reasoning carriers may have been mistaken for opaque continuation state.
- apparently truncated Claude exact-match output while raw text deltas contain the full value:
  aggregate JSON `result` retained only the final assistant text block; validate the joined text
  deltas before diagnosing an RM stream-loss bug.
- Claude Code compacts a qualified model near 200K even though the live catalog defaults to 1M:
  the explicit model argument lost its `[1m]` client hint. Re-run with the configured model string;
  do not diagnose the resulting compact-summary recall error as an RM transport failure when audit
  shows selected HTTP 200 attempts with explicit terminals.
- bare Copilot Chat 400 after MCP initialization: inspect the full tool registry for Gemini schema
  incompatibilities, especially nullable type arrays and scalar enums attached to array schemas.

## Final verification and report

Run:

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/ integration_tests/ \
  skills/router-maestro-live-validation/scripts/
uv run ruff format --check src/ tests/ integration_tests/ \
  skills/router-maestro-live-validation/scripts/
npx -y basedpyright@1.39.10
```

Report:

- deployed image and commit;
- smoke matrix totals;
- each Claude and Codex multi-turn round;
- audit transport/mode/IR evidence;
- offline test, lint, formatting, and type-check results;
- warnings that remain but did not fail behavior.

Do not list a model-metadata fallback warning as tolerated; it invalidates the Codex result.
