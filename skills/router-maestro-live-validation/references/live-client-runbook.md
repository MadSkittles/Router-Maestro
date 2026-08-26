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

Claude example:

```bash
claude -p --model "github-copilot/MODEL" --no-session-persistence \
  --tools="" "Reply with exactly: OK MODEL"
```

Keep the empty tools value attached with `=`. Current Claude Code versions parse `--tools` as a
variadic option, so `--tools ""` before the prompt can consume the prompt and fail without sending
a request.

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

```bash
claude --model "github-copilot/gemini-3.6-flash" \
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
| Codex → Responses Grok | `openai_responses` | `openai_responses` | `identity` | `false` |

Every final request must have HTTP 200 and `outcome=selected`. Confirm that no post-frame replay or
unexpected model/transport switch occurred.

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
- `namespaced tool name exceeds the function-name limit`: function-only transport projection is
  expanding MCP namespace/name pairs beyond the provider limit.
- second-turn Copilot 400 after a successful Codex response: an unverifiable `encrypted_content`
  blob may have been replayed without its upstream reasoning ID.
- `No provider transport is available for anthropic_messages` only on turn 2: empty Anthropic
  reasoning carriers may have been mistaken for opaque continuation state.
- bare Copilot Chat 400 after MCP initialization: inspect the full tool registry for Gemini schema
  incompatibilities, especially nullable type arrays and scalar enums attached to array schemas.

## Final verification and report

Run:

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
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
