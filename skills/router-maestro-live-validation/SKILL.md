---
name: router-maestro-live-validation
description: Run Router-Maestro live validation through Claude Code and Codex, including model smoke sweeps, true multi-turn TUI sessions, file and MCP tool calls, and audit correlation. Use after provider, dispatcher, protocol-conversion, client-config, or deployment changes; do not use as a substitute for offline pytest coverage.
---

# Router-Maestro Live Client Validation

Validate the deployed test instance from the real Claude Code and Codex clients. A one-shot
command is a smoke test; it does not count as multi-turn validation.

Before running, read [references/live-client-runbook.md](references/live-client-runbook.md).

## Required invariants

- Confirm the exact test image/commit and `/health` before sending model traffic.
- Use temporary CLI overrides. Do not rewrite persistent client configuration unless the user
  explicitly requests it.
- Run smoke tests before interactive sessions when the user asks for the full suite.
- Start clean Claude and Codex sessions against the same final image. If a fix is deployed during
  testing, discard the affected session and restart its sequence from round 1.
- Use unique verification tokens and `/tmp` filenames so stale output cannot produce a false pass.
- Exercise both a local file tool and a harmless read-only MCP tool, then verify conversation
  memory after both tool-result round trips.
- Preserve audit evidence and never print credentials, reasoning blobs, capsules, or decrypted
  provider state.
- Report client warnings separately from request failures. Codex model-metadata fallback warnings
  are failures because they can silently change the tool surface and context limits; unrelated
  client warnings are blockers only when they change the requested behavior.

## Completion gate

Do not call the live suite complete until:

1. Claude and Codex both pass every multi-turn round on the same final image.
2. Audit confirms the expected transport and conversion mode for every final request.
3. Codex used explicit metadata for every selected custom model, with no fallback-metadata warning.
4. The local full test suite, Ruff, formatting check, and BasedPyright are clean.
5. Both clients have exited back to the shell prompt and temporary files are gone.
