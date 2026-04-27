# GitHub Copilot Model Context Limits — Measured vs Catalog

Catalog values come from Copilot's `/models` endpoint
(`capabilities.limits.max_prompt_tokens` / `max_context_window_tokens`).
Measured values come from a step-ladder probe that sends synthetic
prompts directly to `api.githubcopilot.com` and records the largest size
the upstream accepts before returning `model_max_prompt_tokens_exceeded`.

- Probe script: `scripts/test_context_sizes.py` (kept on the
  `chore/test-copilot-context-sizes` branch).
- Step ladder: 32K → 64K → 128K → 200K → 272K → 400K → 1M.
- Token approximation: 4 chars/token (English-ish).
- Last sweep: **2026-04-27**.

## Results

| Model | Catalog max_prompt | Catalog ctx | Largest accepted | First rejected | Reported limit |
|---|---|---|---|---|---|
| claude-opus-4.6-1m | 936K | 1.00M | **≥ 1.00M** | — | — |
| gpt-5.4 | 272K | 400K | **400K** | 1M | 272K |
| gpt-5.4-mini | 272K | 400K | **400K** | 1M | 272K |
| gpt-5.5 | 272K | 400K | **400K** | 1M | 272K |
| gpt-5.2-codex | 272K | 400K | **400K** | 1M | 272K |
| gpt-5.3-codex | 272K | 400K | **400K** | 1M | 272K |
| claude-sonnet-4.5 | 168K | 200K | **272K** | 400K | 168K |
| claude-opus-4.5 | 168K | 200K | **272K** | 400K | 168K |
| claude-haiku-4.5 | 136K | 200K | **272K** | 400K | 136K |
| gemini-3.1-pro-preview | 136K | 200K | 200K | 272K | 136K |
| claude-opus-4.6 | 168K | 200K | 200K | 272K | 168K |
| claude-opus-4.7 | 168K | 200K | 200K | 272K | 168K |
| claude-sonnet-4.6 | 168K | 200K | 200K | 272K | 168K |
| claude-sonnet-4 | 128K | 216K | 128K | 200K | 128K |
| gemini-2.5-pro | 128K | 128K | 128K | 200K | 128K |
| gemini-3-flash-preview | 128K | 128K | 128K | 200K | 128K |
| gpt-4.1 | 128K | 128K | 128K | 200K | 128K |
| gpt-5-mini | 128K | 264K | 128K | 200K | 128K |
| gpt-4o | 64K | 128K | 64K | 128K | 64K |
| gpt-5.2 | 272K | 400K | inconclusive | — | reasoning-only model; ignores `max_completion_tokens=1`, every probe times out |

## Endpoint requirements

A few models reject `/chat/completions` with `unsupported_api_for_model`
and must be probed via `/responses`:

- `gpt-5.5`
- `gpt-5.2-codex`
- `gpt-5.3-codex`

Newer GPT models (gpt-5.x family) require `max_completion_tokens` in
place of the legacy `max_tokens` parameter on `/chat/completions`.

## Key findings

1. **gpt-5.4 / 5.4-mini / 5.5 / 5.2-codex / 5.3-codex** accept up to
   400K despite a 272K catalog claim — catalog `max_prompt_tokens` is a
   soft recommendation, not the hard ceiling.
2. **Claude 4.5 family** (opus/sonnet/haiku) all accept 272K despite
   168K/136K catalog claims.
3. **Claude 4.6 / 4.7 family** regressed to a hard 200K ceiling — 272K
   is rejected. Routing decisions should not assume parity with 4.5.
4. **claude-opus-4.6-1m** accepts the full 1M without truncation.
5. The error string `limit of <N>` always echoes the catalog
   `max_prompt_tokens` value, not the real ceiling. Pre-flight rejection
   logic that trusts that number will be **overly conservative** for
   every model in the top half of this table.

## Re-running

```bash
# All models, default ladder
uv run python scripts/test_context_sizes.py

# Subset
uv run python scripts/test_context_sizes.py --models gpt-5.4 claude-opus-4.7

# Custom ladder
uv run python scripts/test_context_sizes.py --steps 100000 250000 500000
```

Re-run after any of: catalog refresh on Copilot's side, addition of new
models, or changes to upstream rate-limit / context-policy behaviour.
