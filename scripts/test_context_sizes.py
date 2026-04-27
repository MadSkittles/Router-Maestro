"""Probe the actual prompt-token ceiling of every GitHub Copilot model.

Bypasses Router-Maestro and calls api.githubcopilot.com directly using the
auth manager from this repo. For each model we walk a step ladder
(32K -> 1M) and send a synthetic prompt sized to that token count
(approximated as 4 chars/token). The first rejection terminates that
model and we record the catalog limit, the largest accepted size, and
the limit reported in the rejection message (when present).

Usage:
    uv run python scripts/test_context_sizes.py
    uv run python scripts/test_context_sizes.py --models gpt-4o claude-opus-4.6-1m
    uv run python scripts/test_context_sizes.py --steps 32000 128000 272000 1000000
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass, field

import httpx

sys.path.insert(0, "src")

from router_maestro.providers.copilot import (  # noqa: E402
    COPILOT_CHAT_URL,
    COPILOT_RESPONSES_URL,
    CopilotProvider,
)

DEFAULT_STEPS = [32_000, 64_000, 128_000, 200_000, 272_000, 400_000, 1_000_000]

# Copilot-side limit messages we care about.
LIMIT_RE = re.compile(r"limit of (\d+)", re.IGNORECASE)
COUNT_RE = re.compile(r"prompt token count of (\d+)", re.IGNORECASE)


@dataclass
class Result:
    model: str
    catalog_max_prompt: int | None
    catalog_max_context: int | None
    largest_accepted: int | None = None
    first_rejected: int | None = None
    reported_limit: int | None = None
    reported_count: int | None = None
    notes: list[str] = field(default_factory=list)


def make_prompt(target_tokens: int) -> str:
    # ~4 chars per token is a reasonable upper bound for English-ish text.
    # Use a repeating short word + space so tokenizers see real word boundaries.
    word = "lorem "  # 6 chars, ~1.5 tokens; aim slightly above target.
    chars = target_tokens * 4
    n = chars // len(word) + 1
    return (word * n).rstrip()


async def probe_one(
    provider: CopilotProvider,
    model_id: str,
    target_tokens: int,
    client: httpx.AsyncClient,
) -> tuple[bool, str]:
    """Return (accepted, message). accepted=True means upstream did not
    reject for context overflow (it may still 4xx for other reasons).
    Retries once on transient httpx errors (h2 protocol, connection reset)."""
    last_err: str | None = None
    for attempt in range(2):
        try:
            return await _probe_once(provider, model_id, target_tokens, client)
        except _TransientError as e:
            last_err = str(e)
            await asyncio.sleep(2)
    return False, f"http-error (after retry): {last_err}"


class _TransientError(Exception):
    pass


async def _probe_once(
    provider: CopilotProvider,
    model_id: str,
    target_tokens: int,
    client: httpx.AsyncClient,
) -> tuple[bool, str]:
    await provider.ensure_token()
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": make_prompt(target_tokens)}],
        "max_completion_tokens": 1,
        "stream": False,
    }
    try:
        resp = await client.post(
            COPILOT_CHAT_URL,
            headers=provider._get_headers(),
            json=body,
        )
    except httpx.HTTPError as e:
        raise _TransientError(f"{type(e).__name__}: {e}")

    # Some older models still require the legacy `max_tokens` parameter.
    if (
        resp.status_code == 400
        and "Unsupported parameter" in resp.text
        and "'max_completion_tokens'" in resp.text
    ):
        body.pop("max_completion_tokens")
        body["max_tokens"] = 1
        try:
            resp = await client.post(
                COPILOT_CHAT_URL,
                headers=provider._get_headers(),
                json=body,
            )
        except httpx.HTTPError as e:
            raise _TransientError(f"{type(e).__name__}: {e}")

    # Fall back to /responses for models gated to that endpoint.
    if (
        resp.status_code == 400
        and "unsupported_api_for_model" in resp.text
    ):
        rbody = {
            "model": model_id,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": make_prompt(target_tokens)}]}
            ],
            "max_output_tokens": 16,
            "stream": False,
        }
        try:
            resp = await client.post(
                COPILOT_RESPONSES_URL,
                headers=provider._get_headers(),
                json=rbody,
            )
        except httpx.HTTPError as e:
            raise _TransientError(f"{type(e).__name__}: {e}")

    if resp.status_code < 400:
        return True, "ok"

    text = resp.text
    if "model_max_prompt_tokens_exceeded" in text or "exceeds the limit" in text:
        return False, text
    # Other 4xx (rate limit, model unsupported, etc.) — surface but don't
    # treat as a hard ceiling.
    return False, f"status={resp.status_code}: {text[:200]}"


async def probe_model(
    provider: CopilotProvider,
    model: object,
    steps: list[int],
) -> Result:
    res = Result(
        model=model.id,
        catalog_max_prompt=getattr(model, "max_prompt_tokens", None),
        catalog_max_context=getattr(model, "max_context_window_tokens", None),
    )
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        for tokens in steps:
            print(f"  -> {model.id} @ {tokens:>9,} tokens ... ", end="", flush=True)
            accepted, msg = await probe_one(provider, model.id, tokens, client)
            if accepted:
                print("ok")
                res.largest_accepted = tokens
                continue
            print("REJECTED")
            res.first_rejected = tokens
            m = LIMIT_RE.search(msg)
            if m:
                res.reported_limit = int(m.group(1))
            m = COUNT_RE.search(msg)
            if m:
                res.reported_count = int(m.group(1))
            if "model_max_prompt_tokens_exceeded" not in msg and "exceeds the limit" not in msg:
                res.notes.append(msg[:240])
            break
    return res


def fmt(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def print_table(results: list[Result]) -> None:
    cols = [
        ("Model", 38),
        ("Catalog prompt", 14),
        ("Catalog ctx", 12),
        ("Largest OK", 11),
        ("First reject", 13),
        ("Reported limit", 14),
        ("Notes", 40),
    ]
    line = " | ".join(f"{h:<{w}}" for h, w in cols)
    print()
    print(line)
    print("-" * len(line))
    for r in sorted(results, key=lambda r: r.model):
        notes = "; ".join(r.notes)[:38]
        row = [
            r.model[:38],
            fmt(r.catalog_max_prompt),
            fmt(r.catalog_max_context),
            fmt(r.largest_accepted),
            fmt(r.first_rejected),
            fmt(r.reported_limit),
            notes,
        ]
        print(" | ".join(f"{c:<{w}}" for c, (_, w) in zip(row, cols)))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="Only test these model ids")
    ap.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=DEFAULT_STEPS,
        help="Prompt-token ladder",
    )
    args = ap.parse_args()

    provider = CopilotProvider()
    if not provider.is_authenticated():
        print("Not authenticated — run: router-maestro auth login copilot", file=sys.stderr)
        return 2

    models = await provider.list_models()
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m.id in wanted]
        if not models:
            print(f"No matching models. Available: {[m.id for m in await provider.list_models()]}")
            return 2

    print(f"Probing {len(models)} models against steps: {[fmt(s) for s in args.steps]}")
    results: list[Result] = []
    for m in models:
        print(f"\n[{m.id}] catalog max_prompt={fmt(getattr(m, 'max_prompt_tokens', None))}")
        try:
            results.append(await probe_model(provider, m, args.steps))
        except Exception as e:  # noqa: BLE001
            print(f"  !! error: {e}")
            results.append(
                Result(
                    model=m.id,
                    catalog_max_prompt=getattr(m, "max_prompt_tokens", None),
                    catalog_max_context=getattr(m, "max_context_window_tokens", None),
                    notes=[f"crash: {e}"],
                )
            )

    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
