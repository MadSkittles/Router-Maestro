#!/usr/bin/env python3
"""Probe Copilot Anthropic models for actual wire-level behavior.

Directly calls the Copilot /v1/messages endpoint for each Claude model to discover:
  1. Which thinking shapes are accepted (enabled / adaptive / disabled)
  2. Which effort values are accepted (low / medium / high / xhigh / max)
  3. Whether mid-conversation system messages are accepted
  4. Which anthropic-beta tokens are accepted / rejected

Uses the local Router-Maestro auth storage (requires prior `router-maestro auth login copilot`).

Usage:
    uv run python scripts/probe_model_profiles.py [--models claude-opus-4.8 ...]
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from router_maestro.auth.github_oauth import CopilotTokenResponse, get_copilot_token
from router_maestro.auth.storage import AuthStorage, AuthType

COPILOT_API = "https://api.githubcopilot.com"
MESSAGES_PATH = "/v1/messages"

# All Claude models known on Copilot
DEFAULT_MODELS = [
    "claude-haiku-4.5",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-sonnet-5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.8",
]

EFFORTS = ["low", "medium", "high", "xhigh", "max"]

THINKING_SHAPES = {
    "enabled": {"type": "enabled", "budget_tokens": 4096},
    "adaptive": {"type": "adaptive"},
    "disabled": {"type": "disabled"},
}

BETA_TOKENS_TO_TEST = [
    "context-1m-2025-08-07",
    "mid-conversation-system-2026-04-07",
    "output-128k-2025-02-19",
    "interleaved-thinking-2025-05-14",
    "advisor-tool-2025-04-02",
]

MINIMAL_MESSAGES = [{"role": "user", "content": "Say hi"}]


@dataclass
class ProbeResult:
    model: str
    accepted_efforts: list[str] = field(default_factory=list)
    rejected_efforts: dict[str, str] = field(default_factory=dict)
    accepted_thinking: list[str] = field(default_factory=list)
    rejected_thinking: dict[str, str] = field(default_factory=dict)
    mid_conv_system: bool | None = None
    mid_conv_system_note: str | None = None
    mid_conv_system_details: dict = field(default_factory=dict)
    accepted_betas: list[str] = field(default_factory=list)
    rejected_betas: dict[str, str] = field(default_factory=dict)
    alive: bool = True
    alive_error: str | None = None
    # Copilot /models advertised capabilities (for comparison)
    catalog_capabilities: dict = field(default_factory=dict)


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Editor-Version": "vscode/1.95.0",
        "Editor-Plugin-Version": "copilot-chat/0.26.7",
        "Copilot-Integration-Id": "vscode-chat",
        "User-Agent": "GitHubCopilotChat/0.26.7",
        "OpenAI-Intent": "conversation-panel",
        "X-GitHub-Api-Version": "2025-04-01",
        "X-Vscode-User-Agent-Library-Version": "electron-fetch",
        "X-Initiator": "user",
    }


async def _get_token() -> str:
    """Get a fresh Copilot bearer token from local auth storage."""
    storage = AuthStorage.load()
    cred = storage.credentials.get("github-copilot")
    if not cred or cred.type != AuthType.OAUTH:
        print("ERROR: No github-copilot OAuth credential found.")
        print("  Run: uv run router-maestro auth login copilot")
        sys.exit(1)

    async with httpx.AsyncClient() as client:
        resp = await get_copilot_token(client, cred.refresh)
    return resp.token


async def _send(
    client: httpx.AsyncClient,
    token: str,
    model: str,
    *,
    messages: list[dict] | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
    beta_tokens: list[str] | None = None,
    max_tokens: int = 128,
) -> tuple[int, dict]:
    """Send a minimal /v1/messages request and return (status, body)."""
    headers = _headers(token)
    if beta_tokens:
        headers["anthropic-beta"] = ",".join(beta_tokens)

    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages or MINIMAL_MESSAGES,
        "stream": False,
    }

    if thinking is not None:
        body["thinking"] = thinking
        if thinking.get("type") == "enabled":
            body["max_tokens"] = max(max_tokens, thinking.get("budget_tokens", 4096) + 128)

    if effort is not None:
        body["output_config"] = {"effort": effort}

    resp = await client.post(
        f"{COPILOT_API}{MESSAGES_PATH}",
        headers=headers,
        json=body,
        timeout=60.0,
    )
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text[:500]}
    return resp.status_code, data


def _error_snippet(data: dict) -> str:
    """Extract a concise error description from a response body."""
    if "error" in data:
        err = data["error"]
        if isinstance(err, dict):
            return err.get("message", str(err))[:200]
        return str(err)[:200]
    return json.dumps(data)[:200]


async def probe_liveness(client: httpx.AsyncClient, token: str, model: str) -> tuple[bool, str]:
    """Check if the model is alive (bare request, no thinking/effort)."""
    status, data = await _send(client, token, model)
    if status == 200:
        return True, ""
    return False, f"[{status}] {_error_snippet(data)}"


async def probe_thinking(
    client: httpx.AsyncClient, token: str, model: str
) -> tuple[list[str], dict[str, str]]:
    """Probe which thinking shapes the model accepts."""
    accepted = []
    rejected = {}
    for shape_name, thinking_obj in THINKING_SHAPES.items():
        status, data = await _send(client, token, model, thinking=thinking_obj)
        if status == 200:
            accepted.append(shape_name)
        else:
            rejected[shape_name] = _error_snippet(data)
        await asyncio.sleep(0.3)
    return accepted, rejected


async def probe_efforts(
    client: httpx.AsyncClient, token: str, model: str, thinking_shape: dict | None
) -> tuple[list[str], dict[str, str]]:
    """Probe which effort values the model accepts.

    Uses the model's accepted thinking shape (if any) because effort often
    requires adaptive thinking to be set.
    """
    accepted = []
    rejected = {}
    for effort in EFFORTS:
        status, data = await _send(client, token, model, thinking=thinking_shape, effort=effort)
        if status == 200:
            accepted.append(effort)
        else:
            rejected[effort] = _error_snippet(data)
        await asyncio.sleep(0.3)
    return accepted, rejected


async def probe_mid_conv_system(
    client: httpx.AsyncClient, token: str, model: str
) -> tuple[bool | None, str | None, dict]:
    """Probe whether mid-conversation system messages are accepted.

    Tests multiple placements:
      - legal:   [user, system, user]  (system follows user, precedes user→assistant)
      - illegal: [user, assistant, system, user]  (system follows assistant)

    Returns (accepted, error_snippet, details_dict).
    """
    details = {}

    # Legal placement: system after user (the 4.8+ rule)
    legal_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hi"},
    ]
    status, data = await _send(client, token, model, messages=legal_messages)
    details["legal_placement"] = {"status": status, "ok": status == 200}
    if status != 200:
        details["legal_placement"]["error"] = _error_snippet(data)

    await asyncio.sleep(0.3)

    # Illegal placement: system after assistant
    illegal_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hi"},
    ]
    status2, data2 = await _send(client, token, model, messages=illegal_messages)
    details["illegal_placement"] = {"status": status2, "ok": status2 == 200}
    if status2 != 200:
        details["illegal_placement"]["error"] = _error_snippet(data2)

    # Interpretation:
    #   legal=200 + illegal=400 with placement error → supports with placement rules
    #   legal=200 + illegal=200 → supports unconditionally
    #   legal=400 "Unexpected role" → does not support at all
    #   legal=400 placement error → supports with stricter rules than we tested
    legal_ok = details["legal_placement"]["ok"]
    illegal_ok = details["illegal_placement"]["ok"]

    if legal_ok:
        if illegal_ok:
            return True, "unconditional", details
        else:
            return True, "placement-rules", details
    else:
        err = details["legal_placement"].get("error", "")
        if "Unexpected role" in err:
            return False, err, details
        else:
            return None, f"ambiguous: {err}", details


async def probe_betas(
    client: httpx.AsyncClient, token: str, model: str
) -> tuple[list[str], dict[str, str]]:
    """Probe which anthropic-beta tokens are accepted."""
    accepted = []
    rejected = {}
    for beta in BETA_TOKENS_TO_TEST:
        status, data = await _send(client, token, model, beta_tokens=[beta])
        if status == 200:
            accepted.append(beta)
        else:
            rejected[beta] = _error_snippet(data)
        await asyncio.sleep(0.3)
    return accepted, rejected


async def fetch_catalog(client: httpx.AsyncClient, token: str) -> dict[str, dict]:
    """Fetch Copilot /models catalog and return capabilities keyed by model id."""
    headers = _headers(token)
    resp = await client.get(f"{COPILOT_API}/models", headers=headers, timeout=30.0)
    if resp.status_code != 200:
        print(f"  WARNING: /models returned {resp.status_code}")
        return {}
    data = resp.json()
    models = data.get("data", data) if isinstance(data, dict) else data
    catalog = {}
    for m in models:
        mid = m.get("id", "")
        caps = m.get("capabilities", {})
        catalog[mid] = {
            "supports": caps.get("supports", {}),
            "limits": caps.get("limits", {}),
            "reasoning_effort_values": [
                v.get("value") for v in (caps.get("supports", {}).get("reasoning_effort", {}).get("values", []))
            ] if isinstance(caps.get("supports", {}).get("reasoning_effort"), dict) else None,
        }
    return catalog


async def probe_model(client: httpx.AsyncClient, token: str, model: str, catalog: dict) -> ProbeResult:
    """Run full probe suite for a single model."""
    result = ProbeResult(model=model)
    result.catalog_capabilities = catalog.get(model, {})

    # 1. Liveness
    print(f"  [{model}] liveness...", end=" ", flush=True)
    alive, err = await probe_liveness(client, token, model)
    result.alive = alive
    result.alive_error = err
    if not alive:
        print(f"DEAD ({err})")
        return result
    print("OK")

    # 2. Thinking shapes
    print(f"  [{model}] thinking shapes...", end=" ", flush=True)
    result.accepted_thinking, result.rejected_thinking = await probe_thinking(client, token, model)
    print(f"{result.accepted_thinking}")

    # 3. Effort values — use the best accepted thinking shape
    thinking_for_effort = None
    if "adaptive" in result.accepted_thinking:
        thinking_for_effort = THINKING_SHAPES["adaptive"]
    elif "enabled" in result.accepted_thinking:
        thinking_for_effort = THINKING_SHAPES["enabled"]

    print(f"  [{model}] effort values (with thinking={thinking_for_effort and thinking_for_effort.get('type')})...", end=" ", flush=True)
    result.accepted_efforts, result.rejected_efforts = await probe_efforts(
        client, token, model, thinking_for_effort
    )
    print(f"{result.accepted_efforts}")

    # 4. Mid-conversation system (with placement-aware probing)
    print(f"  [{model}] mid-conv system...", end=" ", flush=True)
    result.mid_conv_system, result.mid_conv_system_note, result.mid_conv_system_details = (
        await probe_mid_conv_system(client, token, model)
    )
    status_str = "YES" if result.mid_conv_system else ("NO" if result.mid_conv_system is False else "?")
    note = f" ({result.mid_conv_system_note})" if result.mid_conv_system_note else ""
    print(f"{status_str}{note}")

    # 5. Beta tokens
    print(f"  [{model}] beta tokens...", end=" ", flush=True)
    result.accepted_betas, result.rejected_betas = await probe_betas(client, token, model)
    print(f"{result.accepted_betas}")

    return result


def print_summary(results: list[ProbeResult]) -> None:
    """Print a comparison table of all probed models."""
    print("\n" + "=" * 80)
    print("PROBE RESULTS SUMMARY")
    print("=" * 80)

    for r in results:
        print(f"\n{'─' * 60}")
        print(f"Model: {r.model}")
        if not r.alive:
            print(f"  STATUS: DEAD — {r.alive_error}")
            continue
        print(f"  Thinking shapes:  {', '.join(r.accepted_thinking) or 'NONE'}")
        if r.rejected_thinking:
            for shape, err in r.rejected_thinking.items():
                print(f"    rejected {shape}: {err}")
        print(f"  Effort values:    {', '.join(r.accepted_efforts) or 'NONE (field rejected)'}")
        if r.rejected_efforts:
            for effort, err in r.rejected_efforts.items():
                print(f"    rejected {effort}: {err}")
        status_str = "YES" if r.mid_conv_system else ("NO" if r.mid_conv_system is False else "AMBIGUOUS")
        print(f"  Mid-conv system:  {status_str} ({r.mid_conv_system_note or ''})")
        if r.mid_conv_system_details:
            for placement, detail in r.mid_conv_system_details.items():
                ok = "✓" if detail.get("ok") else "✗"
                err_msg = f" — {detail.get('error', '')}" if not detail.get("ok") else ""
                print(f"    {placement}: {ok}{err_msg}")
        print(f"  Beta accepted:    {', '.join(r.accepted_betas) or 'NONE'}")
        if r.rejected_betas:
            for beta, err in r.rejected_betas.items():
                print(f"    rejected {beta}: {err}")

    # Comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE (probe results)")
    print(f"{'=' * 80}")
    header = f"{'Model':<22} {'Thinking':<25} {'Efforts':<30} {'MidSys':<14} {'Betas'}"
    print(header)
    print("─" * len(header))
    for r in results:
        if not r.alive:
            print(f"{r.model:<22} DEAD")
            continue
        thinking = ",".join(r.accepted_thinking) or "none"
        efforts = ",".join(r.accepted_efforts) or "none"
        midsys = "Y(rules)" if r.mid_conv_system else ("N" if r.mid_conv_system is False else "?")
        betas = str(len(r.accepted_betas)) + "/" + str(len(BETA_TOKENS_TO_TEST))
        print(f"{r.model:<22} {thinking:<25} {efforts:<30} {midsys:<14} {betas}")

    # Catalog vs probe discrepancies
    print(f"\n{'=' * 80}")
    print("CATALOG vs PROBE DISCREPANCIES")
    print(f"{'=' * 80}")
    any_discrepancy = False
    for r in results:
        if not r.alive or not r.catalog_capabilities:
            continue
        discrepancies = []
        cat_efforts = r.catalog_capabilities.get("reasoning_effort_values")
        if cat_efforts is not None:
            cat_set = set(cat_efforts)
            probe_set = set(r.accepted_efforts)
            if cat_set != probe_set:
                only_catalog = cat_set - probe_set
                only_probe = probe_set - cat_set
                parts = []
                if only_catalog:
                    parts.append(f"catalog says {only_catalog} but rejected")
                if only_probe:
                    parts.append(f"probe found {only_probe} but not in catalog")
                discrepancies.append(f"  effort: {'; '.join(parts)}")

        cat_thinking = r.catalog_capabilities.get("supports", {}).get("thinking")
        if cat_thinking is not None:
            cat_has_adaptive = bool(cat_thinking) if isinstance(cat_thinking, bool) else "adaptive" in str(cat_thinking)
            probe_has_adaptive = "adaptive" in r.accepted_thinking
            if cat_has_adaptive and not probe_has_adaptive:
                discrepancies.append(f"  thinking: catalog advertises adaptive but probe rejected")
            elif not cat_has_adaptive and probe_has_adaptive:
                discrepancies.append(f"  thinking: catalog does NOT advertise adaptive but probe accepted")

        if discrepancies:
            any_discrepancy = True
            print(f"\n{r.model}:")
            for d in discrepancies:
                print(d)

    if not any_discrepancy:
        print("\n  No discrepancies found between /models catalog and probe results.")


def export_json(results: list[ProbeResult], path: Path) -> None:
    """Export results as JSON for further analysis."""
    out = []
    for r in results:
        out.append({
            "model": r.model,
            "alive": r.alive,
            "alive_error": r.alive_error,
            "accepted_thinking": r.accepted_thinking,
            "rejected_thinking": r.rejected_thinking,
            "accepted_efforts": r.accepted_efforts,
            "rejected_efforts": r.rejected_efforts,
            "mid_conv_system": r.mid_conv_system,
            "mid_conv_system_note": r.mid_conv_system_note,
            "mid_conv_system_details": r.mid_conv_system_details,
            "accepted_betas": r.accepted_betas,
            "rejected_betas": r.rejected_betas,
            "catalog_capabilities": r.catalog_capabilities,
        })
    path.write_text(json.dumps(out, indent=2))
    print(f"\nResults exported to: {path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Copilot Anthropic models for wire-level behavior")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to probe (default: all known: {', '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Export results as JSON to this file",
    )
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS

    print("Obtaining Copilot token...")
    token = await _get_token()
    print("Token obtained.\n")

    print("Fetching /models catalog for comparison...")
    async with httpx.AsyncClient() as client:
        catalog = await fetch_catalog(client, token)
    print(f"Catalog has {len(catalog)} models.\n")

    results: list[ProbeResult] = []
    async with httpx.AsyncClient() as client:
        for model in models:
            print(f"\n{'━' * 60}")
            print(f"Probing: {model}")
            print(f"{'━' * 60}")
            result = await probe_model(client, token, model, catalog)
            results.append(result)
            await asyncio.sleep(1.0)

    print_summary(results)

    if args.output:
        export_json(results, args.output)
    else:
        default_out = Path(__file__).parent / "probe_results.json"
        export_json(results, default_out)


if __name__ == "__main__":
    asyncio.run(main())
