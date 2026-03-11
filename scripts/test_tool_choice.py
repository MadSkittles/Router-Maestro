#!/usr/bin/env python3
"""Diagnostic script: test tool_choice support across Copilot API and Router-Maestro.

Tests whether GitHub Copilot API properly handles various tool_choice formats
by sending identical requests directly to Copilot and via Router-Maestro proxy.

Usage:
    uv run python scripts/test_tool_choice.py [--rm-url URL] [--rm-key KEY] [--model MODEL]
"""

import argparse
import asyncio
import json
import os
import sys
import time

import httpx

# Add project root to path so we can import router_maestro
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router_maestro.auth import AuthManager
from router_maestro.auth.github_oauth import get_copilot_token
from router_maestro.auth.storage import OAuthCredential

COPILOT_CHAT_URL = "https://api.githubcopilot.com/chat/completions"

# Common tool definition used across all tests
AGENT_OUTPUT_TOOL = {
    "type": "function",
    "function": {
        "name": "AgentOutput",
        "description": "Output the action to perform",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Your reasoning"},
                "action": {
                    "type": "string",
                    "enum": ["click", "wait", "done"],
                    "description": "Action type",
                },
                "params": {"type": "object", "description": "Action parameters"},
            },
            "required": ["thought", "action", "params"],
        },
    },
}

TOOL_CHOICE_VARIANTS: list[tuple[str, object]] = [
    ("auto", "auto"),
    ("required", "required"),
    (
        '{"type":"function","function":{"name":"AgentOutput"}}',
        {"type": "function", "function": {"name": "AgentOutput"}},
    ),
]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Always use tools when available."},
    {"role": "user", "content": "Click the plus button."},
]


def _build_payload(model: str, tool_choice: object) -> dict:
    return {
        "model": model,
        "temperature": 0.7,
        "messages": MESSAGES,
        "tools": [AGENT_OUTPUT_TOOL],
        "tool_choice": tool_choice,
        "stream": False,
    }


async def _get_copilot_token(auth_manager: AuthManager) -> str:
    """Get a fresh Copilot bearer token."""
    cred = auth_manager.get_credential("github-copilot")
    if not cred or not isinstance(cred, OAuthCredential):
        raise RuntimeError(
            "Not authenticated with GitHub Copilot. Run: router-maestro auth login copilot"
        )

    async with httpx.AsyncClient() as client:
        token_resp = await get_copilot_token(client, cred.refresh)
    return token_resp.token


def _copilot_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.0.0",
        "Copilot-Integration-Id": "vscode-chat",
    }


def _extract_result(data: dict) -> dict:
    """Extract relevant fields from a chat completion response."""
    choices = data.get("choices", [])
    if not choices:
        return {"finish_reason": "NO_CHOICES", "tool_calls": None, "content_preview": None}

    # Merge across choices (Copilot multi-choice behavior)
    content = None
    tool_calls = []
    finish_reason = "stop"

    for choice in choices:
        msg = choice.get("message", {})
        if content is None and msg.get("content"):
            content = msg["content"]
        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])
        fr = choice.get("finish_reason")
        if fr:
            finish_reason = fr

    content_preview = None
    if content:
        content_preview = content[:80] + ("..." if len(content) > 80 else "")

    tc_summary = None
    if tool_calls:
        names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        tc_summary = f"{len(tool_calls)} call(s): {', '.join(names)}"

    return {
        "finish_reason": finish_reason,
        "tool_calls": tc_summary,
        "content_preview": content_preview,
        "num_choices": len(choices),
    }


async def _test_direct_copilot(
    client: httpx.AsyncClient,
    token: str,
    model: str,
    label: str,
    tool_choice: object,
) -> dict:
    """Send request directly to Copilot API."""
    payload = _build_payload(model, tool_choice)
    try:
        resp = await client.post(
            COPILOT_CHAT_URL,
            json=payload,
            headers=_copilot_headers(token),
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return _extract_result(data)
    except httpx.HTTPStatusError as e:
        body = e.response.text[:200] if e.response else "N/A"
        return {"error": f"HTTP {e.response.status_code}: {body}"}
    except Exception as e:
        return {"error": str(e)[:200]}


async def _test_via_router_maestro(
    client: httpx.AsyncClient,
    rm_url: str,
    rm_key: str,
    model: str,
    label: str,
    tool_choice: object,
) -> dict:
    """Send request via Router-Maestro proxy."""
    payload = _build_payload(model, tool_choice)
    headers = {
        "Authorization": f"Bearer {rm_key}",
        "Content-Type": "application/json",
    }
    url = f"{rm_url.rstrip('/')}/api/openai/v1/chat/completions"
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return _extract_result(data)
    except httpx.HTTPStatusError as e:
        body = e.response.text[:200] if e.response else "N/A"
        return {"error": f"HTTP {e.response.status_code}: {body}"}
    except Exception as e:
        return {"error": str(e)[:200]}


def _fmt_result(result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error'][:60]}"
    parts = [f"fr={result['finish_reason']}"]
    if result.get("tool_calls"):
        parts.append(f"tc={result['tool_calls']}")
    else:
        parts.append("tc=None")
    if result.get("num_choices", 1) > 1:
        parts.append(f"choices={result['num_choices']}")
    return " | ".join(parts)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test tool_choice across Copilot API and Router-Maestro")
    parser.add_argument("--rm-url", default=os.environ.get("RM_URL", "http://localhost:8080"),
                        help="Router-Maestro base URL (default: http://localhost:8080)")
    parser.add_argument("--rm-key", default=os.environ.get("ROUTER_MAESTRO_API_KEY", ""),
                        help="Router-Maestro API key")
    parser.add_argument("--model", default="gpt-4o", help="Model to test (default: gpt-4o)")
    parser.add_argument("--skip-rm", action="store_true", help="Skip Router-Maestro tests")
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"  tool_choice Diagnostic — model: {args.model}")
    print(f"{'='*80}\n")

    # Get Copilot token
    print("[1/3] Authenticating with GitHub Copilot...")
    auth_manager = AuthManager()
    token = await _get_copilot_token(auth_manager)
    print(f"  -> Token obtained (expires in ~30min)\n")

    # Run tests
    print("[2/3] Testing tool_choice variants...\n")

    results: list[dict] = []

    async with httpx.AsyncClient() as client:
        for label, tool_choice in TOOL_CHOICE_VARIANTS:
            row: dict = {"label": label}

            # Direct Copilot
            print(f"  Testing: tool_choice={label}")
            print(f"    Direct Copilot API... ", end="", flush=True)
            t0 = time.monotonic()
            row["direct"] = await _test_direct_copilot(client, token, args.model, label, tool_choice)
            dt = time.monotonic() - t0
            print(f"done ({dt:.1f}s)")

            # Via Router-Maestro
            if not args.skip_rm and args.rm_key:
                print(f"    Via Router-Maestro... ", end="", flush=True)
                t0 = time.monotonic()
                row["rm"] = await _test_via_router_maestro(
                    client, args.rm_url, args.rm_key, args.model, label, tool_choice
                )
                dt = time.monotonic() - t0
                print(f"done ({dt:.1f}s)")
            else:
                row["rm"] = {"skipped": True}

            results.append(row)
            print()

    # Print results table
    print(f"[3/3] Results\n")

    label_w = max(len(r["label"]) for r in results)
    label_w = max(label_w, len("tool_choice"))

    has_rm = any("skipped" not in r.get("rm", {}) for r in results)

    # Header
    header = f"  {'tool_choice':<{label_w}}  |  {'Direct Copilot API':<50}"
    if has_rm:
        header += f"  |  {'Via Router-Maestro':<50}"
    print(header)
    print(f"  {'-'*label_w}--+--{'-'*50}", end="")
    if has_rm:
        print(f"--+--{'-'*50}", end="")
    print()

    # Rows
    for row in results:
        direct_str = _fmt_result(row["direct"])
        line = f"  {row['label']:<{label_w}}  |  {direct_str:<50}"
        if has_rm and "skipped" not in row.get("rm", {}):
            rm_str = _fmt_result(row["rm"])
            line += f"  |  {rm_str:<50}"
        print(line)

    print()

    # Diagnosis
    print("Diagnosis:")
    for row in results:
        direct = row["direct"]
        if "error" in direct:
            print(f"  [{row['label']}] Direct API error — Copilot may not support this format")
        elif direct.get("tool_calls"):
            print(f"  [{row['label']}] Direct API returned tool calls — working correctly")
        else:
            print(f"  [{row['label']}] Direct API returned NO tool calls (fr={direct.get('finish_reason')})")
            if direct.get("content_preview"):
                print(f"    Content: {direct['content_preview']}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
