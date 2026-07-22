"""Per-request audit tracing.

When enabled, captures the request/response lifecycle for each API call as JSON
files for debugging. Each request gets a directory under the trace directory;
early failures may omit upstream records, while retries and fallback add
independently numbered upstream request and response observations:

    {trace_dir}/{request_id}/
        inbound.json        — client request (method, path, headers, body)
        upstream.json, upstream_2.json, ...
                           — requests sent to providers, in observation order
        upstream_resp.json, upstream_resp_2.json, ...
                           — responses received, independently numbered
        outbound.json       — response sent to client

Activation:
    - Set ROUTER_MAESTRO_TRACE=1 env var, OR
    - Set audit.enabled=true in priorities.json

Trace directory defaults to ~/.local/share/router-maestro/traces/
"""

import asyncio
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from router_maestro.config.paths import get_data_dir
from router_maestro.utils import get_logger

if TYPE_CHECKING:
    from router_maestro.providers.base import TerminalOutcome

logger = get_logger("utils.audit")

_SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key", "x-goog-api-key"})
_SENSITIVE_FIELD_KEYS = frozenset(
    {
        "apikey",
        "accesstoken",
        "idtoken",
        "clientsecret",
        "refreshtoken",
        "authorization",
        "password",
        "secret",
        "token",
        "privatekey",
    }
)


def is_tracing_enabled(audit_config_enabled: bool = False) -> bool:
    """Check if tracing is active (env var or config)."""
    return audit_config_enabled or os.environ.get("ROUTER_MAESTRO_TRACE", "") == "1"


def get_trace_dir(config_trace_dir: str | None = None) -> Path:
    """Resolve the trace output directory."""
    if config_trace_dir:
        return Path(config_trace_dir)
    return get_data_dir() / "traces"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact sensitive header values."""
    return {k: ("***" if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()}


def _safe_json(obj: Any) -> Any:
    """Make an object JSON-serializable with graceful fallback."""
    if isinstance(obj, bytes):
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return f"<{len(obj)} bytes>"
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(i) for i in obj]
    return str(obj)


def _redact_payload(obj: Any) -> Any:
    """Return a detached value with credential-like payload fields removed."""
    value = _safe_json(obj)
    if isinstance(value, dict):
        return {
            key: (
                "***"
                if "".join(character for character in str(key).casefold() if character.isalnum())
                in _SENSITIVE_FIELD_KEYS
                else _redact_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


def _write_records(trace_dir: Path, records: dict[str, Any]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    for name, data in records.items():
        path = trace_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str))


class AuditTrace:
    """Collects per-request trace data and writes to disk on flush."""

    def __init__(self, request_id: str, trace_dir: Path):
        self._request_id = request_id
        self._dir = trace_dir / request_id
        self._records: dict[str, Any] = {}
        self._start_time = time.time()

    def _append_record(self, base_name: str, record: dict[str, Any]) -> None:
        index = 1
        name = base_name
        while name in self._records:
            index += 1
            name = f"{base_name}_{index}"
        self._records[name] = record

    def record_inbound(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: Any,
    ) -> None:
        self._records["inbound"] = {
            "timestamp": self._start_time,
            "request_id": self._request_id,
            "method": method,
            "path": path,
            "headers": _redact_headers(headers),
            "body": _redact_payload(body),
        }

    def record_upstream(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any,
    ) -> None:
        self._append_record(
            "upstream",
            {
                "timestamp": time.time(),
                "request_id": self._request_id,
                "method": method,
                "url": url,
                "headers": _redact_headers(headers),
                "body": _redact_payload(body),
            },
        )

    def record_upstream_response(
        self,
        status: int,
        headers: dict[str, str],
        body: Any = None,
        *,
        stream_summary: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": time.time(),
            "request_id": self._request_id,
            "status": status,
            "headers": _redact_headers(headers),
        }
        if stream_summary:
            record["stream_summary"] = stream_summary
        elif body is not None:
            record["body"] = _redact_payload(body)
        self._append_record("upstream_resp", record)

    def record_outbound(
        self,
        status: int,
        headers: dict[str, str] | None = None,
        body_summary: str | None = None,
        outcome: TerminalOutcome | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": time.time(),
            "request_id": self._request_id,
            "status": status,
            "headers": _redact_headers(headers or {}),
            "body_summary": body_summary,
            "duration_ms": round((time.time() - self._start_time) * 1000, 1),
        }
        if outcome is not None:
            record["transport_termination"] = outcome.transport.value
            record["response_status"] = outcome.response_status.value
            record["finish_reason"] = outcome.finish_reason
            record["incomplete_details"] = outcome.incomplete_details
            if outcome.error is not None:
                record["error"] = {
                    "code": outcome.error.code,
                    "message": outcome.error.message,
                }
        self._records["outbound"] = record

    def flush(self) -> None:
        """Write all collected records to disk."""
        if not self._records:
            return

        try:
            records = _redact_payload(deepcopy(self._records))
            _write_records(self._dir, records)
            logger.debug("Audit trace written: %s (%d files)", self._dir, len(self._records))
        except OSError as e:
            logger.warning("Failed to write audit trace: %s", e)

    async def flush_async(self) -> None:
        """Write a redacted snapshot without blocking the event loop."""
        if not self._records:
            return
        records = _redact_payload(deepcopy(self._records))
        try:
            await asyncio.to_thread(_write_records, self._dir, records)
            logger.debug("Audit trace written: %s (%d files)", self._dir, len(records))
        except OSError as error:
            logger.warning("Failed to write audit trace: %s", error)
