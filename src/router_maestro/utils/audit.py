"""Per-request audit tracing.

When enabled, captures the full request/response cycle for each API call
as JSON files for debugging. Each request gets a directory under the trace
dir containing up to 4 files:

    {trace_dir}/{request_id}/
        inbound.json        — client request (method, path, headers, body)
        upstream.json       — request sent to provider
        upstream_resp.json  — provider response (status, headers, body summary)
        outbound.json       — response sent to client

Activation:
    - Set ROUTER_MAESTRO_TRACE=1 env var, OR
    - Set audit.enabled=true in priorities.json

Trace directory defaults to ~/.local/share/router-maestro/traces/
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from router_maestro.config.paths import get_data_dir
from router_maestro.utils import get_logger

logger = get_logger("utils.audit")

_SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key", "x-goog-api-key"})


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


class AuditTrace:
    """Collects per-request trace data and writes to disk on flush."""

    def __init__(self, request_id: str, trace_dir: Path):
        self._request_id = request_id
        self._dir = trace_dir / request_id
        self._records: dict[str, Any] = {}
        self._start_time = time.time()

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
            "body": _safe_json(body),
        }

    def record_upstream(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any,
    ) -> None:
        self._records["upstream"] = {
            "timestamp": time.time(),
            "request_id": self._request_id,
            "method": method,
            "url": url,
            "headers": _redact_headers(headers),
            "body": _safe_json(body),
        }

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
            "headers": dict(headers),
        }
        if stream_summary:
            record["stream_summary"] = stream_summary
        elif body is not None:
            record["body"] = _safe_json(body)
        self._records["upstream_resp"] = record

    def record_outbound(
        self,
        status: int,
        headers: dict[str, str] | None = None,
        body_summary: str | None = None,
    ) -> None:
        self._records["outbound"] = {
            "timestamp": time.time(),
            "request_id": self._request_id,
            "status": status,
            "headers": headers or {},
            "body_summary": body_summary,
            "duration_ms": round((time.time() - self._start_time) * 1000, 1),
        }

    def flush(self) -> None:
        """Write all collected records to disk."""
        if not self._records:
            return

        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            for name, data in self._records.items():
                path = self._dir / f"{name}.json"
                path.write_text(json.dumps(data, indent=2, default=str))
            logger.debug("Audit trace written: %s (%d files)", self._dir, len(self._records))
        except OSError as e:
            logger.warning("Failed to write audit trace: %s", e)
