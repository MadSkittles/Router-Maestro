"""Tests for per-request audit tracing."""

import json
import tempfile
from pathlib import Path

from router_maestro.utils.audit import AuditTrace, is_tracing_enabled


class TestAuditTrace:
    """Tests for AuditTrace."""

    def test_full_lifecycle(self, tmp_path):
        trace = AuditTrace("req-123", tmp_path)
        trace.record_inbound("POST", "/v1/messages", {"Authorization": "Bearer sk-xxx"}, {"model": "opus"})
        trace.record_upstream("POST", "https://api.example.com/v1/messages", {"Authorization": "Bearer tok"}, {"model": "opus"})
        trace.record_upstream_response(200, {"content-type": "application/json"}, {"id": "msg_1"})
        trace.record_outbound(200, body_summary="streamed 1500 bytes")
        trace.flush()

        trace_dir = tmp_path / "req-123"
        assert trace_dir.exists()
        assert (trace_dir / "inbound.json").exists()
        assert (trace_dir / "upstream.json").exists()
        assert (trace_dir / "upstream_resp.json").exists()
        assert (trace_dir / "outbound.json").exists()

        inbound = json.loads((trace_dir / "inbound.json").read_text())
        assert inbound["method"] == "POST"
        assert inbound["headers"]["Authorization"] == "***"
        assert inbound["body"]["model"] == "opus"

    def test_sensitive_headers_redacted(self, tmp_path):
        trace = AuditTrace("req-456", tmp_path)
        trace.record_inbound(
            "POST", "/v1/messages",
            {"Authorization": "Bearer secret", "x-api-key": "my-key", "X-Request-Id": "abc"},
            {},
        )
        trace.flush()

        inbound = json.loads((tmp_path / "req-456" / "inbound.json").read_text())
        assert inbound["headers"]["Authorization"] == "***"
        assert inbound["headers"]["x-api-key"] == "***"
        assert inbound["headers"]["X-Request-Id"] == "abc"

    def test_no_records_no_write(self, tmp_path):
        trace = AuditTrace("req-empty", tmp_path)
        trace.flush()
        assert not (tmp_path / "req-empty").exists()

    def test_stream_summary(self, tmp_path):
        trace = AuditTrace("req-stream", tmp_path)
        trace.record_upstream_response(200, {}, stream_summary="42 chunks, 5000 bytes")
        trace.flush()

        resp = json.loads((tmp_path / "req-stream" / "upstream_resp.json").read_text())
        assert resp["stream_summary"] == "42 chunks, 5000 bytes"
        assert "body" not in resp


class TestTracingEnabled:
    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("ROUTER_MAESTRO_TRACE", "1")
        assert is_tracing_enabled(audit_config_enabled=False) is True

    def test_config(self, monkeypatch):
        monkeypatch.delenv("ROUTER_MAESTRO_TRACE", raising=False)
        assert is_tracing_enabled(audit_config_enabled=True) is True

    def test_disabled(self, monkeypatch):
        monkeypatch.delenv("ROUTER_MAESTRO_TRACE", raising=False)
        assert is_tracing_enabled(audit_config_enabled=False) is False
