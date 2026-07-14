"""Tests for per-request audit tracing."""

import asyncio
import json

from router_maestro.providers.base import unexpected_eof_outcome
from router_maestro.utils.audit import AuditTrace, is_tracing_enabled


class TestAuditTrace:
    """Tests for AuditTrace."""

    def test_full_lifecycle(self, tmp_path):
        trace = AuditTrace("req-123", tmp_path)
        trace.record_inbound(
            "POST", "/v1/messages", {"Authorization": "Bearer sk-xxx"}, {"model": "opus"}
        )
        trace.record_upstream(
            "POST",
            "https://api.example.com/v1/messages",
            {"Authorization": "Bearer tok"},
            {"model": "opus"},
        )
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
            "POST",
            "/v1/messages",
            {"Authorization": "Bearer secret", "x-api-key": "my-key", "X-Request-Id": "abc"},
            {},
        )
        trace.flush()

        inbound = json.loads((tmp_path / "req-456" / "inbound.json").read_text())
        assert inbound["headers"]["Authorization"] == "***"
        assert inbound["headers"]["x-api-key"] == "***"
        assert inbound["headers"]["X-Request-Id"] == "abc"

    def test_common_nested_credential_field_spellings_are_redacted(self, tmp_path):
        trace = AuditTrace("req-payload-secrets", tmp_path)
        secrets = {
            "accessToken": "access-camel",
            "access_token": "access-snake",
            "id_token": "id-secret",
            "client_secret": "client-snake",
            "client-secret": "client-kebab",
            "private_key": "private-secret",
            "nested": {"refreshToken": "refresh-camel"},
            "ordinary_text": "the word token is safe in normal content",
        }
        trace.record_inbound("POST", "/v1/messages", {}, secrets)
        trace.flush()

        body = json.loads((tmp_path / "req-payload-secrets" / "inbound.json").read_text())["body"]
        assert body["accessToken"] == "***"
        assert body["access_token"] == "***"
        assert body["id_token"] == "***"
        assert body["client_secret"] == "***"
        assert body["client-secret"] == "***"
        assert body["private_key"] == "***"
        assert body["nested"]["refreshToken"] == "***"
        assert body["ordinary_text"] == "the word token is safe in normal content"

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

    def test_upstream_attempts_are_append_only(self, tmp_path):
        trace = AuditTrace("req-retry", tmp_path)
        trace.record_upstream("POST", "https://one.invalid", {}, {"attempt": 1})
        trace.record_upstream("POST", "https://two.invalid", {}, {"attempt": 2})
        trace.record_upstream_response(503, {}, {"attempt": 1})
        trace.record_upstream_response(200, {}, {"attempt": 2})
        trace.flush()

        trace_dir = tmp_path / "req-retry"
        assert json.loads((trace_dir / "upstream.json").read_text())["url"] == (
            "https://one.invalid"
        )
        assert json.loads((trace_dir / "upstream_2.json").read_text())["url"] == (
            "https://two.invalid"
        )
        assert json.loads((trace_dir / "upstream_resp.json").read_text())["status"] == 503
        assert json.loads((trace_dir / "upstream_resp_2.json").read_text())["status"] == 200

    def test_outbound_separates_wire_status_from_terminal_outcome(self, tmp_path):
        trace = AuditTrace("req-eof", tmp_path)
        trace.record_outbound(
            200,
            body_summary="upstream stream ended unexpectedly",
            outcome=unexpected_eof_outcome(),
        )
        trace.flush()

        outbound = json.loads((tmp_path / "req-eof" / "outbound.json").read_text())
        assert outbound["status"] == 200
        assert outbound["transport_termination"] == "unexpected_eof"
        assert outbound["response_status"] == "unknown"
        assert outbound["error"]["code"] == "unexpected_eof"


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


def test_async_flush_redacts_before_scheduling_thread(monkeypatch, tmp_path):
    trace = AuditTrace("req-thread", tmp_path)
    trace.record_upstream(
        "POST",
        "https://example.invalid",
        {"Authorization": "Bearer secret"},
        {"api_key": "payload-secret", "nested": {"token": "nested-secret"}},
    )
    captured = []

    async def fake_to_thread(function, *args, **kwargs):
        captured.append((function, args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    asyncio.run(trace.flush_async())

    function, args, kwargs = captured[0]
    serialized = json.dumps(args, default=str) + json.dumps(kwargs, default=str)
    assert function.__name__ == "_write_records"
    assert "Bearer secret" not in serialized
    assert "payload-secret" not in serialized
    assert "nested-secret" not in serialized
