"""Regression tests for streaming-error body logging.

Background
----------
When `client.stream()` raises an `HTTPStatusError` via `raise_for_status()`,
the response body has *not* been read yet — `response.text` raises
`httpx.ResponseNotRead`. Earlier versions of `_raise_http_status_error`
caught that as a generic `Exception` and silently set the body to "",
producing useless logs like `Copilot stream API error: 400 -`.

This module pins down two guarantees:

1. A streamed response passed in without first calling `aread()` still yields
   a safe, useful typed error.
2. After the caller pre-reads the body via `await response.aread()`, the raw
   payload still never enters logs or the client-safe `ProviderError` message.
"""

import httpx
import pytest

from router_maestro.providers.base import BaseProvider, ProviderError
from router_maestro.utils import get_logger


def _streamed_status_error(body: bytes, status: int = 400) -> httpx.HTTPStatusError:
    """Build a real `HTTPStatusError` whose response body is not yet read."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=status, content=body)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with client.stream("POST", "https://example.invalid/chat") as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Important: do NOT read the body here — that's the whole
                # point of the regression. We want the exception to carry a
                # response that hasn't been consumed yet.
                return exc
    raise AssertionError("raise_for_status should have raised")


def test_helper_keeps_unread_stream_body_out_of_safe_error(monkeypatch):
    """Without a pre-read, implementation diagnostics must not enter the error."""
    err = _streamed_status_error(b'{"error":{"message":"missing image media_type"}}')

    # MockTransport buffers the body, so we have to simulate the
    # "streamed but not read yet" state by forcing `.text` to raise
    # `ResponseNotRead` the way real httpx does on an unread stream.
    def _raise_not_read(self):
        raise httpx.ResponseNotRead()

    monkeypatch.setattr(httpx.Response, "text", property(_raise_not_read))

    with pytest.raises(ProviderError) as excinfo:
        BaseProvider._raise_http_status_error(
            "Copilot",
            err,
            get_logger("test"),
            stream=True,
            include_body=True,
        )

    msg = str(excinfo.value)
    assert "400" in msg
    assert "response body not read" not in msg


def test_helper_keeps_read_body_out_of_logs_and_safe_error(caplog):
    """A read upstream body remains private in both logs and safe errors."""
    payload = b'{"error":{"message":"image media_type unsupported"}}'
    err = _streamed_status_error(payload)

    # Simulate what the streaming handler now does before invoking the
    # helper: pull the body in synchronously via .read() (the async path
    # uses `await response.aread()`).
    err.response.read()

    with pytest.raises(ProviderError) as excinfo:
        BaseProvider._raise_http_status_error(
            "Copilot",
            err,
            get_logger("test"),
            stream=True,
            include_body=True,
        )

    msg = str(excinfo.value)
    assert "image media_type unsupported" not in msg
    assert "image media_type unsupported" not in caplog.text
    assert excinfo.value.status_code == 400
