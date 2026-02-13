"""Tests for timeout handling."""

import logging

import httpx
import pytest

from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    TIMEOUT_STREAMING,
    BaseProvider,
    ProviderError,
)


class TestTimeoutConstants:
    """Tests for timeout constant values."""

    def test_non_streaming_timeout_values(self):
        assert TIMEOUT_NON_STREAMING.connect == 30.0
        assert TIMEOUT_NON_STREAMING.read == 120.0
        assert TIMEOUT_NON_STREAMING.write == 30.0
        assert TIMEOUT_NON_STREAMING.pool == 30.0

    def test_streaming_timeout_values(self):
        assert TIMEOUT_STREAMING.connect == 30.0
        assert TIMEOUT_STREAMING.read == 600.0
        assert TIMEOUT_STREAMING.write == 30.0
        assert TIMEOUT_STREAMING.pool == 30.0

    def test_constants_are_httpx_timeout_instances(self):
        assert isinstance(TIMEOUT_NON_STREAMING, httpx.Timeout)
        assert isinstance(TIMEOUT_STREAMING, httpx.Timeout)


class TestRaiseTimeoutError:
    """Tests for BaseProvider._raise_timeout_error."""

    def test_raises_provider_error_with_504(self):
        error = httpx.ReadTimeout("read timed out")
        with pytest.raises(ProviderError) as exc_info:
            BaseProvider._raise_timeout_error("TestProvider", error, logging.getLogger("test"))
        assert exc_info.value.status_code == 504
        assert exc_info.value.retryable is True

    def test_message_includes_timeout_type_read(self):
        error = httpx.ReadTimeout("read timed out")
        with pytest.raises(ProviderError, match=r"ReadTimeout"):
            BaseProvider._raise_timeout_error("TestProvider", error, logging.getLogger("test"))

    def test_message_includes_timeout_type_connect(self):
        error = httpx.ConnectTimeout("connect timed out")
        with pytest.raises(ProviderError, match=r"ConnectTimeout"):
            BaseProvider._raise_timeout_error("TestProvider", error, logging.getLogger("test"))

    def test_message_includes_timeout_type_write(self):
        error = httpx.WriteTimeout("write timed out")
        with pytest.raises(ProviderError, match=r"WriteTimeout"):
            BaseProvider._raise_timeout_error("TestProvider", error, logging.getLogger("test"))

    def test_message_includes_timeout_type_pool(self):
        error = httpx.PoolTimeout("pool timed out")
        with pytest.raises(ProviderError, match=r"PoolTimeout"):
            BaseProvider._raise_timeout_error("TestProvider", error, logging.getLogger("test"))

    def test_message_includes_label(self):
        error = httpx.ReadTimeout("read timed out")
        with pytest.raises(ProviderError, match=r"TestProvider timed out"):
            BaseProvider._raise_timeout_error("TestProvider", error, logging.getLogger("test"))

    def test_stream_flag_in_log(self, caplog):
        error = httpx.ReadTimeout("read timed out")
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ProviderError):
                BaseProvider._raise_timeout_error(
                    "TestProvider", error, logging.getLogger("test"), stream=True
                )
        assert "stream" in caplog.text

    def test_non_stream_no_stream_in_log(self, caplog):
        error = httpx.ReadTimeout("read timed out")
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ProviderError):
                BaseProvider._raise_timeout_error(
                    "TestProvider", error, logging.getLogger("test"), stream=False
                )
        assert " stream " not in caplog.text
