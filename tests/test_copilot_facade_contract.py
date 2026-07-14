"""Contract tests for the stable Copilot provider facade."""

import importlib
import inspect
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

import httpx
import pytest

import router_maestro.providers as providers
from router_maestro.providers.copilot import CopilotProvider


def test_copilot_provider_import_identity_and_module_layout() -> None:
    copilot_module = importlib.import_module("router_maestro.providers.copilot")

    assert isinstance(copilot_module, ModuleType)
    assert providers.CopilotProvider is copilot_module.CopilotProvider
    assert CopilotProvider is copilot_module.CopilotProvider

    module_path = Path(inspect.getfile(copilot_module)).resolve()
    assert module_path.name == "copilot.py"
    assert not (module_path.parent / "copilot" / "__init__.py").exists()


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("auth_session", "CopilotAuthSession"),
        ("transport", "CopilotTransport"),
        ("catalog", "CopilotCatalog"),
        ("chat_codec", "CopilotChatCodec"),
        ("responses_codec", "CopilotResponsesCodec"),
    ],
)
def test_copilot_support_imports_are_provider_independent(
    module_name: str,
    class_name: str,
) -> None:
    support_package = importlib.import_module("router_maestro.providers.copilot_support")
    support_module = importlib.import_module(
        f"router_maestro.providers.copilot_support.{module_name}"
    )
    collaborator = getattr(support_module, class_name)

    assert getattr(support_package, class_name) is collaborator

    source = inspect.getsource(support_module)
    assert "router_maestro.server" not in source
    assert "CopilotProvider" not in source


def test_copilot_facade_compatibility_seams_round_trip_without_network() -> None:
    provider = CopilotProvider()
    client = object()
    models_cache = object()

    provider._client = client  # type: ignore[assignment]
    provider._cached_token = "copilot-token"
    provider._token_expires = 1_234_567_890
    provider._api_base = "https://copilot.example.test/"
    provider._models_ttl_cache = models_cache

    assert provider._client is client
    assert provider._cached_token == "copilot-token"
    assert provider._token_expires == 1_234_567_890
    assert provider._api_base == "https://copilot.example.test/"
    assert provider._models_ttl_cache is models_cache
    assert callable(provider._send_with_auth_retry)
    assert callable(provider._stream_with_auth_retry)


@pytest.mark.asyncio
async def test_native_anthropic_token_count_public_contract_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = CopilotProvider.count_native_anthropic_tokens
    signature = inspect.signature(method)

    assert inspect.iscoroutinefunction(method)
    assert list(signature.parameters) == ["self", "payload", "model"]
    assert signature.parameters["payload"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["payload"].default is inspect.Parameter.empty
    assert signature.parameters["model"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["model"].default is inspect.Parameter.empty
    assert signature.return_annotation is int

    provider = CopilotProvider()
    ensure_token = AsyncMock()
    send_with_auth_retry = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"input_tokens": 7},
            request=httpx.Request(
                "POST",
                "https://api.githubcopilot.com/v1/messages/count_tokens",
            ),
        )
    )
    monkeypatch.setattr(provider, "ensure_token", ensure_token)
    monkeypatch.setattr(provider, "_send_with_auth_retry", send_with_auth_retry)
    payload = {
        "model": "claude-sonnet-4.5",
        "messages": [{"role": "user", "content": "Count me"}],
    }

    result = await provider.count_native_anthropic_tokens(
        payload,
        model="claude-sonnet-4.5",
    )

    assert result == 7
    ensure_token.assert_awaited_once_with()
    send_with_auth_retry.assert_awaited_once_with(
        "POST",
        "/v1/messages/count_tokens",
        json=payload,
        model="claude-sonnet-4.5",
    )
