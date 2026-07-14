"""Credential and discovery contracts for custom providers."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from router_maestro.auth.discovery import (
    ProviderAuthDefinition,
    ProviderAuthSource,
    provider_auth_definitions,
)
from router_maestro.auth.repository import CredentialRepository
from router_maestro.auth.storage import ApiKeyCredential, AuthType, OAuthCredential
from router_maestro.cli.auth import discover_auth_providers
from router_maestro.cli.client import ServerNotRunningError
from router_maestro.config.providers import (
    CustomProviderConfig,
    CustomProviderOptions,
    ProvidersConfig,
)
from router_maestro.providers.custom_factory import (
    CustomCredentialSource,
    create_custom_provider,
    default_custom_api_key_env,
    resolve_custom_provider_credential,
)
from router_maestro.providers.openai_compat import OpenAICompatibleProvider


def _config(**options) -> CustomProviderConfig:
    return CustomProviderConfig(
        baseURL="http://localhost:11434/v1",
        options=CustomProviderOptions(**options),
    )


def test_custom_provider_options_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="unused"):
        CustomProviderOptions.model_validate({"unused": True})


@pytest.mark.parametrize("value", ["", "9INVALID", "HAS-DASH", "HAS SPACE"])
def test_custom_provider_options_validate_explicit_environment_name(value: str) -> None:
    with pytest.raises(ValidationError):
        CustomProviderOptions(api_key_env=value)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("ollama", "OLLAMA_API_KEY"),
        ("my-provider", "MY_PROVIDER_API_KEY"),
        ("my.provider-v2", "MY_PROVIDER_V2_API_KEY"),
    ],
)
def test_default_environment_name_normalizes_provider(provider: str, expected: str) -> None:
    assert default_custom_api_key_env(provider) == expected


def test_environment_credential_overrides_repository(tmp_path) -> None:
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("ollama", ApiKeyCredential(key="repository-key"))

    resolved = resolve_custom_provider_credential(
        "ollama",
        _config(),
        credential_repository=repository,
        environ={"OLLAMA_API_KEY": "environment-key"},
    )

    assert resolved is not None
    assert resolved.source is CustomCredentialSource.ENVIRONMENT
    assert resolved.api_key == "environment-key"


def test_explicit_environment_name_replaces_derived_name(tmp_path) -> None:
    repository = CredentialRepository(tmp_path / "auth.json")

    resolved = resolve_custom_provider_credential(
        "ollama",
        _config(api_key_env="LOCAL_LLM_TOKEN"),
        credential_repository=repository,
        environ={
            "OLLAMA_API_KEY": "derived-key",
            "LOCAL_LLM_TOKEN": "explicit-key",
        },
    )

    assert resolved is not None
    assert resolved.api_key == "explicit-key"


def test_empty_environment_value_falls_back_to_repository(tmp_path) -> None:
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("ollama", ApiKeyCredential(key="repository-key"))

    resolved = resolve_custom_provider_credential(
        "ollama",
        _config(),
        credential_repository=repository,
        environ={"OLLAMA_API_KEY": ""},
    )

    assert resolved is not None
    assert resolved.source is CustomCredentialSource.REPOSITORY
    assert resolved.api_key == "repository-key"


def test_non_api_repository_credential_is_not_used_as_custom_key(tmp_path) -> None:
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider(
        "ollama",
        OAuthCredential(refresh="refresh", access="access"),
    )

    assert (
        resolve_custom_provider_credential(
            "ollama",
            _config(),
            credential_repository=repository,
            environ={},
        )
        is None
    )


def test_required_custom_provider_without_key_is_skipped(tmp_path) -> None:
    provider = create_custom_provider(
        "ollama",
        _config(),
        credential_repository=CredentialRepository(tmp_path / "auth.json"),
        environ={},
    )

    assert provider is None


def test_explicit_anonymous_provider_is_authenticated_without_authorization(tmp_path) -> None:
    provider = create_custom_provider(
        "ollama",
        _config(allow_unauthenticated=True),
        credential_repository=CredentialRepository(tmp_path / "auth.json"),
        environ={},
    )

    assert provider is not None
    assert provider.is_authenticated() is True
    assert provider._get_headers() == {"Content-Type": "application/json"}


def test_none_key_without_explicit_anonymous_permission_is_not_authenticated() -> None:
    provider = OpenAICompatibleProvider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key=None,
    )

    assert provider.is_authenticated() is False
    assert "Authorization" not in provider._get_headers()


def test_provider_auth_definitions_are_builtin_first_then_sorted_custom() -> None:
    definitions = provider_auth_definitions(
        ProvidersConfig(
            providers={
                "zeta-provider": _config(allow_unauthenticated=True),
                "alpha-provider": _config(api_key_env="ALPHA_TOKEN"),
            }
        )
    )

    assert [definition.provider for definition in definitions] == [
        "github-copilot",
        "openai",
        "anthropic",
        "alpha-provider",
        "zeta-provider",
    ]
    assert definitions[0] == ProviderAuthDefinition(
        provider="github-copilot",
        display_name="GitHub Copilot",
        auth_type=AuthType.OAUTH,
        credential_required=True,
        source=ProviderAuthSource.BUILTIN,
    )
    assert definitions[3].api_key_env == "ALPHA_TOKEN"
    assert definitions[3].credential_required is True
    assert definitions[4].api_key_env == "ZETA_PROVIDER_API_KEY"
    assert definitions[4].credential_required is False


def test_builtin_auth_discovery_does_not_advertise_environment_credentials() -> None:
    definitions = provider_auth_definitions(ProvidersConfig())

    assert all(definition.api_key_env is None for definition in definitions)


class _DiscoveryClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    async def list_auth_providers(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _counted_local_loader(
    calls: list[None],
    config: ProvidersConfig,
) -> Callable[[], ProvidersConfig]:
    def load() -> ProvidersConfig:
        calls.append(None)
        return config

    return load


@pytest.mark.asyncio
async def test_cli_discovery_prefers_server_definitions_over_local_config() -> None:
    server_definition = {
        "provider": "remote-custom",
        "display_name": "Remote Custom",
        "auth_type": "api",
        "credential_required": True,
        "source": "custom",
        "api_key_env": "REMOTE_CUSTOM_API_KEY",
    }
    client = _DiscoveryClient([server_definition])
    local_calls: list[None] = []

    definitions = await discover_auth_providers(
        client,
        context_name="local",
        load_local_config=_counted_local_loader(local_calls, ProvidersConfig()),
    )

    assert definitions == (ProviderAuthDefinition.from_mapping(server_definition),)
    assert client.calls == 1
    assert local_calls == []


@pytest.mark.asyncio
async def test_cli_local_connection_failure_uses_typed_local_config() -> None:
    client = _DiscoveryClient(error=ServerNotRunningError("http://localhost:8080"))
    local_calls: list[None] = []
    local_config = ProvidersConfig(providers={"ollama": _config()})

    definitions = await discover_auth_providers(
        client,
        context_name="local",
        load_local_config=_counted_local_loader(local_calls, local_config),
    )

    assert definitions == provider_auth_definitions(local_config)
    assert client.calls == 1
    assert local_calls == [None]


@pytest.mark.asyncio
async def test_cli_remote_connection_failure_never_reads_local_config() -> None:
    error = ServerNotRunningError("https://remote.example")
    client = _DiscoveryClient(error=error)
    local_calls: list[None] = []

    with pytest.raises(ServerNotRunningError) as exc_info:
        await discover_auth_providers(
            client,
            context_name="remote",
            load_local_config=_counted_local_loader(local_calls, ProvidersConfig()),
        )

    assert exc_info.value is error
    assert client.calls == 1
    assert local_calls == []
