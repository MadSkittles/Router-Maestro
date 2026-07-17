"""Cross-provider request-option fidelity contracts."""

from __future__ import annotations

from logging import getLogger
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from router_maestro.providers.anthropic import AnthropicProvider
from router_maestro.providers.base import (
    ChatRequest,
    Message,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesRequest,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    Feature,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan
from router_maestro.routing.router import Router
from router_maestro.server.schemas.gemini import (
    GeminiContent,
    GeminiGenerateContentRequest,
    GeminiGenerationConfig,
    GeminiPart,
)
from router_maestro.server.schemas.openai import ChatCompletionRequest, ChatMessage
from router_maestro.server.schemas.responses import ResponsesRequest as WireResponsesRequest
from router_maestro.server.translation_gemini import translate_gemini_to_openai
from router_maestro.utils.reasoning import pick_closest_effort


class _OpenAIProvider(OpenAIChatProvider):
    name = "openai-test"

    def __init__(self) -> None:
        super().__init__("https://example.invalid/v1", getLogger(__name__))

    def _get_headers(self) -> dict[str, str]:
        return {}

    async def list_models(self):
        return []

    def is_authenticated(self) -> bool:
        return True


def _request(**options) -> ChatRequest:
    model = options.pop("model", "model")
    return ChatRequest(
        model=model,
        messages=[Message(role="user", content="hello")],
        **options,
    )


def test_omitted_temperature_remains_absent_across_request_boundaries() -> None:
    chat_wire = ChatCompletionRequest(
        model="model",
        messages=[ChatMessage(role="user", content="hello")],
    )
    responses_wire = WireResponsesRequest(model="model", input="hello")
    chat_internal = _request()
    responses_internal = ResponsesRequest(model="model", input="hello")
    gemini_internal = translate_gemini_to_openai(
        GeminiGenerateContentRequest(
            contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])]
        ),
        "model",
    )

    assert chat_wire.temperature is None
    assert responses_wire.temperature is None
    assert chat_internal.temperature is None
    assert responses_internal.temperature is None
    assert gemini_internal.temperature is None


def test_explicit_temperature_is_preserved_across_request_boundaries() -> None:
    assert (
        ChatCompletionRequest(
            model="model",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=1.0,
        ).temperature
        == 1.0
    )
    assert WireResponsesRequest(model="model", input="hello", temperature=1.0).temperature == 1.0
    translated = translate_gemini_to_openai(
        GeminiGenerateContentRequest(
            contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
            generation_config=GeminiGenerationConfig(temperature=1.0),
        ),
        "model",
    )
    assert translated.temperature == 1.0


def test_gemini_generation_penalties_are_preserved_across_request_boundary() -> None:
    translated = translate_gemini_to_openai(
        GeminiGenerateContentRequest.model_validate(
            {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "generationConfig": {
                    "frequencyPenalty": 0.3,
                    "presencePenalty": -0.1,
                },
            }
        ),
        "model",
    )

    assert translated.frequency_penalty == 0.3
    assert translated.presence_penalty == -0.1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ChatCompletionRequest(
            model="model",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=None,
        ),
        lambda: WireResponsesRequest(model="model", input="hello", temperature=None),
    ],
)
def test_openai_wire_requests_reject_explicit_null_temperature(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_chat_payloads_omit_unspecified_temperature_and_forward_explicit_default() -> None:
    openai = _OpenAIProvider()
    anthropic = AnthropicProvider()
    copilot = CopilotProvider()

    assert "temperature" not in openai._build_payload(_request(), stream=False)
    assert "temperature" not in anthropic._build_payload(_request())
    assert "temperature" not in copilot._build_chat_payload(_request(), stream=False)

    assert openai._build_payload(_request(temperature=1.0), stream=False)["temperature"] == 1.0
    assert anthropic._build_payload(_request(temperature=1.0))["temperature"] == 1.0
    assert (
        copilot._build_chat_payload(_request(temperature=1.0), stream=False)["temperature"] == 1.0
    )


def test_openai_chat_forwards_typed_native_options() -> None:
    provider = _OpenAIProvider()
    request = _request(
        temperature=0.2,
        top_p=0.8,
        frequency_penalty=0.3,
        presence_penalty=-0.1,
        stop=["END"],
        user="user-123",
        metadata={"trace": "abc"},
        service_tier="flex",
        reasoning_effort="high",
    )

    payload = provider._build_payload(request, stream=False)

    assert payload == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "stream": False,
        "top_p": 0.8,
        "frequency_penalty": 0.3,
        "presence_penalty": -0.1,
        "stop": ["END"],
        "user": "user-123",
        "metadata": {"trace": "abc"},
        "service_tier": "flex",
        "reasoning_effort": "high",
    }


def test_openai_and_copilot_chat_preserve_assistant_refusal_history() -> None:
    request = ChatRequest(
        model="model",
        messages=[Message(role="assistant", content=None, refusal="I cannot help")],
    )

    openai_payload = _OpenAIProvider()._build_payload(request, stream=False)
    copilot_messages, _has_images = CopilotProvider()._build_messages_payload(request)

    expected = {"role": "assistant", "content": None, "refusal": "I cannot help"}
    assert openai_payload["messages"] == [expected]
    assert copilot_messages == [expected]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking_budget", [1, 1023])
def test_openai_chat_rejects_positive_budget_below_lowest_reasoning_tier(
    stream,
    thinking_budget,
) -> None:
    provider = _OpenAIProvider()

    with pytest.raises(RequestOptionError) as caught:
        provider.validate_chat_request(
            _request(thinking_budget=thinking_budget),
            stream=stream,
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.status_code == 400
    assert caught.value.retryable is False
    assert caught.value.parameter == "thinking_budget"


@pytest.mark.parametrize("stream", [False, True])
def test_openai_chat_maps_lowest_reasoning_budget_exactly(stream) -> None:
    provider = _OpenAIProvider()

    payload = provider._build_payload(
        _request(thinking_budget=1024),
        stream=stream,
    )

    assert payload["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    ("options", "parameter"),
    [
        ({"top_k": 20}, "top_k"),
        ({"candidate_count": 2}, "candidate_count"),
        ({"response_mime_type": "application/json"}, "response_mime_type"),
    ],
)
def test_openai_chat_rejects_options_it_cannot_represent(options, parameter) -> None:
    provider = _OpenAIProvider()

    with pytest.raises(ProviderError) as caught:
        provider._build_payload(_request(**options), stream=False)

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.status_code == 400
    assert caught.value.parameter == parameter


def test_stop_sequences_translate_to_openai_stop() -> None:
    provider = _OpenAIProvider()

    payload = provider._build_payload(
        _request(stop_sequences=["END", "STOP"]),
        stream=False,
    )

    assert payload["stop"] == ["END", "STOP"]


def test_empty_gemini_stop_sequences_reach_openai_payload() -> None:
    translated = translate_gemini_to_openai(
        GeminiGenerateContentRequest.model_validate(
            {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "generationConfig": {"stopSequences": []},
            }
        ),
        "model",
    )

    payload = _OpenAIProvider()._build_payload(translated, stream=False)

    assert "stop" in payload
    assert payload["stop"] == []


def test_conflicting_stop_shapes_are_rejected() -> None:
    provider = _OpenAIProvider()

    with pytest.raises(ProviderError) as caught:
        provider._build_payload(
            _request(stop="END", stop_sequences=["STOP"]),
            stream=False,
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "stop"


def test_anthropic_chat_forwards_native_and_translated_options() -> None:
    provider = AnthropicProvider()
    request = _request(
        temperature=0.4,
        top_p=0.7,
        top_k=32,
        stop=["END"],
        user="user-123",
        metadata={"trace_id": "abc"},
        service_tier="standard_only",
        reasoning_effort="medium",
    )

    payload = provider._build_payload(request)

    assert payload["temperature"] == 0.4
    assert payload["top_p"] == 0.7
    assert payload["top_k"] == 32
    assert payload["stop_sequences"] == ["END"]
    assert payload["metadata"] == {"trace_id": "abc", "user_id": "user-123"}
    assert payload["service_tier"] == "standard_only"
    assert payload["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize(
    ("options", "parameter"),
    [
        ({"frequency_penalty": 0.1}, "frequency_penalty"),
        ({"presence_penalty": 0.1}, "presence_penalty"),
        ({"candidate_count": 2}, "candidate_count"),
        ({"response_mime_type": "application/json"}, "response_mime_type"),
    ],
)
def test_anthropic_chat_rejects_options_it_cannot_represent(options, parameter) -> None:
    provider = AnthropicProvider()

    with pytest.raises(ProviderError) as caught:
        provider._build_payload(_request(**options))

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.status_code == 400
    assert caught.value.parameter == parameter


@pytest.mark.parametrize(
    "extensions",
    [
        {"vendor_flag": "on"},
        {"model": "other"},
        {"topP": 0.9},
        {"reasoning-effort": "high"},
    ],
)
def test_openai_compatible_rejects_unapproved_provider_extensions(extensions) -> None:
    provider = _OpenAIProvider()
    request = _request(
        temperature=0.25,
        top_p=0.75,
        provider_extensions=extensions,
    )

    with pytest.raises(ProviderError) as caught:
        provider._build_payload(request, stream=False)

    key = next(iter(extensions))
    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.status_code == 400
    assert caught.value.parameter == f"provider_extensions.{key}"


def test_legacy_unknown_extra_is_rejected_as_an_unapproved_extension() -> None:
    provider = _OpenAIProvider()
    request = _request(
        temperature=0.25,
        extra={"vendor_flag": "on"},
    )

    with pytest.raises(ProviderError) as caught:
        provider._build_payload(request, stream=False)

    assert request.temperature == 0.25
    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "provider_extensions.vendor_flag"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.4),
        ("max_tokens", 128),
        ("tools", [{"type": "function", "function": {"name": "x"}}]),
        ("tool_choice", "auto"),
        ("thinking_budget", 4096),
        ("thinking_type", "enabled"),
        ("reasoning_effort", "high"),
        ("top_p", 0.8),
        ("frequency_penalty", 0.2),
        ("presence_penalty", -0.1),
        ("stop", ["END"]),
        ("user", "user-123"),
        ("top_k", 32),
        ("stop_sequences", ["STOP"]),
        ("metadata", {"trace": "abc"}),
        ("service_tier", "flex"),
        ("candidate_count", 2),
        ("response_mime_type", "application/json"),
    ],
)
def test_legacy_extra_promotes_every_defaulted_typed_core_field(field, value) -> None:
    request = _request(extra={field: value})

    assert getattr(request, field) == value
    assert field not in request.provider_extensions
    assert field not in request.extra


def test_legacy_extra_promotes_optional_core_field() -> None:
    request = _request(extra={"reasoning_effort": "high", "max_tokens": 128})

    assert request.reasoning_effort == "high"
    assert request.max_tokens == 128


def test_legacy_extra_promotes_temperature_instead_of_silently_dropping_it() -> None:
    request = _request(extra={"temperature": 0.75})

    assert request.temperature == 0.75
    assert request.provider_extensions == {}


@pytest.mark.parametrize(
    ("typed", "legacy", "parameter"),
    [
        ({"temperature": 0.25}, {"temperature": 0.75}, "temperature"),
        ({"reasoning_effort": "low"}, {"reasoning_effort": "high"}, "reasoning_effort"),
        ({"model": "typed-model"}, {"model": "legacy-model"}, "model"),
    ],
)
def test_conflicting_typed_and_legacy_core_values_are_rejected(
    typed,
    legacy,
    parameter,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        _request(extra=legacy, **typed)


def test_matching_typed_and_legacy_core_values_are_deduplicated() -> None:
    request = _request(
        temperature=0.25,
        reasoning_effort="low",
        extra={"temperature": 0.25, "reasoning_effort": "low"},
    )

    assert request.temperature == 0.25
    assert request.reasoning_effort == "low"
    assert request.provider_extensions == {}


def test_non_optional_legacy_core_value_is_not_silently_dropped() -> None:
    legacy_messages = [Message(role="user", content="legacy")]

    with pytest.raises(ValueError, match="messages"):
        _request(extra={"messages": legacy_messages})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream", True),
    ],
)
def test_legacy_extra_rejects_non_null_default_core_fields(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        _request(extra={field: value})


def test_legacy_extra_rejects_invalid_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        _request(extra={"reasoning_effort": "ultra"})


def test_reasoning_tier_selection_never_substitutes_upward() -> None:
    assert pick_closest_effort("minimal", ["minimal", "low"]) == "minimal"
    assert pick_closest_effort("low", ["minimal"]) == "minimal"
    assert pick_closest_effort("minimal", ["low"]) is None
    assert pick_closest_effort("high", ["low", "medium"]) == "medium"
    assert pick_closest_effort("medium", ["low", "high"]) == "low"
    assert pick_closest_effort("low", ["medium", "high"]) is None
    assert pick_closest_effort("ultra", ["low", "high"]) is None


def test_copilot_chat_payload_forwards_openai_native_options() -> None:
    provider = CopilotProvider()
    request = _request(
        top_p=0.8,
        frequency_penalty=0.2,
        presence_penalty=-0.1,
        stop=["END"],
        user="user-123",
        service_tier="default",
    )

    payload = provider._build_chat_payload(request, stream=False)

    assert payload["top_p"] == 0.8
    assert payload["frequency_penalty"] == 0.2
    assert payload["presence_penalty"] == -0.1
    assert payload["stop"] == ["END"]
    assert payload["user"] == "user-123"
    assert payload["service_tier"] == "default"


@pytest.mark.parametrize("parameter", ["top_k", "candidate_count", "response_mime_type"])
def test_copilot_chat_rejects_non_chat_options(parameter) -> None:
    provider = CopilotProvider()

    with pytest.raises(ProviderError) as caught:
        provider._build_chat_payload(_request(**{parameter: 1}), stream=False)

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == parameter


def test_copilot_responses_payload_forwards_supported_options() -> None:
    provider = CopilotProvider()
    request = ResponsesRequest(
        model="gpt-5.4",
        input="hi",
        top_p=0.7,
        metadata={"trace": "abc"},
        service_tier="flex",
    )

    payload = provider._build_responses_payload(request)

    assert "temperature" not in payload
    assert payload["top_p"] == 0.7
    assert payload["metadata"] == {"trace": "abc"}
    assert payload["service_tier"] == "flex"


def test_copilot_responses_rejects_explicit_temperature() -> None:
    provider = CopilotProvider()

    with pytest.raises(RequestOptionError) as caught:
        provider._build_responses_payload(
            ResponsesRequest(model="gpt-5.4", input="hi", temperature=0.4)
        )

    assert caught.value.status_code == 400
    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "temperature"


def test_copilot_tool_prevalidation_replaces_incompatible_fallback_without_spending_slot() -> None:
    operation = Operation.RESPONSES
    features = RequestFeatures(tools=True)

    def candidate(provider, model: str) -> RouteCandidate:
        ref = ModelRef(provider.name, model)
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: CapabilitySupport.SUPPORTED},
            features={Feature.TOOLS: CapabilitySupport.SUPPORTED},
        )
        return RouteCandidate(
            model=ref,
            provider=provider,
            capabilities=capabilities,
            evaluated_operation=operation,
            evaluated_features=features,
            support=CapabilitySupport.SUPPORTED,
        )

    primary = MagicMock()
    primary.name = "primary"
    rejected = CopilotProvider()
    retained = MagicMock()
    retained.name = "retained"
    primary_candidate = candidate(primary, "primary-model")
    rejected_candidate = candidate(rejected, "copilot-model")
    retained_candidate = candidate(retained, "retained-model")
    plan = RoutePlan(
        operation=operation,
        features=features,
        primary=primary_candidate,
        fallbacks=(rejected_candidate,),
        explicit=False,
        fallback_pool=(rejected_candidate, retained_candidate),
        max_fallback_attempts=1,
    )
    request = ResponsesRequest(
        model="router-maestro",
        input="hi",
        tools=[{"type": "web_search"}],
    )

    validated = Router.prevalidate_plan(
        plan,
        lambda route_candidate: route_candidate.provider.validate_responses_request(
            ResponsesRequest(
                model=route_candidate.model.upstream_id,
                input=request.input,
                tools=request.tools,
            )
        ),
    )

    assert validated.fallbacks == (retained_candidate,)
    primary.validate_responses_request.assert_called_once()
    retained.validate_responses_request.assert_called_once()


@pytest.mark.parametrize(
    "extensions",
    [
        {"vendor": True},
        {"model": "other"},
        {"topP": 0.9},
        {"maxOutputTokens": 10},
    ],
)
def test_copilot_responses_rejects_unapproved_extensions(extensions) -> None:
    provider = CopilotProvider()
    request = ResponsesRequest(
        model="gpt-5.4",
        input="hi",
        temperature=0.4,
        provider_extensions=extensions,
    )

    with pytest.raises(ProviderError) as caught:
        provider._build_responses_payload(request)

    key = next(iter(extensions))
    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == f"provider_extensions.{key}"


def test_anthropic_rejects_unapproved_provider_extensions() -> None:
    provider = AnthropicProvider()

    with pytest.raises(ProviderError) as caught:
        provider._build_payload(_request(provider_extensions={"vendor": True}))

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "provider_extensions.vendor"


def test_copilot_chat_rejects_unapproved_provider_extensions() -> None:
    provider = CopilotProvider()

    with pytest.raises(ProviderError) as caught:
        provider._build_chat_payload(
            _request(provider_extensions={"vendor": True}),
            stream=False,
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "provider_extensions.vendor"


def test_copilot_chat_rejects_reasoning_when_only_higher_tiers_are_available() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["medium", "high"],
            )
        ]
    )

    with pytest.raises(ProviderError) as caught:
        provider._build_chat_payload(
            _request(model="gpt-5.4", reasoning_effort="low"),
            stream=False,
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "reasoning_effort"


@pytest.mark.parametrize(
    ("requested", "allowed", "expected"),
    [
        pytest.param("minimal", ["minimal", "low"], "minimal", id="exact-minimal"),
        pytest.param("low", ["minimal"], "minimal", id="low-down-to-minimal"),
        pytest.param("low", ["none", "low"], "low", id="ignore-none-and-match-low"),
        pytest.param("high", ["none", "low"], "low", id="ignore-none-and-step-down"),
    ],
)
def test_copilot_chat_warm_catalog_maps_minimal_exactly_or_downward(
    requested,
    allowed,
    expected,
) -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=allowed,
            )
        ]
    )

    payload = provider._build_chat_payload(
        _request(model="gpt-5.4", reasoning_effort=requested),
        stream=False,
    )

    assert payload["reasoning_effort"] == expected


@pytest.mark.parametrize("requested", ["minimal", "low"])
def test_copilot_chat_warm_catalog_none_only_rejects_public_effort(requested) -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["none"],
            )
        ]
    )

    with pytest.raises(RequestOptionError) as caught:
        provider._build_chat_payload(
            _request(model="gpt-5.4", reasoning_effort=requested),
            stream=False,
        )

    assert caught.value.parameter == "reasoning_effort"


def test_copilot_chat_warm_catalog_does_not_raise_minimal_to_low() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["low"],
            )
        ]
    )

    with pytest.raises(RequestOptionError) as caught:
        provider._build_chat_payload(
            _request(model="gpt-5.4", reasoning_effort="minimal"),
            stream=False,
        )

    assert caught.value.parameter == "reasoning_effort"


def test_copilot_chat_rejects_budget_when_catalog_says_reasoning_unsupported() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-4o",
                name="gpt-4o",
                provider="github-copilot",
                reasoning_effort_values=[],
            )
        ]
    )

    with pytest.raises(ProviderError) as caught:
        provider._build_chat_payload(
            _request(model="gpt-4o", thinking_budget=4096),
            stream=False,
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "thinking_budget"


@pytest.mark.parametrize("stream", [False, True])
def test_copilot_chat_preflight_rejects_tiny_budget_without_upward_substitution(stream) -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["low", "medium", "high"],
            )
        ]
    )

    with pytest.raises(ProviderError) as caught:
        provider.validate_chat_request(
            _request(model="gpt-5.4", thinking_budget=1),
            stream=stream,
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "thinking_budget"


def test_copilot_chat_budget_maps_exact_or_downward_only() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["low", "high"],
            )
        ]
    )

    payload = provider._build_chat_payload(
        _request(model="gpt-5.4", thinking_budget=4096),
        stream=False,
    )

    assert payload["reasoning_effort"] == "low"


def test_copilot_chat_warm_budget_without_downward_tier_reports_budget_parameter() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["medium", "high"],
            )
        ]
    )

    with pytest.raises(ProviderError) as caught:
        provider.validate_chat_request(
            _request(model="gpt-5.4", thinking_budget=1024),
            stream=True,
        )

    assert caught.value.parameter == "thinking_budget"


def test_copilot_chat_warm_explicit_effort_reports_effort_parameter() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["medium", "high"],
            )
        ]
    )

    with pytest.raises(ProviderError) as caught:
        provider.validate_chat_request(
            _request(model="gpt-5.4", thinking_budget=16384, reasoning_effort="low"),
            stream=True,
        )

    assert caught.value.parameter == "reasoning_effort"


@pytest.mark.parametrize("stream", [False, True])
def test_copilot_chat_cold_gpt5_tiny_budget_rejects_preflight(stream) -> None:
    provider = CopilotProvider()

    with pytest.raises(ProviderError) as caught:
        provider.validate_chat_request(
            _request(model="gpt-5.4", thinking_budget=1),
            stream=stream,
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.retryable is False
    assert caught.value.parameter == "thinking_budget"


def test_copilot_chat_explicit_effort_takes_precedence_over_budget() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["low", "medium", "high", "xhigh"],
            )
        ]
    )

    payload = provider._build_chat_payload(
        _request(model="gpt-5.4", thinking_budget=16384, reasoning_effort="low"),
        stream=False,
    )

    assert payload["reasoning_effort"] == "low"


def test_copilot_responses_rejects_reasoning_when_catalog_says_unsupported() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-4o",
                name="gpt-4o",
                provider="github-copilot",
                reasoning_effort_values=[],
            )
        ]
    )

    with pytest.raises(ProviderError) as caught:
        provider._build_responses_payload(
            ResponsesRequest(model="gpt-4o", input="hi", reasoning_effort="high")
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "reasoning_effort"


@pytest.mark.parametrize("stream", [False, True])
def test_copilot_responses_cold_catalog_rejects_known_unsupported_family(stream) -> None:
    provider = CopilotProvider()
    request = ResponsesRequest(
        model="gpt-4o",
        input="hi",
        stream=stream,
        reasoning_effort="high",
    )

    with pytest.raises(ProviderError) as caught:
        provider.validate_responses_request(request)

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "reasoning_effort"


def test_copilot_responses_unknown_family_keeps_explicit_reasoning_observable() -> None:
    provider = CopilotProvider()

    payload = provider._build_responses_payload(
        ResponsesRequest(model="future-reasoner", input="hi", reasoning_effort="high")
    )

    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}


@pytest.mark.parametrize(
    "model",
    ["gemini-3.2-pro", "claude-sonnet-5", "mai-code-1-flash-picker"],
)
def test_copilot_responses_new_reasoning_families_keep_low_when_catalog_is_cold(model) -> None:
    provider = CopilotProvider()

    payload = provider._build_responses_payload(
        ResponsesRequest(model=model, input="hi", reasoning_effort="low")
    )

    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}


def test_copilot_responses_cold_catalog_preserves_supported_reasoning_family() -> None:
    provider = CopilotProvider()

    payload = provider._build_responses_payload(
        ResponsesRequest(model="gpt-5.4", input="hi", reasoning_effort="xhigh")
    )

    assert payload["reasoning"] == {"effort": "xhigh", "summary": "auto"}


@pytest.mark.parametrize("model", ["claude-sonnet-4", "gpt-4o", "gemini-2.5-pro"])
def test_copilot_chat_rejects_explicit_reasoning_for_known_unsupported_models(model) -> None:
    provider = CopilotProvider()

    with pytest.raises(ProviderError) as caught:
        provider._build_chat_payload(
            _request(model=model, reasoning_effort="medium"),
            stream=False,
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "reasoning_effort"


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4",
        "claude-opus-4-20250101",
        "claude-opus-4.1",
        "claude-haiku-4",
        "claude-haiku-4.1",
        "claude-haiku-4.5",
    ],
)
@pytest.mark.parametrize("cache_state", ["cold", "missing-metadata"])
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("adapter", ["chat", "responses"])
def test_copilot_missing_catalog_reasoning_metadata_rejects_known_old_claude4(
    model,
    cache_state,
    stream,
    adapter,
) -> None:
    provider = CopilotProvider()
    if cache_state == "missing-metadata":
        provider._models_ttl_cache.set(
            [
                ModelInfo(
                    id=model,
                    name=model,
                    provider="github-copilot",
                    reasoning_effort_values=None,
                )
            ]
        )

    with pytest.raises(RequestOptionError) as caught:
        if adapter == "chat":
            provider.validate_chat_request(
                _request(model=model, stream=stream, reasoning_effort="low"),
                stream=stream,
            )
        else:
            provider.validate_responses_request(
                ResponsesRequest(
                    model=model,
                    input="hi",
                    stream=stream,
                    reasoning_effort="low",
                )
            )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "reasoning_effort"


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4.6",
        "claude-opus-4-6",
        "claude-opus-4-6-1m",
        "claude-opus-4-6[1m]",
        "claude-opus-4.7",
        "claude-opus-4-7-20260101",
        "claude-opus-4-7[1m]",
        "claude-opus-4.8",
        "claude-opus-4-8[1m]",
        "claude-sonnet-4.6",
        "claude-sonnet-4-6[1m]",
        "claude-opus-4.9",
        "claude-sonnet-4.7",
        "claude-haiku-4.9",
        "claude-haiku-4-9",
        "claude-opus-5",
        "claude-haiku-5.1",
        "claude-sonnet-5",
    ],
)
@pytest.mark.parametrize("adapter", ["chat", "responses"])
def test_copilot_missing_catalog_metadata_keeps_supported_and_future_claude_observable(
    model,
    adapter,
) -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id=model,
                name=model,
                provider="github-copilot",
                reasoning_effort_values=None,
            )
        ]
    )

    if adapter == "chat":
        payload = provider._build_chat_payload(
            _request(model=model, reasoning_effort="low"),
            stream=False,
        )
        assert payload["reasoning_effort"] == "low"
    else:
        payload = provider._build_responses_payload(
            ResponsesRequest(model=model, input="hi", reasoning_effort="low")
        )
        assert payload["reasoning"] == {"effort": "low", "summary": "auto"}


@pytest.mark.parametrize("adapter", ["chat", "responses"])
def test_copilot_live_catalog_can_enable_reasoning_on_static_old_claude(adapter) -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="claude-haiku-4.1",
                name="claude-haiku-4.1",
                provider="github-copilot",
                reasoning_effort_values=["low"],
            )
        ]
    )

    if adapter == "chat":
        payload = provider._build_chat_payload(
            _request(model="claude-haiku-4.1", reasoning_effort="low"),
            stream=False,
        )
        assert payload["reasoning_effort"] == "low"
    else:
        payload = provider._build_responses_payload(
            ResponsesRequest(
                model="claude-haiku-4.1",
                input="hi",
                reasoning_effort="low",
            )
        )
        assert payload["reasoning"] == {"effort": "low", "summary": "auto"}


def test_copilot_responses_rejects_when_only_higher_tier_is_available() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["medium", "high"],
            )
        ]
    )

    with pytest.raises(ProviderError) as caught:
        provider._build_responses_payload(
            ResponsesRequest(model="gpt-5.4", input="hi", reasoning_effort="low")
        )

    assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert caught.value.parameter == "reasoning_effort"


@pytest.mark.parametrize(
    ("requested", "allowed", "expected"),
    [
        pytest.param("minimal", ["minimal", "low"], "minimal", id="exact-minimal"),
        pytest.param("low", ["minimal"], "minimal", id="low-down-to-minimal"),
        pytest.param("low", ["none", "low"], "low", id="ignore-none-and-match-low"),
        pytest.param("high", ["none", "low"], "low", id="ignore-none-and-step-down"),
    ],
)
def test_copilot_responses_warm_catalog_maps_minimal_exactly_or_downward(
    requested,
    allowed,
    expected,
) -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=allowed,
            )
        ]
    )

    payload = provider._build_responses_payload(
        ResponsesRequest(model="gpt-5.4", input="hi", reasoning_effort=requested)
    )

    assert payload["reasoning"] == {"effort": expected, "summary": "auto"}


@pytest.mark.parametrize("requested", ["minimal", "low"])
def test_copilot_responses_warm_catalog_none_only_rejects_public_effort(requested) -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["none"],
            )
        ]
    )

    with pytest.raises(RequestOptionError) as caught:
        provider._build_responses_payload(
            ResponsesRequest(model="gpt-5.4", input="hi", reasoning_effort=requested)
        )

    assert caught.value.parameter == "reasoning_effort"


def test_copilot_responses_warm_catalog_does_not_raise_minimal_to_low() -> None:
    provider = CopilotProvider()
    provider._models_ttl_cache.set(
        [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider="github-copilot",
                reasoning_effort_values=["low"],
            )
        ]
    )

    with pytest.raises(RequestOptionError) as caught:
        provider._build_responses_payload(
            ResponsesRequest(model="gpt-5.4", input="hi", reasoning_effort="minimal")
        )

    assert caught.value.parameter == "reasoning_effort"


@pytest.mark.parametrize("model", ["gpt-5.4", "future-reasoner"])
@pytest.mark.parametrize("adapter", ["chat", "responses"])
def test_copilot_cold_catalog_keeps_explicit_minimal_observable(model, adapter) -> None:
    provider = CopilotProvider()

    if adapter == "chat":
        payload = provider._build_chat_payload(
            _request(model=model, reasoning_effort="minimal"),
            stream=False,
        )
        assert payload["reasoning_effort"] == "minimal"
    else:
        payload = provider._build_responses_payload(
            ResponsesRequest(model=model, input="hi", reasoning_effort="minimal")
        )
        assert payload["reasoning"] == {"effort": "minimal", "summary": "auto"}


def test_router_model_rebuild_preserves_typed_options_and_extensions() -> None:
    request = _request(
        top_p=0.7,
        top_k=32,
        stop_sequences=["END"],
        metadata={"trace": "abc"},
        service_tier="standard_only",
        provider_extensions={"vendor": True},
    )

    rebuilt = Router()._create_request_with_model(request, "resolved-model")

    assert rebuilt.model == "resolved-model"
    assert rebuilt.top_p == 0.7
    assert rebuilt.top_k == 32
    assert rebuilt.stop_sequences == ["END"]
    assert rebuilt.metadata == {"trace": "abc"}
    assert rebuilt.service_tier == "standard_only"
    assert rebuilt.provider_extensions == {"vendor": True}
