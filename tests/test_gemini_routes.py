"""Tests for Gemini-compatible API routes and translation."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.providers.base import ChatResponse, ChatStreamChunk
from router_maestro.server.routes.gemini import (
    TEST_RESPONSE_TEXT,
    _estimate_input_tokens,
    _extract_model_from_path,
    router,
)
from router_maestro.server.schemas.gemini import (
    GeminiCandidate,
    GeminiContent,
    GeminiFunctionCall,
    GeminiFunctionCallingConfig,
    GeminiFunctionDeclaration,
    GeminiFunctionResponse,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiGenerationConfig,
    GeminiPart,
    GeminiStreamState,
    GeminiTool,
    GeminiToolConfig,
    GeminiUsageMetadata,
)
from router_maestro.server.translation_gemini import (
    normalize_schema_types,
    translate_gemini_to_openai,
    translate_openai_chunk_to_gemini,
    translate_openai_to_gemini,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def app():
    """Create a test FastAPI app with the Gemini router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_router():
    """Create a mock model router."""
    mock = AsyncMock()
    mock.chat_completion = AsyncMock(
        return_value=(
            ChatResponse(
                content="Hello! How can I help?",
                model="gemini-2.5-pro",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            ),
            "github-copilot",
        )
    )
    return mock


# ============================================================================
# Schema type normalization tests
# ============================================================================


class TestNormalizeSchemaTypes:
    """Tests for schema type normalization (uppercase -> lowercase)."""

    def test_uppercase_string(self):
        assert normalize_schema_types({"type": "STRING"}) == {
            "type": "string"
        }

    def test_uppercase_object(self):
        assert normalize_schema_types({"type": "OBJECT"}) == {
            "type": "object"
        }

    def test_uppercase_array(self):
        assert normalize_schema_types({"type": "ARRAY"}) == {"type": "array"}

    def test_uppercase_boolean(self):
        assert normalize_schema_types({"type": "BOOLEAN"}) == {
            "type": "boolean"
        }

    def test_uppercase_number(self):
        assert normalize_schema_types({"type": "NUMBER"}) == {
            "type": "number"
        }

    def test_uppercase_integer(self):
        assert normalize_schema_types({"type": "INTEGER"}) == {
            "type": "integer"
        }

    def test_lowercase_passthrough(self):
        assert normalize_schema_types({"type": "string"}) == {
            "type": "string"
        }

    def test_mixed_case(self):
        assert normalize_schema_types({"type": "String"}) == {
            "type": "string"
        }

    def test_type_unspecified_removed(self):
        result = normalize_schema_types(
            {"type": "TYPE_UNSPECIFIED", "description": "test"}
        )
        assert "type" not in result
        assert result["description"] == "test"

    def test_nested_properties(self):
        schema = {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "age": {"type": "INTEGER"},
            },
        }
        result = normalize_schema_types(schema)
        assert result["type"] == "object"
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["age"]["type"] == "integer"

    def test_array_items(self):
        schema = {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        }
        result = normalize_schema_types(schema)
        assert result["type"] == "array"
        assert result["items"]["type"] == "string"

    def test_non_schema_fields_preserved(self):
        """default, example, const, enum should not be recursed into."""
        schema = {
            "type": "OBJECT",
            "default": {"type": "OBJECT"},
            "enum": [{"type": "STRING"}],
        }
        result = normalize_schema_types(schema)
        assert result["type"] == "object"
        # Non-schema fields kept as-is
        assert result["default"] == {"type": "OBJECT"}
        assert result["enum"] == [{"type": "STRING"}]

    def test_none_passthrough(self):
        assert normalize_schema_types(None) is None

    def test_primitive_passthrough(self):
        assert normalize_schema_types("hello") == "hello"
        assert normalize_schema_types(42) == 42

    def test_list_normalization(self):
        result = normalize_schema_types(
            [{"type": "STRING"}, {"type": "INTEGER"}]
        )
        assert result == [{"type": "string"}, {"type": "integer"}]

    def test_deeply_nested(self):
        schema = {
            "type": "OBJECT",
            "properties": {
                "items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "value": {"type": "STRING"},
                        },
                    },
                }
            },
        }
        result = normalize_schema_types(schema)
        inner = result["properties"]["items"]["items"]["properties"]["value"]
        assert inner["type"] == "string"


# ============================================================================
# Gemini -> OpenAI translation tests
# ============================================================================


class TestTranslateGeminiToOpenAI:
    """Tests for Gemini to OpenAI request translation."""

    def test_simple_text_message(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user",
                    parts=[GeminiPart(text="Hello")],
                )
            ]
        )
        result = translate_gemini_to_openai(request, "gemini-2.5-pro")
        assert result.model == "gemini-2.5-pro"
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"

    def test_system_instruction(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user",
                    parts=[GeminiPart(text="Hello")],
                )
            ],
            system_instruction=GeminiContent(
                parts=[GeminiPart(text="You are helpful")]
            ),
        )
        result = translate_gemini_to_openai(request, "model")
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are helpful"
        assert result.messages[1].role == "user"

    def test_multi_turn_conversation(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hi")]
                ),
                GeminiContent(
                    role="model", parts=[GeminiPart(text="Hello!")]
                ),
                GeminiContent(
                    role="user", parts=[GeminiPart(text="How are you?")]
                ),
            ]
        )
        result = translate_gemini_to_openai(request, "model")
        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"
        assert result.messages[2].role == "user"

    def test_generation_config(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hi")]
                )
            ],
            generation_config=GeminiGenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,
            ),
        )
        result = translate_gemini_to_openai(request, "model")
        assert result.temperature == 0.7
        assert result.max_tokens == 8192

    def test_function_call_in_model_content(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user",
                    parts=[GeminiPart(text="What's the weather?")],
                ),
                GeminiContent(
                    role="model",
                    parts=[
                        GeminiPart(
                            function_call=GeminiFunctionCall(
                                name="get_weather",
                                args={"location": "Tokyo"},
                                id="call_123",
                            )
                        )
                    ],
                ),
                GeminiContent(
                    role="user",
                    parts=[
                        GeminiPart(
                            function_response=GeminiFunctionResponse(
                                name="get_weather",
                                id="call_123",
                                response={"temp": "20C"},
                            )
                        )
                    ],
                ),
            ]
        )
        result = translate_gemini_to_openai(request, "model")
        # user, assistant (with tool_calls), tool
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].tool_calls is not None
        assert (
            result.messages[1].tool_calls[0]["function"]["name"]
            == "get_weather"
        )
        assert result.messages[2].role == "tool"
        assert result.messages[2].tool_call_id == "call_123"

    def test_tools_translation(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hi")]
                )
            ],
            tools=[
                GeminiTool(
                    function_declarations=[
                        GeminiFunctionDeclaration(
                            name="get_weather",
                            description="Get weather",
                            parameters={
                                "type": "OBJECT",
                                "properties": {
                                    "location": {"type": "STRING"}
                                },
                                "required": ["location"],
                            },
                        )
                    ]
                )
            ],
        )
        result = translate_gemini_to_openai(request, "model")
        assert result.tools is not None
        assert len(result.tools) == 1
        tool = result.tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        # Schema types should be normalized
        assert tool["function"]["parameters"]["type"] == "object"
        props = tool["function"]["parameters"]["properties"]
        assert props["location"]["type"] == "string"

    def test_tool_config_auto(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hi")]
                )
            ],
            tool_config=GeminiToolConfig(
                function_calling_config=GeminiFunctionCallingConfig(
                    mode="AUTO"
                )
            ),
        )
        result = translate_gemini_to_openai(request, "model")
        assert result.tool_choice == "auto"

    def test_tool_config_any(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hi")]
                )
            ],
            tool_config=GeminiToolConfig(
                function_calling_config=GeminiFunctionCallingConfig(
                    mode="ANY"
                )
            ),
        )
        result = translate_gemini_to_openai(request, "model")
        assert result.tool_choice == "required"

    def test_tool_config_none(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hi")]
                )
            ],
            tool_config=GeminiToolConfig(
                function_calling_config=GeminiFunctionCallingConfig(
                    mode="NONE"
                )
            ),
        )
        result = translate_gemini_to_openai(request, "model")
        assert result.tool_choice == "none"

    def test_empty_contents(self):
        request = GeminiGenerateContentRequest(contents=[])
        result = translate_gemini_to_openai(request, "model")
        assert result.messages == []

    def test_default_temperature(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hi")]
                )
            ]
        )
        result = translate_gemini_to_openai(request, "model")
        assert result.temperature == 1.0


# ============================================================================
# OpenAI -> Gemini translation tests (non-streaming)
# ============================================================================


class TestTranslateOpenAIToGemini:
    """Tests for OpenAI to Gemini response translation."""

    def test_simple_text_response(self):
        response = ChatResponse(
            content="Hello!",
            model="gemini-2.5-pro",
            finish_reason="stop",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        result = translate_openai_to_gemini(response, "gemini-2.5-pro")
        assert len(result.candidates) == 1
        assert result.candidates[0].content.parts[0].text == "Hello!"
        assert result.candidates[0].content.role == "model"
        assert result.candidates[0].finish_reason == "STOP"
        assert result.usage_metadata.prompt_token_count == 10
        assert result.usage_metadata.candidates_token_count == 5
        assert result.model_version == "gemini-2.5-pro"

    def test_length_finish_reason(self):
        response = ChatResponse(
            content="truncated",
            model="model",
            finish_reason="length",
        )
        result = translate_openai_to_gemini(response, "model")
        assert result.candidates[0].finish_reason == "MAX_TOKENS"

    def test_input_tokens_fallback(self):
        """When response has no usage, input_tokens estimate is used."""
        response = ChatResponse(
            content="Hi", model="model", usage=None
        )
        result = translate_openai_to_gemini(
            response, "model", input_tokens=42
        )
        assert result.usage_metadata.prompt_token_count == 42

    def test_empty_content(self):
        response = ChatResponse(content="", model="model")
        result = translate_openai_to_gemini(response, "model")
        assert len(result.candidates) == 1

    def test_json_serialization_uses_camel_case(self):
        """Verify JSON output uses camelCase field names."""
        response = ChatResponse(
            content="Hi",
            model="model",
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 3},
        )
        result = translate_openai_to_gemini(response, "model")
        data = json.loads(
            result.model_dump_json(by_alias=True, exclude_none=True)
        )
        assert "finishReason" in data["candidates"][0]
        assert "usageMetadata" in data
        assert "promptTokenCount" in data["usageMetadata"]
        assert "modelVersion" in data


# ============================================================================
# Streaming translation tests
# ============================================================================


class TestTranslateOpenAIChunkToGemini:
    """Tests for streaming chunk translation."""

    def test_text_chunk(self):
        state = GeminiStreamState()
        chunk = {
            "choices": [
                {"delta": {"content": "Hello"}, "finish_reason": None}
            ],
            "usage": None,
        }
        result = translate_openai_chunk_to_gemini(chunk, state, "model")
        assert result is not None
        assert result.candidates[0].content.parts[0].text == "Hello"
        assert state.accumulated_text == "Hello"

    def test_finish_chunk(self):
        state = GeminiStreamState(estimated_input_tokens=10)
        chunk = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = translate_openai_chunk_to_gemini(chunk, state, "model")
        assert result is not None
        assert result.candidates[0].finish_reason == "STOP"
        assert result.usage_metadata is not None
        assert result.model_version == "model"

    def test_empty_chunk_returns_none(self):
        state = GeminiStreamState()
        chunk = {
            "choices": [{"delta": {}, "finish_reason": None}],
            "usage": None,
        }
        result = translate_openai_chunk_to_gemini(chunk, state, "model")
        assert result is None

    def test_no_choices_returns_none(self):
        state = GeminiStreamState()
        chunk = {"usage": {"prompt_tokens": 5}}
        result = translate_openai_chunk_to_gemini(chunk, state, "model")
        assert result is None

    def test_tool_call_chunk(self):
        state = GeminiStreamState()
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        }
        result = translate_openai_chunk_to_gemini(chunk, state, "model")
        assert result is not None
        fc = result.candidates[0].content.parts[0].function_call
        assert fc is not None
        assert fc.name == "get_weather"

    def test_tool_call_accumulation_on_finish(self):
        state = GeminiStreamState()
        # First chunk: start of tool call
        chunk1 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"loc',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        }
        translate_openai_chunk_to_gemini(chunk1, state, "model")

        # Second chunk: continuation
        chunk2 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": 'ation": "Tokyo"}'
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        }
        translate_openai_chunk_to_gemini(chunk2, state, "model")

        # Third chunk: finish
        chunk3 = {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": None,
        }
        result = translate_openai_chunk_to_gemini(chunk3, state, "model")
        assert result is not None
        assert result.candidates[0].finish_reason == "STOP"
        # The final tool call should have parsed args
        fc = result.candidates[0].content.parts[0].function_call
        assert fc.name == "get_weather"
        assert fc.args == {"location": "Tokyo"}

    def test_usage_tracking(self):
        state = GeminiStreamState()
        chunk = {
            "choices": [
                {"delta": {"content": "Hi"}, "finish_reason": None}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3},
        }
        translate_openai_chunk_to_gemini(chunk, state, "model")
        assert state.accumulated_prompt_tokens == 20
        assert state.accumulated_completion_tokens == 3


# ============================================================================
# Route helper tests
# ============================================================================


class TestRouteHelpers:
    """Tests for route helper functions."""

    def test_extract_model_from_path_with_method(self):
        result = _extract_model_from_path("gemini-2.5-pro:generateContent")
        assert result == "gemini-2.5-pro"

    def test_extract_model_from_path_no_method(self):
        assert _extract_model_from_path("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_estimate_input_tokens_non_empty(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hello world")]
                )
            ]
        )
        tokens = _estimate_input_tokens(request)
        assert tokens > 0

    def test_estimate_input_tokens_empty(self):
        request = GeminiGenerateContentRequest()
        tokens = _estimate_input_tokens(request)
        assert tokens >= 0


# ============================================================================
# Route integration tests
# ============================================================================


class TestGenerateContentEndpoint:
    """Integration tests for the generateContent endpoint."""

    def test_test_model(self, client):
        """Test model returns a mock response."""
        response = client.post(
            "/api/gemini/v1beta/models/test:generateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "Hello"}]}
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "candidates" in data
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        assert text == TEST_RESPONSE_TEXT
        assert data["candidates"][0]["finishReason"] == "STOP"
        assert "usageMetadata" in data
        assert data["modelVersion"] == "test"

    def test_generate_content_success(self, client, mock_router):
        """Test successful generateContent with mocked router."""
        with patch(
            "router_maestro.server.routes.gemini.get_router",
            return_value=mock_router,
        ):
            response = client.post(
                "/api/gemini/v1beta/models/gemini-2.5-pro:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Hello"}]}
                    ],
                },
            )
        assert response.status_code == 200
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        assert text == "Hello! How can I help?"
        assert data["candidates"][0]["finishReason"] == "STOP"

    def test_generate_content_with_system_instruction(
        self, client, mock_router
    ):
        """Test generateContent with system instruction."""
        with patch(
            "router_maestro.server.routes.gemini.get_router",
            return_value=mock_router,
        ):
            response = client.post(
                "/api/gemini/v1beta/models/gemini-2.5-pro:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Hi"}]}
                    ],
                    "systemInstruction": {
                        "parts": [{"text": "You are a helpful assistant"}]
                    },
                },
            )
        assert response.status_code == 200

    def test_generate_content_with_generation_config(
        self, client, mock_router
    ):
        """Test generateContent with generation config."""
        with patch(
            "router_maestro.server.routes.gemini.get_router",
            return_value=mock_router,
        ):
            response = client.post(
                "/api/gemini/v1beta/models/gemini-2.5-pro:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Hi"}]}
                    ],
                    "generationConfig": {
                        "temperature": 0.5,
                        "maxOutputTokens": 1024,
                    },
                },
            )
        assert response.status_code == 200

    def test_generate_content_provider_error(self, client):
        """Test generateContent returns error on provider failure."""
        from router_maestro.providers import ProviderError

        mock = AsyncMock()
        mock.chat_completion = AsyncMock(
            side_effect=ProviderError("Model not found", status_code=404)
        )
        with patch(
            "router_maestro.server.routes.gemini.get_router",
            return_value=mock,
        ):
            response = client.post(
                "/api/gemini/v1beta/models/bad-model:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Hi"}]}
                    ],
                },
            )
        assert response.status_code == 404


class TestStreamGenerateContentEndpoint:
    """Integration tests for the streamGenerateContent endpoint."""

    def test_test_model_streaming(self, client):
        """Test model returns a streamed mock response."""
        response = client.post(
            "/api/gemini/v1beta/models/test:streamGenerateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "Hello"}]}
                ],
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        events = _parse_sse_events(response.text)
        assert len(events) >= 2  # content + final

        # First event should have text content
        first = json.loads(events[0])
        text = first["candidates"][0]["content"]["parts"][0]["text"]
        assert text == TEST_RESPONSE_TEXT

        # Last event should have finishReason and usage
        last = json.loads(events[-1])
        assert last["candidates"][0]["finishReason"] == "STOP"
        assert "usageMetadata" in last

    def test_stream_generate_content_success(self, client):
        """Test streaming with mocked router."""

        async def mock_stream():
            yield ChatStreamChunk(
                content="Hello ", finish_reason=None
            )
            yield ChatStreamChunk(
                content="world!",
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 2},
            )

        mock = AsyncMock()
        mock.chat_completion_stream = AsyncMock(
            return_value=(mock_stream(), "test-provider")
        )

        with patch(
            "router_maestro.server.routes.gemini.get_router",
            return_value=mock,
        ):
            response = client.post(
                "/api/gemini/v1beta/models/gemini-2.5-pro"
                ":streamGenerateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Hi"}]}
                    ],
                },
            )
        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        assert len(events) >= 2


class TestCountTokensEndpoint:
    """Integration tests for the countTokens endpoint."""

    def test_count_tokens(self, client):
        """Test basic token counting."""
        response = client.post(
            "/api/gemini/v1beta/models/gemini-2.5-pro:countTokens",
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Hello world"}],
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "totalTokens" in data
        assert data["totalTokens"] > 0

    def test_count_tokens_empty_request(self, client):
        """Test token counting with minimal request."""
        response = client.post(
            "/api/gemini/v1beta/models/gemini-2.5-pro:countTokens",
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert "totalTokens" in data


# ============================================================================
# Schema tests
# ============================================================================


class TestGeminiSchemas:
    """Tests for Gemini Pydantic schemas."""

    def test_generate_content_request_minimal(self):
        req = GeminiGenerateContentRequest()
        assert req.contents is None
        assert req.system_instruction is None

    def test_generate_content_request_full(self):
        req = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user", parts=[GeminiPart(text="Hi")]
                )
            ],
            system_instruction=GeminiContent(
                parts=[GeminiPart(text="Be helpful")]
            ),
            generation_config=GeminiGenerationConfig(temperature=0.5),
            tools=[
                GeminiTool(
                    function_declarations=[
                        GeminiFunctionDeclaration(
                            name="test", description="A test"
                        )
                    ]
                )
            ],
        )
        assert len(req.contents) == 1
        assert req.generation_config.temperature == 0.5

    def test_generate_content_response(self):
        resp = GeminiGenerateContentResponse(
            candidates=[
                GeminiCandidate(
                    content=GeminiContent(
                        parts=[GeminiPart(text="Hello")],
                        role="model",
                    ),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=GeminiUsageMetadata(
                prompt_token_count=5,
                candidates_token_count=3,
                total_token_count=8,
            ),
            model_version="test",
        )
        assert resp.candidates[0].content.parts[0].text == "Hello"
        assert resp.usage_metadata.total_token_count == 8

    def test_function_call_part(self):
        part = GeminiPart(
            function_call=GeminiFunctionCall(
                name="test_fn",
                args={"key": "value"},
                id="call_1",
            )
        )
        assert part.function_call.name == "test_fn"
        assert part.function_call.args == {"key": "value"}

    def test_function_response_part(self):
        part = GeminiPart(
            function_response=GeminiFunctionResponse(
                name="test_fn",
                id="call_1",
                response={"result": "ok"},
            )
        )
        assert part.function_response.name == "test_fn"

    def test_stream_state_defaults(self):
        state = GeminiStreamState()
        assert state.accumulated_text == ""
        assert state.estimated_input_tokens == 0
        assert state.has_sent_content is False
        assert state.tool_calls_buffer == []

    def test_request_from_camel_case_json(self):
        """Verify requests can be parsed from camelCase JSON."""
        data = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hi"}]}
            ],
            "systemInstruction": {"parts": [{"text": "Be helpful"}]},
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 1024,
            },
        }
        req = GeminiGenerateContentRequest.model_validate(data)
        assert req.system_instruction is not None
        assert req.generation_config.max_output_tokens == 1024

    def test_response_serializes_to_camel_case(self):
        """Verify responses serialize to camelCase JSON."""
        resp = GeminiGenerateContentResponse(
            candidates=[
                GeminiCandidate(finish_reason="STOP", index=0)
            ],
            usage_metadata=GeminiUsageMetadata(
                prompt_token_count=5,
                candidates_token_count=3,
                total_token_count=8,
            ),
            model_version="test",
        )
        data = json.loads(
            resp.model_dump_json(by_alias=True, exclude_none=True)
        )
        assert "finishReason" in data["candidates"][0]
        assert "usageMetadata" in data
        assert "promptTokenCount" in data["usageMetadata"]
        assert "modelVersion" in data


# ============================================================================
# Helpers
# ============================================================================


def _parse_sse_events(text: str) -> list[str]:
    """Parse SSE event data from raw response text."""
    events = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data and data != "[DONE]":
                events.append(data)
    return events


# ============================================================================
# Tool parameters default schema tests
# ============================================================================


class TestToolParametersDefaults:
    """Test that tools without parameters get valid default schema."""

    def test_tool_without_parameters_gets_default(self):
        """Tools with no parameters field should get a valid OpenAI schema."""
        tools = [
            GeminiTool(
                function_declarations=[
                    GeminiFunctionDeclaration(
                        name="list_files",
                        description="List files in directory",
                        parameters=None,
                    )
                ]
            )
        ]
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    parts=[GeminiPart(text="hello")], role="user"
                )
            ],
            tools=tools,
        )
        chat_request = translate_gemini_to_openai(request, "test-model")
        assert chat_request.tools is not None
        assert len(chat_request.tools) == 1
        params = chat_request.tools[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params

    def test_tool_with_empty_parameters_gets_default(self):
        """Tools with empty {} parameters should get a valid schema."""
        tools = [
            GeminiTool(
                function_declarations=[
                    GeminiFunctionDeclaration(
                        name="get_time",
                        description="Get current time",
                        parameters={},
                    )
                ]
            )
        ]
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    parts=[GeminiPart(text="hello")], role="user"
                )
            ],
            tools=tools,
        )
        chat_request = translate_gemini_to_openai(request, "test-model")
        params = chat_request.tools[0]["function"]["parameters"]
        assert params["type"] == "object"

    def test_tool_with_valid_parameters_preserved(self):
        """Tools with valid parameters should keep them as-is."""
        tools = [
            GeminiTool(
                function_declarations=[
                    GeminiFunctionDeclaration(
                        name="read_file",
                        description="Read a file",
                        parameters={
                            "type": "OBJECT",
                            "properties": {
                                "path": {"type": "STRING"}
                            },
                            "required": ["path"],
                        },
                    )
                ]
            )
        ]
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    parts=[GeminiPart(text="hello")], role="user"
                )
            ],
            tools=tools,
        )
        chat_request = translate_gemini_to_openai(request, "test-model")
        params = chat_request.tools[0]["function"]["parameters"]
        assert params["type"] == "object"  # normalized from OBJECT
        assert "path" in params["properties"]

    def test_many_tools_without_parameters(self):
        """Multiple tools without parameters (like Gemini CLI sends)."""
        decls = [
            GeminiFunctionDeclaration(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters=None,
            )
            for i in range(16)
        ]
        tools = [GeminiTool(function_declarations=decls)]
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    parts=[GeminiPart(text="hello")], role="user"
                )
            ],
            tools=tools,
        )
        chat_request = translate_gemini_to_openai(request, "test-model")
        assert chat_request.tools is not None
        assert len(chat_request.tools) == 16
        for tool in chat_request.tools:
            params = tool["function"]["parameters"]
            assert params["type"] == "object"


# ============================================================================
# Auth middleware x-goog-api-key tests
# ============================================================================


class TestGoogApiKeyAuth:
    """Test x-goog-api-key header authentication."""

    @pytest.fixture
    def auth_app(self):
        """Create a FastAPI app with auth middleware."""
        from router_maestro.server.middleware.auth import verify_api_key

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint(dep=pytest.importorskip("fastapi").Depends(verify_api_key)):
            return {"ok": True}

        return app

    @pytest.fixture
    def auth_client(self, auth_app):
        return TestClient(auth_app)

    @patch.dict("os.environ", {"ROUTER_MAESTRO_API_KEY": "test-key-123"})
    def test_x_goog_api_key_accepted(self, auth_client):
        """x-goog-api-key header should authenticate successfully."""
        response = auth_client.get(
            "/test", headers={"x-goog-api-key": "test-key-123"}
        )
        assert response.status_code == 200

    @patch.dict("os.environ", {"ROUTER_MAESTRO_API_KEY": "test-key-123"})
    def test_x_goog_api_key_wrong_key(self, auth_client):
        """Wrong x-goog-api-key should return 401."""
        response = auth_client.get(
            "/test", headers={"x-goog-api-key": "wrong-key"}
        )
        assert response.status_code == 401

    @patch.dict("os.environ", {"ROUTER_MAESTRO_API_KEY": "test-key-123"})
    def test_bearer_still_works(self, auth_client):
        """Bearer auth should still work alongside x-goog-api-key support."""
        response = auth_client.get(
            "/test",
            headers={"Authorization": "Bearer test-key-123"},
        )
        assert response.status_code == 200

    @patch.dict("os.environ", {"ROUTER_MAESTRO_API_KEY": "test-key-123"})
    def test_x_api_key_still_works(self, auth_client):
        """x-api-key header should still work."""
        response = auth_client.get(
            "/test", headers={"x-api-key": "test-key-123"}
        )
        assert response.status_code == 200
