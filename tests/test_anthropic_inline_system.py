"""Tests for hoisting inline ``role="system"`` messages into the top-level ``system`` field.

The Anthropic Messages API only accepts ``user`` and ``assistant`` roles in
``messages``. Some clients still inline a system message into the array; we
silently hoist it so the request validates and routes normally instead of
returning 422.
"""

import pytest
from pydantic import ValidationError

from router_maestro.server.schemas.anthropic import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)


class TestHoistInlineSystemMessages:
    def test_inline_system_string_hoisted_when_no_top_level(self):
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "system", "content": "You are concise."},
                    {"role": "assistant", "content": "Hello."},
                ],
            }
        )
        assert req.system == "You are concise."
        assert len(req.messages) == 2
        assert req.messages[0].role == "user"
        assert req.messages[1].role == "assistant"

    def test_inline_system_block_list_hoisted(self):
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "First rule."},
                            {"type": "text", "text": "Second rule."},
                        ],
                    },
                    {"role": "user", "content": "Hi"},
                ],
            }
        )
        assert req.system == "First rule.\n\nSecond rule."
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_inline_system_merges_with_existing_system_string(self):
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "system": "Be helpful.",
                "messages": [
                    {"role": "system", "content": "Also be terse."},
                    {"role": "user", "content": "Hi"},
                ],
            }
        )
        assert req.system == "Be helpful.\n\nAlso be terse."

    def test_inline_system_appends_to_existing_system_list(self):
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "system": [{"type": "text", "text": "Be helpful."}],
                "messages": [
                    {"role": "system", "content": "Also be terse."},
                    {"role": "user", "content": "Hi"},
                ],
            }
        )
        assert isinstance(req.system, list)
        assert len(req.system) == 2
        assert req.system[0].text == "Be helpful."
        assert req.system[1].text == "Also be terse."

    def test_multiple_inline_system_messages_concatenated(self):
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [
                    {"role": "system", "content": "Rule A."},
                    {"role": "user", "content": "Hi"},
                    {"role": "system", "content": "Rule B."},
                    {"role": "assistant", "content": "Hello."},
                ],
            }
        )
        assert req.system == "Rule A.\n\nRule B."
        assert [m.role for m in req.messages] == ["user", "assistant"]

    def test_no_inline_system_leaves_request_unchanged(self):
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "system": "Top-level only.",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        assert req.system == "Top-level only."
        assert len(req.messages) == 1

    def test_empty_inline_system_content_is_dropped_silently(self):
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "Hi"},
                ],
            }
        )
        assert req.system is None
        assert len(req.messages) == 1

    def test_count_tokens_request_also_hoists_inline_system(self):
        req = AnthropicCountTokensRequest.model_validate(
            {
                "model": "claude-opus-4-8",
                "messages": [
                    {"role": "system", "content": "You are concise."},
                    {"role": "user", "content": "Hi"},
                ],
            }
        )
        assert req.system == "You are concise."
        assert len(req.messages) == 1

    def test_unknown_role_still_rejected(self):
        with pytest.raises(ValidationError):
            AnthropicMessagesRequest.model_validate(
                {
                    "model": "claude-opus-4-8",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "tool", "content": "nope"},
                    ],
                }
            )
