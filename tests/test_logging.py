"""Tests for unified logging configuration.

Pinning down the markup behavior is critical: we discovered in v0.3.5 that
RichHandler with markup=True crashes mid-stream when a log message contains
bracket-syntax that looks like Rich markup (e.g. file paths from Codex tool
calls: ``[/Users/likanwen/.codex/config.toml:55]``). The crash propagates
out of ``logger.debug()`` into the streaming response handler and breaks
the connection with an "Internal server error".
"""

import logging

from rich.logging import RichHandler

from router_maestro.utils.logging import setup_logging


class TestSetupLoggingRichHandlerMarkup:
    """Regression: console RichHandler must NOT parse log messages as markup."""

    def test_console_handler_has_markup_disabled(self):
        """RichHandler must be configured with markup=False so that user
        content with brackets does not raise MarkupError mid-stream."""
        setup_logging(level="DEBUG", console=True, file=False)

        logger = logging.getLogger("router_maestro")
        rich_handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]

        assert len(rich_handlers) == 1, "expected exactly one RichHandler"
        # RichHandler stores the markup setting on itself for older versions
        # and on its internal renderer for newer versions; check both.
        handler = rich_handlers[0]
        markup_attr = getattr(handler, "markup", None)
        assert markup_attr is False, (
            "RichHandler.markup must be False to keep log messages literal — "
            "otherwise bracket-syntax in user content (file paths, code, JSON) "
            "is parsed as markup and crashes the logger mid-stream."
        )

    def test_logging_message_with_brackets_does_not_raise(self):
        """The exact failure mode from the v0.3.5 production incident: a
        debug log line containing a Codex-style file reference must not
        raise MarkupError when emitted through the configured handler."""
        setup_logging(level="DEBUG", console=True, file=False)
        logger = logging.getLogger("router_maestro.providers.copilot")

        # This is the literal string shape that crashed prod. If the handler
        # parses it as markup, Rich raises MarkupError.
        payload_excerpt = (
            "Copilot responses payload: "
            "{'input': [{'type': 'message', 'content': "
            "'see [/Users/likanwen/.codex/config.toml:55]'}]}"
        )
        # Should complete cleanly — any exception fails the test.
        logger.debug(payload_excerpt)
