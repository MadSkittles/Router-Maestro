"""Leak Guard: detect protocol-level leaks in streaming responses.

Detects two types of leaks from Copilot-served Claude models:

1. **Control envelope leaks** (abort on detection):
   - <task-notification>...</task-notification>
   - <teammate-message ...>...</teammate-message>
   - <channel ...>...</channel>
   - <cross-session-message ...>...</cross-session-message>
   - <tick>...</tick>

   These are Claude Code internal protocol envelopes that the model should
   never emit as output. When detected, the stream is aborted so the client
   retries the turn.

2. **Invoke leaks** (recover at stream finish):
   - <invoke name="X">...</invoke>

   Tool calls leaked as XML text instead of structured tool_use blocks.
   These are NOT abort triggers — instead, at stream end they are recovered
   into structured tool_calls via the existing recover_invoke_tool_calls().

Detection uses bounded-state prefix matchers that process each delta without
accumulating the full response text. Memory is O(max_tag_length), not O(response).
Invoke recovery still accumulates text (it needs the full content at finish).
"""

import json

from router_maestro.providers.base import ChatStreamChunk
from router_maestro.providers.tool_parsing import recover_invoke_tool_calls
from router_maestro.utils import get_logger

logger = get_logger("pipeline.leak_guard")

# Longest closing tag we need to detect across chunk boundaries.
# "</cross-session-message>" = 26 chars. Window must hold at least
# open + some content + close of the shortest envelope (<tick>x</tick> = 14).
_WINDOW_SIZE = 512

# Each entry: (open_prefix, close_tag). The open_prefix is what starts the
# envelope — we match it as a literal prefix. The close_tag completes it.
_CONTROL_TAGS: list[tuple[str, str, str]] = [
    ("<task-notification", "</task-notification>", "task-notification"),
    ("<teammate-message", "</teammate-message>", "teammate-message"),
    ("<channel", "</channel>", "channel"),
    ("<cross-session-message", "</cross-session-message>", "cross-session-message"),
    ("<tick>", "</tick>", "tick"),
]


class _TagMatcher:
    """Bounded state machine matching an open..close tag pair.

    States:
        IDLE       → scanning for open_prefix
        IN_OPEN    → saw start of open_prefix, matching rest
        OPEN       → open tag matched, scanning for close_tag
        IN_CLOSE   → saw start of close_tag, matching rest

    Memory: O(len(open_prefix) + len(close_tag)) — no content buffered.
    """

    __slots__ = (
        "_open",
        "_close",
        "_tag_name",
        "_state",
        "_oi",
        "_ci",
        "_content_len",
        "_in_fence",
    )

    IDLE = 0
    IN_OPEN = 1
    OPEN = 2
    IN_CLOSE = 3

    def __init__(self, open_prefix: str, close_tag: str, tag_name: str):
        self._open = open_prefix
        self._close = close_tag
        self._tag_name = tag_name
        self._state = self.IDLE
        self._oi = 0  # chars of open_prefix matched
        self._ci = 0  # chars of close_tag matched
        self._content_len = 0  # chars between open close (for non-empty check)
        self._in_fence = False

    @property
    def tag_name(self) -> str:
        return self._tag_name

    def reset(self) -> None:
        self._state = self.IDLE
        self._oi = 0
        self._ci = 0
        self._content_len = 0

    def feed(self, c: str) -> bool:
        """Feed one character. Returns True if a closed envelope is confirmed."""
        # Code-fence tracking: ``` toggles fence state.
        # We don't track inline code here — it's an approximation that covers
        # the majority of false-positive cases (fenced code blocks).
        # Full fence tracking would need backtick run counting; we simplify to
        # just tracking whether we've seen an odd number of ``` sequences.

        if self._state == self.IDLE:
            if c == self._open[0]:
                self._oi = 1
                if self._oi == len(self._open):
                    self._state = self.OPEN
                    self._ci = 0
                    self._content_len = 0
                else:
                    self._state = self.IN_OPEN
            return False

        if self._state == self.IN_OPEN:
            if c == self._open[self._oi]:
                self._oi += 1
                if self._oi == len(self._open):
                    # For tags like "<tick>" the open IS a complete tag.
                    # For "<task-notification" we need to see '>' or whitespace.
                    if self._open.endswith(">"):
                        self._state = self.OPEN
                        self._ci = 0
                        self._content_len = 0
                    else:
                        # Need delimiter after tag name (whitespace or >)
                        self._state = self.OPEN
                        self._ci = 0
                        self._content_len = 0
                return False
            else:
                # Mismatch — check if this char restarts the pattern
                self._state = self.IDLE
                self._oi = 0
                if c == self._open[0]:
                    self._oi = 1
                    self._state = self.IN_OPEN if self._oi < len(self._open) else self.OPEN
                return False

        if self._state == self.OPEN:
            self._content_len += 1
            if c == self._close[0]:
                self._ci = 1
                if self._ci == len(self._close):
                    # Closed! Confirm only if non-empty for <tick>
                    if self._tag_name == "tick" and self._content_len <= len(self._close):
                        self.reset()
                        return False
                    return True
                self._state = self.IN_CLOSE
            return False

        if self._state == self.IN_CLOSE:
            self._content_len += 1
            if c == self._close[self._ci]:
                self._ci += 1
                if self._ci == len(self._close):
                    if self._tag_name == "tick" and self._content_len <= len(self._close):
                        self.reset()
                        return False
                    return True
                return False
            else:
                # Mismatch in close tag — back to OPEN state
                self._ci = 0
                self._state = self.OPEN
                # Check if this char starts the close tag
                if c == self._close[0]:
                    self._ci = 1
                    self._state = self.IN_CLOSE
                return False

        return False


class LeakGuard:
    """Streaming guard that detects leaked protocol markup.

    Control envelopes are detected via O(1)-state character-fed matchers.
    Invoke leaks are accumulated for recovery at stream finish.

    Memory: O(window_size) for control detection + O(response_length) for
    invoke recovery. The invoke accumulation is necessary because recovery
    needs the full text to parse parameters.
    """

    def __init__(self, allowed_tool_names: set[str] | None = None):
        self._tool_names = allowed_tool_names
        self._tripped = False
        self._trip_reason: str | None = None
        self._in_fence = False
        self._backtick_run = 0

        # Character-fed matchers for control envelopes
        self._matchers = [
            _TagMatcher(open_prefix, close_tag, tag_name)
            for open_prefix, close_tag, tag_name in _CONTROL_TAGS
        ]

        # Invoke recovery still needs accumulated text
        self._invoke_text = ""

    def feed_chunk(self, chunk: ChatStreamChunk) -> str | None:
        return None

    def feed_text(self, text: str) -> str | None:
        """Feed a text delta. Returns abort reason if control envelope detected."""
        if self._tripped:
            return self._trip_reason

        # Accumulate for invoke recovery (unavoidable — recovery needs full text)
        self._invoke_text += text

        # Feed character-by-character to matchers
        for c in text:
            # Track code fences (``` toggles)
            if c == "`":
                self._backtick_run += 1
            else:
                if self._backtick_run >= 3:
                    self._in_fence = not self._in_fence
                self._backtick_run = 0

            # Skip detection inside code fences
            if self._in_fence:
                continue

            for matcher in self._matchers:
                if matcher.feed(c):
                    self._tripped = True
                    self._trip_reason = f"response_leak:control_envelope:{matcher.tag_name}"
                    logger.warning(
                        "Control envelope leak detected: <%s>",
                        matcher.tag_name,
                    )
                    return self._trip_reason

        return None

    def check_invoke_at_finish(self) -> list[dict] | None:
        """Called at stream end to recover any leaked invoke tool calls."""
        if not self._invoke_text:
            return None
        return recover_invoke_tool_calls(self._invoke_text, self._tool_names)

    @property
    def accumulated_text(self) -> str:
        return self._invoke_text


class RawFrameLeakGuard:
    """Leak guard for the anthropic_beta passthrough route.

    Operates on raw SSE frame data strings. Extracts text from
    content_block_delta events and feeds them to an inner LeakGuard.
    """

    def __init__(
        self,
        allowed_tool_names: set[str] | None = None,
        *,
        inner: LeakGuard | None = None,
    ):
        self._inner = inner or LeakGuard(allowed_tool_names=allowed_tool_names)

    def feed_frame(self, event_type: str, data_str: str) -> str | None:
        """Feed a raw SSE frame. Returns abort reason if leak detected."""
        if event_type != "content_block_delta":
            return None

        text = self._extract_text(data_str)
        if text:
            return self._inner.feed_text(text)
        return None

    @staticmethod
    def _extract_text(data_str: str) -> str | None:
        """Extract text content from a content_block_delta payload."""
        try:
            data = json.loads(data_str)
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                return delta.get("text")
            if delta_type == "thinking_delta":
                return delta.get("thinking")
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return None
