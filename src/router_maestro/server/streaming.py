"""Shared SSE streaming utilities with keepalive and error handling."""

import asyncio
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse

from router_maestro.utils import get_logger

logger = get_logger("server.streaming")

SSE_KEEPALIVE_INTERVAL = 5.0  # seconds

SSE_KEEPALIVE_COMMENT = ": keepalive\n\n"

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def resilient_sse_generator(
    inner: AsyncGenerator[str, None],
    keepalive_frame: str = SSE_KEEPALIVE_COMMENT,
) -> AsyncGenerator[str, None]:
    """Wrap an SSE generator with keepalive frames and error handling.

    Emits ``keepalive_frame`` every ``SSE_KEEPALIVE_INTERVAL`` seconds when
    no data is produced by the inner generator.  Defaults to a SSE comment
    (``: keepalive\\n\\n``), which is ignored by compliant clients but keeps
    bytes flowing on the wire to prevent intermediary proxies and client
    socket timeouts from killing the connection.

    Routes whose clients track a "time since last *protocol* event" timer
    (notably Claude Code, which times out at ~15s of comment-only traffic
    on the Anthropic Messages stream) should pass a real protocol event
    frame here instead — e.g. ``event: ping\\ndata: {"type":"ping"}\\n\\n``.

    Exception handling:
    - ``asyncio.CancelledError`` — logged as client disconnect, re-raised
    - ``ConnectionResetError`` — logged as warning
    - Any other ``Exception`` — logged with traceback
    """
    # We use asyncio.wait with a timeout so that we never cancel the
    # in-flight ``__anext__`` call.  Cancelling an async-generator's
    # ``__anext__`` can leave the generator in a broken state.
    next_task: asyncio.Task[str] | None = None
    try:
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(inner.__anext__())

            done, _ = await asyncio.wait(
                {next_task},
                timeout=SSE_KEEPALIVE_INTERVAL,
            )

            if done:
                try:
                    value = next_task.result()
                except StopAsyncIteration:
                    return
                next_task = None
                yield value
            else:
                # Timeout — no data within the keepalive interval
                yield keepalive_frame

    except asyncio.CancelledError:
        logger.info("Client disconnected (stream cancelled)")
        raise
    except ConnectionResetError:
        logger.warning("Client connection reset during stream")
    except Exception:
        logger.error("Unexpected error in SSE stream", exc_info=True)
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
            try:
                await next_task
            except BaseException:
                pass
        await inner.aclose()


def sse_streaming_response(
    generator: AsyncGenerator[str, None],
    keepalive_frame: str = SSE_KEEPALIVE_COMMENT,
) -> StreamingResponse:
    """Create a ``StreamingResponse`` with SSE headers and keepalive wrapper.

    This is the single entry-point that all streaming endpoints should use
    instead of constructing ``StreamingResponse`` directly.

    ``keepalive_frame`` overrides the default SSE comment with a route-
    specific protocol event when the client requires one (see
    ``resilient_sse_generator`` for context).
    """
    return StreamingResponse(
        resilient_sse_generator(generator, keepalive_frame=keepalive_frame),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
