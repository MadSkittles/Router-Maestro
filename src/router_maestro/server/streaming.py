"""Shared SSE streaming utilities with keepalive and error handling."""

import asyncio
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse

from router_maestro.utils import get_logger

logger = get_logger("server.streaming")

SSE_KEEPALIVE_INTERVAL = 15.0  # seconds

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def resilient_sse_generator(
    inner: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Wrap an SSE generator with keepalive comments and error handling.

    Emits ``: keepalive\\n\\n`` every ``SSE_KEEPALIVE_INTERVAL`` seconds when
    no data is produced by the inner generator.  SSE comments are ignored by
    compliant clients but keep bytes flowing on the wire, preventing
    intermediary proxies and client-side socket timeouts from killing the
    connection.

    Exception handling:
    - ``asyncio.CancelledError`` — logged as client disconnect, re-raised
    - ``ConnectionResetError`` — logged as warning
    - Any other ``Exception`` — logged with traceback
    """
    # We use asyncio.wait with FIRST_COMPLETED so that we never cancel the
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
                yield ": keepalive\n\n"

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
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass
        await inner.aclose()


def sse_streaming_response(
    generator: AsyncGenerator[str, None],
) -> StreamingResponse:
    """Create a ``StreamingResponse`` with SSE headers and keepalive wrapper.

    This is the single entry-point that all streaming endpoints should use
    instead of constructing ``StreamingResponse`` directly.
    """
    return StreamingResponse(
        resilient_sse_generator(generator),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
