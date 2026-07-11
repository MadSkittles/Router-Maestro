"""Helpers for managing asynchronous iterator lifecycles."""

from collections.abc import AsyncIterator
from typing import Any

from router_maestro.utils.logging import get_logger

logger = get_logger("utils.async_iterators")


async def close_async_iterator(iterator: AsyncIterator[Any] | None) -> None:
    """Close an async iterator when it exposes the optional ``aclose`` hook."""
    if iterator is None:
        return
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            logger.warning("Failed to close async iterator", exc_info=True)
