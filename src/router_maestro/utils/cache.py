"""Generic time-based cache with configurable TTL."""

import time
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Generic time-based cache with configurable TTL.

    Args:
        ttl: Time-to-live in seconds for cached values.
    """

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._value: T | None = None
        self._timestamp: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if the cached value is present and not expired."""
        return self._value is not None and (time.time() - self._timestamp) < self._ttl

    def get(self) -> T | None:
        """Return the cached value if valid, otherwise None."""
        if self.is_valid:
            return self._value
        return None

    def set(self, value: T) -> None:
        """Store a value with the current timestamp."""
        self._value = value
        self._timestamp = time.time()

    def clear(self) -> None:
        """Clear the cached value and timestamp."""
        self._value = None
        self._timestamp = 0.0
