"""Token usage tracker."""

import time
from datetime import datetime

from router_maestro.stats.storage import StatsStorage, UsageRecord


class UsageTracker:
    """Tracks token usage for requests."""

    _instance: "UsageTracker | None" = None

    def __init__(self) -> None:
        self.storage = StatsStorage()

    @classmethod
    def get_instance(cls) -> "UsageTracker":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool = True,
        latency_ms: int | None = None,
    ) -> None:
        """Record a usage event.

        Args:
            provider: Provider name
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            success: Whether the request was successful
            latency_ms: Latency in milliseconds
        """
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            success=success,
            latency_ms=latency_ms,
        )
        self.storage.record(record)


class RequestTimer:
    """Context manager for timing requests."""

    def __init__(self) -> None:
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self) -> "RequestTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int((self.end_time - self.start_time) * 1000)
