"""SQLite storage for token usage statistics."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from router_maestro.config import STATS_DB_FILE


class UsageRecord(BaseModel):
    """A single token usage record."""

    id: int | None = None
    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    latency_ms: int | None = None


class StatsStorage:
    """SQLite storage for token usage statistics."""

    def __init__(self, db_path: Path = STATS_DB_FILE) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    latency_ms INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage(provider)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(model)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def record(self, record: UsageRecord) -> None:
        """Record a usage event."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO usage (
                    timestamp, provider, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    success, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp.isoformat(),
                    record.provider,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    1 if record.success else 0,
                    record.latency_ms,
                ),
            )
            conn.commit()

    def get_usage_by_day(
        self, days: int = 7, provider: str | None = None, model: str | None = None
    ) -> list[dict]:
        """Get usage aggregated by day.

        Args:
            days: Number of days to look back
            provider: Filter by provider (optional)
            model: Filter by model (optional)

        Returns:
            List of dicts with date, total_tokens, request_count
        """
        with self._get_connection() as conn:
            query = """
                SELECT
                    DATE(timestamp) as date,
                    SUM(total_tokens) as total_tokens,
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    COUNT(*) as request_count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                FROM usage
                WHERE timestamp >= datetime('now', ?)
            """
            params: list = [f"-{days} days"]

            if provider:
                query += " AND provider = ?"
                params.append(provider)
            if model:
                query += " AND model = ?"
                params.append(model)

            query += " GROUP BY DATE(timestamp) ORDER BY date"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_usage_by_hour(
        self, days: int = 7, provider: str | None = None, model: str | None = None
    ) -> list[dict]:
        """Get usage aggregated by hour.

        Args:
            days: Number of days to look back
            provider: Filter by provider (optional)
            model: Filter by model (optional)

        Returns:
            List of dicts with date, hour, total_tokens, request_count
        """
        with self._get_connection() as conn:
            query = """
                SELECT
                    DATE(timestamp) as date,
                    CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                    SUM(total_tokens) as total_tokens,
                    COUNT(*) as request_count
                FROM usage
                WHERE timestamp >= datetime('now', ?)
            """
            params: list = [f"-{days} days"]

            if provider:
                query += " AND provider = ?"
                params.append(provider)
            if model:
                query += " AND model = ?"
                params.append(model)

            query += " GROUP BY DATE(timestamp), hour ORDER BY date, hour"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_usage_by_model(self, days: int = 7) -> list[dict]:
        """Get usage aggregated by model.

        Args:
            days: Number of days to look back

        Returns:
            List of dicts with model, provider, total_tokens, request_count
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    model,
                    provider,
                    SUM(total_tokens) as total_tokens,
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    COUNT(*) as request_count,
                    AVG(latency_ms) as avg_latency_ms
                FROM usage
                WHERE timestamp >= datetime('now', ?)
                GROUP BY model, provider
                ORDER BY total_tokens DESC
                """,
                (f"-{days} days",),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_total_usage(self, days: int = 7) -> dict:
        """Get total usage statistics.

        Args:
            days: Number of days to look back

        Returns:
            Dict with total statistics
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    SUM(total_tokens) as total_tokens,
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    COUNT(*) as request_count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    AVG(latency_ms) as avg_latency_ms
                FROM usage
                WHERE timestamp >= datetime('now', ?)
                """,
                (f"-{days} days",),
            )
            row = cursor.fetchone()
            return dict(row) if row else {}
