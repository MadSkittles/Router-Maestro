"""Tests for stats module."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from router_maestro.stats.storage import StatsStorage, UsageRecord


class TestStatsStorage:
    """Tests for StatsStorage."""

    def test_init_creates_db(self):
        """Test that initialization creates the database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stats.db"
            storage = StatsStorage(db_path)

            assert db_path.exists()

    def test_record_and_retrieve(self):
        """Test recording and retrieving usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stats.db"
            storage = StatsStorage(db_path)

            record = UsageRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                success=True,
                latency_ms=500,
            )
            storage.record(record)

            # Check total usage
            total = storage.get_total_usage(days=1)
            assert total["total_tokens"] == 150
            assert total["request_count"] == 1

    def test_usage_by_model(self):
        """Test usage aggregation by model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stats.db"
            storage = StatsStorage(db_path)

            # Record usage for multiple models
            for model in ["gpt-4o", "gpt-4o", "gpt-3.5-turbo"]:
                storage.record(UsageRecord(
                    timestamp=datetime.now(),
                    provider="openai",
                    model=model,
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    success=True,
                ))

            by_model = storage.get_usage_by_model(days=1)

            assert len(by_model) == 2
            gpt4o = next(m for m in by_model if m["model"] == "gpt-4o")
            assert gpt4o["request_count"] == 2

    def test_empty_database(self):
        """Test querying empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stats.db"
            storage = StatsStorage(db_path)

            total = storage.get_total_usage(days=7)
            # Should return dict with None values
            assert total.get("total_tokens") is None or total.get("request_count") == 0
