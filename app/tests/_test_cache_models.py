"""Tests for cache models."""

from datetime import datetime, timedelta

import pytest
from apps.cache.models import CacheEntry


@pytest.mark.no_db
class TestCacheEntry:
    """Test suite for CacheEntry model (no database required)."""

    def test_create_with_ttl_sets_correct_expiration(self) -> None:
        """Test that create_with_ttl calculates expiration correctly."""
        ttl_seconds = 3600
        before_creation = datetime.utcnow()

        cache_entry = CacheEntry.create_with_ttl(
            idempotency_key="test_key_123",
            result="Test LLM response",
            template_name="test_template",
            input_variables_hash="abc123hash",
            ttl_seconds=ttl_seconds,
        )

        after_creation = datetime.utcnow()

        # Verify all fields are set correctly
        assert cache_entry.idempotency_key == "test_key_123"
        assert cache_entry.result == "Test LLM response"
        assert cache_entry.template_name == "test_template"
        assert cache_entry.input_variables_hash == "abc123hash"

        # Verify created_at is within reasonable bounds (inherited from BaseEntity)
        assert before_creation <= cache_entry.created_at <= after_creation

        # Verify expires_at is created_at + ttl_seconds
        expected_expiration = cache_entry.created_at + timedelta(seconds=ttl_seconds)
        assert cache_entry.expires_at == expected_expiration

    def test_is_expired_returns_false_for_fresh_entry(self) -> None:
        """Test that is_expired returns False for non-expired entries."""
        # Create entry that expires in 1 hour
        cache_entry = CacheEntry.create_with_ttl(
            idempotency_key="test_key_456",
            result="Fresh result",
            template_name="test_template",
            input_variables_hash="def456hash",
            ttl_seconds=3600,
        )

        assert cache_entry.is_expired() is False

    def test_is_expired_returns_true_for_expired_entry(self) -> None:
        """Test that is_expired returns True for expired entries."""
        # Create entry with negative TTL (already expired)
        cache_entry = CacheEntry.create_with_ttl(
            idempotency_key="test_key_789",
            result="Expired result",
            template_name="test_template",
            input_variables_hash="ghi789hash",
            ttl_seconds=-1,  # Expires 1 second in the past
        )

        assert cache_entry.is_expired() is True

    def test_is_expired_boundary_condition(self) -> None:
        """Test is_expired at the exact expiration moment."""
        # Create entry that expires in 0 seconds (expires immediately)
        cache_entry = CacheEntry.create_with_ttl(
            idempotency_key="test_key_boundary",
            result="Boundary result",
            template_name="test_template",
            input_variables_hash="boundary_hash",
            ttl_seconds=0,
        )

        # Should be expired (current time > expires_at)
        assert cache_entry.is_expired() is True

    def test_create_with_ttl_different_ttl_values(self) -> None:
        """Test create_with_ttl with various TTL values."""
        test_cases = [
            (60, "1 minute"),
            (3600, "1 hour"),
            (86400, "1 day"),
            (604800, "1 week"),
        ]

        for ttl_seconds, description in test_cases:
            cache_entry = CacheEntry.create_with_ttl(
                idempotency_key=f"test_key_{description}",
                result=f"Result for {description}",
                template_name="test_template",
                input_variables_hash=f"hash_{description}",
                ttl_seconds=ttl_seconds,
            )

            expected_expiration = cache_entry.created_at + timedelta(
                seconds=ttl_seconds
            )
            assert cache_entry.expires_at == expected_expiration
            assert cache_entry.is_expired() is False

    def test_cache_entry_settings(self) -> None:
        """Test that CacheEntry has correct Settings configuration."""
        assert CacheEntry.Settings.name == "cache_entries"
        assert len(CacheEntry.Settings.indexes) == 3

        # Verify index configurations - extract field names from IndexModel
        index_fields = []
        for idx in CacheEntry.Settings.indexes:
            # IndexModel stores keys as list of tuples: [("field_name", direction)]
            keys = idx.document["key"]
            index_fields.append(tuple(keys))

        # Check that required indexes exist
        assert any("idempotency_key" in str(idx) for idx in index_fields)
        assert any("expires_at" in str(idx) for idx in index_fields)
        assert any("template_name" in str(idx) for idx in index_fields)

        # Verify unique constraint on idempotency_key index
        idempotency_index = next(
            idx
            for idx in CacheEntry.Settings.indexes
            if "idempotency_key" in str(idx.document["key"])
        )
        assert idempotency_index.document.get("unique") is True

    def test_cache_entry_inherits_from_base_entity(self) -> None:
        """Test that CacheEntry inherits standard fields from BaseEntity."""
        cache_entry = CacheEntry.create_with_ttl(
            idempotency_key="test_key_base",
            result="Test result",
            template_name="test_template",
            input_variables_hash="test_hash",
            ttl_seconds=3600,
        )

        # Verify BaseEntity fields exist
        assert hasattr(cache_entry, "uid")
        assert hasattr(cache_entry, "created_at")
        assert hasattr(cache_entry, "updated_at")

        # Verify uid is set (BaseEntity auto-generates it)
        assert cache_entry.uid is not None


@pytest.mark.asyncio
class TestCacheEntryDatabase:
    """Integration tests for CacheEntry with database operations."""

    async def test_cache_entry_can_be_saved_and_retrieved(
        self, mongo_client: object
    ) -> None:
        """Test that CacheEntry can be persisted to database."""
        # This test will be implemented after the model is created
        # and database initialization includes CacheEntry
        pytest.skip("Database integration test - requires full setup")

    async def test_idempotency_key_uniqueness(self, mongo_client: object) -> None:
        """Test that duplicate idempotency_key raises error."""
        # This test will be implemented after the model is created
        pytest.skip("Database integration test - requires full setup")
