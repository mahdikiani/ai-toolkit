"""Unit tests for conditions utilities."""

import asyncio
import contextlib

import pytest

from utils.conditions import Conditions


@pytest.mark.unit
class TestConditions:
    """Tests for Conditions class."""

    def setup_method(self) -> None:
        """Reset conditions singleton state before each test."""
        # Clear all conditions to ensure test isolation
        Conditions._conditions.clear()

    def test_get_condition_creates_new_condition(self) -> None:
        """get_condition should create a new asyncio.Condition for new UIDs."""
        conditions = Conditions()
        condition = conditions.get_condition("test_uid_1")

        assert isinstance(condition, asyncio.Condition)

    def test_get_condition_returns_same_condition(self) -> None:
        """get_condition should return the same condition for the same UID."""
        conditions = Conditions()
        condition1 = conditions.get_condition("test_uid_2")
        condition2 = conditions.get_condition("test_uid_2")

        assert condition1 is condition2

    def test_cleanup_condition_removes_condition(self) -> None:
        """cleanup_condition should remove the condition for the given UID."""
        conditions = Conditions()
        conditions.get_condition("test_uid_3")
        assert "test_uid_3" in Conditions._conditions

        conditions.cleanup_condition("test_uid_3")

        assert "test_uid_3" not in Conditions._conditions

    def test_cleanup_condition_handles_missing_uid(self) -> None:
        """cleanup_condition should not raise for non-existent UIDs."""
        conditions = Conditions()
        # Should not raise
        conditions.cleanup_condition("nonexistent_uid")

    async def test_release_condition_notifies_waiters(self) -> None:
        """release_condition should notify all waiters."""
        conditions = Conditions()
        uid = "test_uid_release"
        notified = asyncio.Event()

        async def waiter() -> None:
            condition = conditions.get_condition(uid)
            async with condition:
                await condition.wait()
            notified.set()

        # Start waiter in background
        waiter_task = asyncio.create_task(waiter())
        # Give waiter time to start waiting
        await asyncio.sleep(0.01)

        # Release the condition
        await conditions.release_condition(uid)

        # Wait for waiter to be notified
        await asyncio.wait_for(notified.wait(), timeout=1.0)
        assert notified.is_set()

        waiter_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter_task

    async def test_release_condition_cleans_up(self) -> None:
        """release_condition should clean up the condition after notifying."""
        conditions = Conditions()
        uid = "test_uid_cleanup"
        conditions.get_condition(uid)

        # Release without any waiters
        await conditions.release_condition(uid)

        assert uid not in Conditions._conditions

    async def test_release_condition_handles_missing_uid(self) -> None:
        """release_condition should not raise for non-existent UIDs."""
        conditions = Conditions()
        # Should not raise
        await conditions.release_condition("nonexistent_uid")
