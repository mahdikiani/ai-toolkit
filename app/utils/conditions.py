"""Utilities for managing async conditions and synchronization primitives."""

import asyncio
from typing import ClassVar

from singleton import Singleton


class Conditions(metaclass=Singleton):
    """Singleton manager for asyncio conditions keyed by unique identifiers."""

    _conditions: ClassVar[dict[str, asyncio.Condition]] = {}

    def get_condition(self, uid: str) -> asyncio.Condition:
        """Get or create condition for an imagination."""
        if uid not in self._conditions:
            self._conditions[uid] = asyncio.Condition()
        return self._conditions[uid]

    def cleanup_condition(self, uid: str) -> None:
        """
        Remove and clean up the condition for the given identifier.

        Args:
            uid: Unique identifier for the condition.
        """
        self._conditions.pop(uid, None)

    async def release_condition(self, uid: str) -> None:
        """
        Notify all waiters on the condition and clean it up.

        Args:
            uid: Unique identifier for the condition.
        """
        if uid not in self._conditions:
            return

        condition = self.get_condition(uid)
        async with condition:
            condition.notify_all()
        self.cleanup_condition(uid)

    async def wait_condition(self, uid: str) -> None:
        """
        Wait on the condition for the given identifier until notified.

        Args:
            uid: Unique identifier for the condition.
        """
        condition = self.get_condition(uid)
        async with condition:
            await condition.wait()
        self.cleanup_condition(uid)
