"""Provide module functionality."""
import asyncio
from typing import ClassVar

from singleton import Singleton


class Conditions(metaclass=Singleton):
    """Represent Conditions."""

    _conditions: ClassVar[dict[str, asyncio.Condition]] = {}

    def get_condition(self, uid: str) -> asyncio.Condition:
        """Get or create condition for an imagination."""
        if uid not in self._conditions:
            self._conditions[uid] = asyncio.Condition()
        return self._conditions[uid]

    def cleanup_condition(self, uid: str) -> None:
        """Run cleanup condition."""
        self._conditions.pop(uid, None)

    async def release_condition(self, uid: str) -> None:
        """Run release condition."""
        if uid not in self._conditions:
            return

        condition = self.get_condition(uid)
        async with condition:
            condition.notify_all()
        self.cleanup_condition(uid)

    async def wait_condition(self, uid: str) -> None:
        """Run wait condition."""
        condition = self.get_condition(uid)
        async with condition:
            await condition.wait()
        self.cleanup_condition(uid)
