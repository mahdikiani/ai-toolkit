"""
Minimal conftest for chunker preservation tests.

These tests don't need database access, so we provide a minimal setup.
"""

from collections.abc import Generator

import pytest


@pytest.fixture(scope="session", autouse=False)
def db() -> Generator[None]:
    """Override the db fixture to do nothing for chunker tests."""
    yield None
