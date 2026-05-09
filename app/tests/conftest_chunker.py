"""
Minimal conftest for chunker preservation tests.

These tests don't need database access, so we provide a minimal setup.
"""

import pytest


@pytest.fixture(scope="session", autouse=False)
def db() -> None:
    """Override the db fixture to do nothing for chunker tests."""
    yield None
