"""Test fixtures and configuration for ai-toolkit test suite."""

import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from typing import cast

import httpx
import pytest
import pytest_asyncio
from beanie import init_beanie
from fastapi_mongo_base import models as base_mongo_models
from fastapi_mongo_base.utils.basic import get_all_subclasses
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorClientSession
from pymongo.asynchronous.database import AsyncDatabase

from server.config import Settings
from server.server import app as fastapi_app

pytest_plugins = ["tests.fixtures.file_fixtures"]

# Tests must not hit the real finance API from a developer's local .env.
Settings.finance_api_key = None


@pytest.fixture(scope="session")
def mongo_client() -> Generator[AsyncIOMotorClient[dict[str, object]]]:
    """Create a mock MongoDB client for testing."""
    from mongomock_motor import AsyncMongoMockClient

    client: AsyncIOMotorClient[dict[str, object]] = AsyncMongoMockClient()
    yield client


async def init_db(mongo_client: AsyncIOMotorClient[dict[str, object]]) -> None:
    """Initialize Beanie ORM with the test database."""
    database = mongo_client.get_database("test_db")

    original_list_collection_names = database.list_collection_names

    async def list_collection_names(
        session: AsyncIOMotorClientSession | None = None,
        comment: object | None = None,
        **kwargs: object,
    ) -> list[str]:
        """Compatibility wrapper for older mongomock_motor versions."""
        del comment  # mongomock_motor does not accept this kwarg
        filter_value = kwargs.pop("filter", None)
        filter_data = (
            {key: value for key, value in filter_value.items() if isinstance(key, str)}
            if isinstance(filter_value, Mapping)
            else None
        )
        kwargs.pop("authorizedCollections", None)
        kwargs.pop("nameOnly", None)
        call_kwargs: dict[str, object] = {}
        if session is not None:
            call_kwargs["session"] = session
        if filter_data is not None:
            call_kwargs["filter"] = filter_data
        return await original_list_collection_names(**call_kwargs)

    database.list_collection_names: Callable[..., Awaitable[list[str]]] = (
        list_collection_names
    )

    await init_beanie(
        database=cast(AsyncDatabase[dict[str, object]], database),
        document_models=get_all_subclasses(base_mongo_models.BaseEntity),
    )


@pytest_asyncio.fixture(scope="session")
async def db(
    mongo_client: AsyncIOMotorClient[dict[str, object]],
) -> AsyncGenerator[None]:
    """Initialize and cleanup the test database."""
    Settings.config_logger()
    logging.info("Initializing test database")
    await init_db(mongo_client)
    logging.info("Test database initialized")
    yield
    logging.info("Cleaning up test database")


@pytest_asyncio.fixture(scope="session")
async def client(db: None) -> AsyncGenerator[httpx.AsyncClient]:
    """Provide an AsyncClient for the FastAPI app."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fastapi_app),
        base_url=f"https://test.uln.me{Settings.base_path}",
    ) as ac:
        yield ac


@pytest_asyncio.fixture(scope="session")
async def authenticated_client(
    client: httpx.AsyncClient,
) -> AsyncGenerator[httpx.AsyncClient]:
    """Provide an authenticated HTTP client for testing."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fastapi_app),
        base_url=client.base_url,
        headers={"x-api-key": os.getenv("API_KEY") or ""},
    ) as ac:
        yield ac


@pytest.fixture
def mock_user() -> dict:
    """Provide a mock authenticated user."""
    return {
        "user_id": "test_user_123",
        "uid": "test_user_123",
        "tenant_id": "test_tenant_456",
        "email": "test@example.com",
    }
