"""Test fixtures and configuration for ai-toolkit test suite."""

import logging
import os
from collections.abc import AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio
from beanie import init_beanie
from fastapi_mongo_base import models as base_mongo_models
from fastapi_mongo_base.utils.basic import get_all_subclasses

from server.config import Settings
from server.server import app as fastapi_app

pytest_plugins = ["tests.fixtures.file_fixtures"]

# Tests must not hit the real finance API from a developer's local .env.
Settings.finance_api_key = None


@pytest.fixture(scope="session", autouse=True)  # noqa: RUF076
def setup_debugpy() -> None:
    """Set up debugpy for remote debugging if enabled."""
    if os.getenv("DEBUGPY", "False").lower() in ("true", "1", "yes"):
        import debugpy  # noqa: T100

        debugpy.listen(("127.0.0.1", 3020))  # noqa: T100
        logging.info("Waiting for debugpy client")
        debugpy.wait_for_client()  # noqa: T100


@pytest.fixture(scope="session")
def mongo_client() -> Generator[object]:
    """Create a mock MongoDB client for testing."""
    from mongomock_motor import AsyncMongoMockClient

    client: AsyncMongoMockClient = AsyncMongoMockClient()
    yield client


async def init_db(mongo_client: object) -> None:
    """Initialize Beanie ORM with the test database."""
    database = mongo_client.get_database("test_db")  # type: ignore

    original_list_collection_names = database.list_collection_names

    async def list_collection_names(*args: object, **kwargs: object) -> list[str]:
        """Compatibility wrapper for older mongomock_motor versions."""
        kwargs.pop("authorizedCollections", None)
        kwargs.pop("nameOnly", None)
        return await original_list_collection_names(*args, **kwargs)

    database.list_collection_names = list_collection_names

    await init_beanie(
        database=database,  # type: ignore
        document_models=get_all_subclasses(base_mongo_models.BaseEntity),
    )


@pytest_asyncio.fixture(scope="session")
async def db(mongo_client: object) -> AsyncGenerator[None]:
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
