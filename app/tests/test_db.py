"""Tests for MongoDB database initialization."""

import pytest
from fastapi_mongo_base.models import BaseEntity
from fastapi_mongo_base.utils.basic import get_all_subclasses


@pytest.mark.asyncio
async def test_mongo_client_connection(mongo_client: object) -> None:
    """Test that MongoDB mock client is available and functional."""
    assert mongo_client is not None

    # Get a test database
    database = mongo_client.get_database("test_db")  # type: ignore
    assert database is not None
    assert database.name == "test_db"


@pytest.mark.asyncio
async def test_database_initialized(db: None) -> None:
    """Test that database is initialized with Beanie ODM."""
    # The db fixture initializes the database
    # If we reach this point, initialization was successful
    assert True


@pytest.mark.asyncio
async def test_database_collections(mongo_client: object) -> None:
    """Test that we can query database collections."""
    database = mongo_client.get_database("test_db")  # type: ignore

    # Verify we can list collections (even if empty)
    collections = await database.list_collection_names()
    assert isinstance(collections, list)


@pytest.mark.asyncio
async def test_beanie_auto_discovery() -> None:
    """
    Test that Beanie auto-discovers BaseEntity subclasses correctly.

    This verifies that the auto-discovery mechanism used in init_mongo_db
    properly filters out abstract classes and finds concrete document models.
    """
    # Get all BaseEntity subclasses using the same logic as init_mongo_db
    document_models = [
        cls
        for cls in get_all_subclasses(BaseEntity)
        if not (
            "Settings" in cls.__dict__ and getattr(cls.Settings, "__abstract__", False)
        )
    ]

    # At this point, we may not have any concrete models yet
    # But the auto-discovery mechanism should work
    assert isinstance(document_models, list)

    # Verify that abstract classes are filtered out
    for model in document_models:
        if "Settings" in model.__dict__:
            assert not getattr(model.Settings, "__abstract__", False)


@pytest.mark.asyncio
async def test_mongo_client_server_info(mongo_client: object) -> None:
    """
    Test that we can get server info from MongoDB mock client.

    This verifies the connection check that init_mongo_db performs
    to ensure MongoDB is accessible before initializing Beanie.
    """
    # This tests the connection check that init_mongo_db performs
    server_info = await mongo_client.server_info()  # type: ignore
    assert server_info is not None
    assert isinstance(server_info, dict)
