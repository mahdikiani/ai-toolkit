"""Beanie Artifact entity."""

from typing import ClassVar

from fastapi_mongo_base.models import TenantUserEntity
from pymongo import ASCENDING, IndexModel

from .schemas import ArtifactSchema


class Artifact(ArtifactSchema, TenantUserEntity):
    """Durable Artifact SoR — metadata in Mongo, bytes in Media."""

    class Settings:
        """Collection indexes for common lookup patterns."""

        indexes: ClassVar[list[IndexModel]] = [
            *TenantUserEntity.Settings.indexes,
            IndexModel([("parent_artifact_id", ASCENDING)]),
            IndexModel([("format", ASCENDING), ("source", ASCENDING)]),
        ]
