"""Provide module functionality."""
from fastapi import APIRouter
from fastapi_mongo_base.core import app_factory

from apps.ocr.routes import router as ocr_router
from apps.transcribe.routes import router as transcribe_router
from apps.translate.routes import router as translate_router

from . import config

app = app_factory.create_app(settings=config.Settings())
server_router = APIRouter()

for router in [ocr_router, transcribe_router, translate_router]:
    server_router.include_router(router)

app.include_router(server_router, prefix=config.Settings.base_path)
