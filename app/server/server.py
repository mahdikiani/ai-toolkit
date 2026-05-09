"""FastAPI application server setup."""

from fastapi import APIRouter
from fastapi_mongo_base.core import app_factory

from apps.language.prompts.routes import router as prompts_router
from apps.language.translate.routes import router as translate_router
from apps.ocr.routes import router as ocr_router
from apps.transcribe.routes import router as transcribe_router
from apps.youtube.routes import router as youtube_router

from . import config

app = app_factory.create_app(settings=config.Settings())
server_router = APIRouter()

for router in [
    prompts_router,
    ocr_router,
    transcribe_router,
    translate_router,
    youtube_router,
]:
    server_router.include_router(router)

app.include_router(server_router, prefix=config.Settings.base_path)
