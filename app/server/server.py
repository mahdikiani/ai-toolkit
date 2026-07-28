"""FastAPI application server setup."""

from fastapi import APIRouter
from fastapi_mongo_base.core import app_factory

from apps.language.chat.routes import router as chat_router
from apps.language.promptic.routes import router as promptic_router
from apps.language.prompts.routes import router as prompts_router
from apps.language.translate.routes import router as translate_router
from apps.ocr.convert_routes import router as convert_router
from apps.ocr.routes import router as ocr_router
from apps.openai_compat.routes import router as openai_router
from apps.transcribe.routes import router as transcribe_router
from apps.webpage.routes import router as webpage_router
from apps.youtube.routes import router as youtube_router

from . import config

app = app_factory.create_app(settings=config.Settings())
server_router = APIRouter()

for router in [
    prompts_router,
    promptic_router,
    chat_router,
    convert_router,
    openai_router,
    ocr_router,
    transcribe_router,
    translate_router,
    webpage_router,
    youtube_router,
]:
    server_router.include_router(router)

app.include_router(server_router, prefix=config.Settings.base_path)
