"""FastAPI application server setup."""

from apps.chat.routes import chat_router
from apps.executions.routes import router as executions_router
from apps.language.transcribe.routes import router as transcribe_router
from fastapi import APIRouter
from fastapi_mongo_base.core import app_factory

from apps.language.prompts.routes import router as prompts_router
from apps.language.translate.routes import router as translate_router
from apps.ocr.routes import router as ocr_router

from . import config

app = app_factory.create_app(settings=config.Settings())
server_router = APIRouter()

for router in [
    prompts_router,
    executions_router,
    ocr_router,
    transcribe_router,
    translate_router,
    chat_router,
]:
    server_router.include_router(router)

app.include_router(server_router, prefix=config.Settings.base_path)
