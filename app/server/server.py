"""FastAPI application server setup."""

import os

from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mongo_base.core import app_factory

from apps.imagination.routes import router as imagination_router
from apps.language.chat.routes import router as chat_router
from apps.language.promptic.routes import router as promptic_router
from apps.language.prompts.routes import router as prompts_router
from apps.language.translate.routes import router as translate_router
from apps.ocr.convert_routes import router as convert_router
from apps.ocr.routes import router as ocr_router
from apps.ocr.services import resume_stuck_ocr_tasks
from apps.openai_compat.routes import router as openai_router
from apps.texttospeech.routes import router as texttospeech_router
from apps.transcribe.routes import router as transcribe_router
from apps.videogen.routes import router as videogen_router
from apps.voicemorph.routes import router as voicemorph_router
from apps.webpage.routes import router as webpage_router
from apps.websearch.routes import router as websearch_router
from apps.youtube.routes import router as youtube_router

from . import config

app = app_factory.create_app(
    settings=config.Settings(),
    # Custom CORS so https://*.uln.me works via regex (panel, portal, …).
    origins=[],
    init_functions=[resume_stuck_ocr_tasks],
)

_cors_regex = os.getenv("CORS_ORIGIN_REGEX", r"https://.*\.uln\.me").strip() or None
_cors_origins = config.Settings().cors_origins or ["http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

server_router = APIRouter()

for router in [
    prompts_router,
    promptic_router,
    chat_router,
    convert_router,
    imagination_router,
    openai_router,
    ocr_router,
    texttospeech_router,
    transcribe_router,
    translate_router,
    videogen_router,
    voicemorph_router,
    webpage_router,
    websearch_router,
    youtube_router,
]:
    server_router.include_router(router)

app.include_router(server_router, prefix=config.Settings.base_path)
