"""Chat sessions, threads, messages, and persisted assistant replies."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi_mongo_base.schemas import PaginatedResponse
from fastapi_mongo_base.utils import usso_routes
from usso.integrations.fastapi import USSOAuthentication

from server.config import Settings

from .models import ChatMessage, ChatSession, ChatThread
from .schemas import (
    ChatCompletionResponse,
    ChatMessageCreate,
    ChatMessageSchema,
    ChatSessionCreate,
    ChatSessionSchema,
    ChatSessionUpdate,
    ChatThreadCreate,
    ChatThreadSchema,
)
from .services import (
    complete_assistant_message,
    iter_openrouter_sse_deltas,
    messages_as_openrouter,
    thread_model,
)


class ChatSessionRouter(usso_routes.AbstractTenantUSSORouter):
    """CRUD for sessions plus nested threads and messages."""

    model = ChatSession
    schema = ChatSessionSchema
    resource = "chat_session"

    def __init__(self) -> None:
        """Initialize chat router with authentication and endpoints."""
        super().__init__(
            user_dependency=USSOAuthentication(), prefix="/sessions", tags=["Chat"]
        )

    def config_schemas(self, schema: type) -> None:
        """Config schemas."""
        super().config_schemas(schema)
        self.create_request_schema = ChatSessionCreate
        self.update_request_schema = ChatSessionUpdate

    def config_routes(self, **kwargs: object) -> None:
        """Register nested routes before generic `/{uid}` handlers."""
        self.router.add_api_route(
            "/{session_uid}/threads",
            self.list_session_threads,
            methods=["GET"],
            response_model=PaginatedResponse[ChatThreadSchema],
        )
        self.router.add_api_route(
            "/{session_uid}/threads",
            self.create_thread,
            methods=["POST"],
            response_model=ChatThreadSchema,
            status_code=201,
        )
        self.router.add_api_route(
            "/{session_uid}/threads/{thread_uid}",
            self.retrieve_thread,
            methods=["GET"],
            response_model=ChatThreadSchema,
        )
        self.router.add_api_route(
            "/{session_uid}/threads/{thread_uid}/messages",
            self.list_messages,
            methods=["GET"],
            response_model=PaginatedResponse[ChatMessageSchema],
        )
        self.router.add_api_route(
            "/{session_uid}/threads/{thread_uid}/messages",
            self.post_message,
            methods=["POST"],
        )
        super().config_routes(**kwargs)

    async def create_item(
        self,
        request: Request,
        data: ChatSessionCreate,
    ) -> ChatSession:
        """Create session and auto-create first thread."""
        """Create a new chat session with an initial thread."""
        user = await self.get_user(request)
        await self.authorize(
            action="create",
            user=user,
            filter_data=data.model_dump(exclude_none=True),
        )
        session = await ChatSession.create_item({
            "title": data.title,
            "user_id": self._owner_id_for_create(user),
            "tenant_id": user.tenant_id,
        })
        thread = await ChatThread.create_item({
            "session_uid": session.uid,
            "title": data.initial_thread_title or "Thread 1",
            "chat_model": data.initial_chat_model,
            "user_id": self._owner_id_for_create(user),
            "tenant_id": user.tenant_id,
        })
        session.active_thread_uid = thread.uid
        await session.save()
        return session

    async def list_session_threads(
        self,
        request: Request,
        session_uid: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
    ) -> PaginatedResponse[ChatThreadSchema]:
        """Paginate threads under a session."""
        """List threads within a chat session with pagination."""
        user = await self.get_user(request)
        session = await self.get_item(
            uid=session_uid,
            tenant_id=user.tenant_id,
            user_id=None,
            ignore_user_id=True,
        )
        await self.authorize(
            action="read",
            user=user,
            filter_data=session.model_dump(),
        )
        filters = self.get_list_filter_queries(user=user)
        items, total = await ChatThread.list_total_combined(
            tenant_id=user.tenant_id,
            offset=offset,
            limit=limit,
            session_uid=session_uid,
            **filters,
        )
        return PaginatedResponse(
            items=[ChatThreadSchema.model_validate(i) for i in items],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def create_thread(
        self,
        request: Request,
        session_uid: str,
        data: ChatThreadCreate,
    ) -> ChatThread:
        """Open a new thread (e.g. different model or branch)."""
        """Create a new thread within a chat session."""
        user = await self.get_user(request)
        session = await self.get_item(
            uid=session_uid,
            tenant_id=user.tenant_id,
            user_id=None,
            ignore_user_id=True,
        )
        await self.authorize(
            action="read",
            user=user,
            filter_data=session.model_dump(),
        )
        return await ChatThread.create_item({
            **data.model_dump(exclude_none=True),
            "session_uid": session.uid,
            "user_id": self._owner_id_for_create(user),
            "tenant_id": user.tenant_id,
        })

    async def retrieve_thread(
        self,
        request: Request,
        session_uid: str,
        thread_uid: str,
    ) -> ChatThread:
        """Return one thread if it belongs to the session."""
        """Retrieve a specific thread by its UID."""
        user = await self.get_user(request)
        session = await self.get_item(
            uid=session_uid,
            tenant_id=user.tenant_id,
            user_id=None,
            ignore_user_id=True,
        )
        await self.authorize(
            action="read",
            user=user,
            filter_data=session.model_dump(),
        )
        thread = await ChatThread.get_item(
            uid=thread_uid,
            tenant_id=user.tenant_id,
            user_id=None,
            ignore_user_id=True,
        )
        if thread is None or thread.session_uid != session.uid:
            raise HTTPException(status_code=404, detail="Thread not found")
        await self.authorize(
            action="read",
            user=user,
            filter_data=thread.model_dump(),
        )
        return thread

    async def list_messages(
        self,
        request: Request,
        session_uid: str,
        thread_uid: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=Settings.page_max_limit),
    ) -> PaginatedResponse[ChatMessageSchema]:
        """Paginate messages in a thread."""
        """List messages within a thread with pagination."""
        thread = await self.retrieve_thread(request, session_uid, thread_uid)
        filters = self.get_list_filter_queries(user=await self.get_user(request))
        items, total = await ChatMessage.list_total_combined(
            tenant_id=thread.tenant_id,
            offset=offset,
            limit=limit,
            thread_uid=thread.uid,
            **filters,
        )
        return PaginatedResponse(
            items=[ChatMessageSchema.model_validate(i) for i in items],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def post_message(  # noqa: ANN201
        self,
        request: Request,
        session_uid: str,
        thread_uid: str,
        data: ChatMessageCreate,
    ):  # -> ChatCompletionResponse | StreamingResponse:
        """Append a message; optionally run assistant completion."""
        """Post a user message and optionally generate an assistant reply."""
        user = await self.get_user(request)
        thread = await self.retrieve_thread(request, session_uid, thread_uid)

        user_msg = await ChatMessage.create_item({
            "thread_uid": thread.uid,
            "user_id": self._owner_id_for_create(user),
            "tenant_id": user.tenant_id,
            "role": data.role,
            "content": data.content,
            "reply_to_uid": data.reply_to_uid,
        })

        if not data.generate_reply:
            return ChatCompletionResponse(
                user_message=ChatMessageSchema.model_validate(user_msg),
                assistant_message=None,
            )

        if data.stream:

            async def sse() -> AsyncIterator[str]:
                chunks: list[str] = []
                payload = {
                    "model": thread_model(thread),
                    "messages": await messages_as_openrouter(thread),
                    "temperature": 0.7,
                }
                async for delta in iter_openrouter_sse_deltas(payload):
                    chunks.append(delta)
                    chunk_evt = json.dumps(
                        {"choices": [{"delta": {"content": delta}}]},
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk_evt}\n\n"
                full = "".join(chunks)
                assistant = await ChatMessage.create_item({
                    "thread_uid": thread.uid,
                    "user_id": self._owner_id_for_create(user),
                    "tenant_id": user.tenant_id,
                    "role": "assistant",
                    "content": full.strip(),
                    "completion_extra": {"model": payload["model"], "streamed": True},
                })
                done_evt = json.dumps(
                    {"assistant_message_uid": assistant.uid},
                    ensure_ascii=False,
                )
                yield f"data: {done_evt}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        assistant = await complete_assistant_message(
            thread=thread,
            user_id=self._owner_id_for_create(user),
            tenant_id=user.tenant_id,
        )
        return ChatCompletionResponse(
            user_message=ChatMessageSchema.model_validate(user_msg),
            assistant_message=ChatMessageSchema.model_validate(assistant),
        )


router = APIRouter(prefix="/chat", tags=["Chat"])
router.include_router(ChatSessionRouter().router)
