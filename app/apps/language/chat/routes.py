"""Chat sessions, threads, messages, and persisted assistant replies."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from fastapi_mongo_base.schemas import BaseEntitySchema, PaginatedResponse
from fastapi_mongo_base.utils import usso_routes
from usso import UserData
from usso.integrations.fastapi import USSOAuthentication

from apps.language.shared.exceptions import ThreadNotFoundError
from server.config import Settings

from .models import ChatMessage, ChatSession, ChatThread
from .schemas import (
    ChatCompletionResponse,
    ChatMessageCreate,
    ChatMessageSchema,
    ChatQuickStartCreate,
    ChatQuickStartResponse,
    ChatQuickThreadCreate,
    ChatQuickThreadResponse,
    ChatSessionCreate,
    ChatSessionSchema,
    ChatSessionUpdate,
    ChatThreadCreate,
    ChatThreadSchema,
)
from .services import (
    bootstrap_session,
    complete_assistant_message,
    iter_billed_reply_stream,
    maybe_apply_session_title_if_ready,
    maybe_apply_suggested_thread_title,
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

    def config_schemas(self, schema: type[BaseEntitySchema], **kwargs: object) -> None:
        """Config schemas."""
        super().config_schemas(schema, **kwargs)
        self.create_request_schema = ChatSessionCreate
        self.update_request_schema = ChatSessionUpdate

    def config_routes(
        self,
        *,
        prefix: str = "",
        list_route: bool = True,
        retrieve_route: bool = True,
        create_route: bool = True,
        update_route: bool = True,
        delete_route: bool = True,
        statistics_route: bool = False,
        mine_route: bool = False,
        **kwargs: object,
    ) -> None:
        """Register nested routes before generic `/{uid}` handlers."""
        self.router.add_api_route(
            "/{session_uid}/messages",
            self.quick_new_thread,
            methods=["POST"],
            status_code=201,
            response_model=None,
        )
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
            response_model=None,
        )
        super().config_routes(
            prefix=prefix,
            list_route=list_route,
            retrieve_route=retrieve_route,
            create_route=create_route,
            update_route=update_route,
            delete_route=delete_route,
            statistics_route=statistics_route,
            mine_route=mine_route,
            **kwargs,
        )

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
        owner_id = self._owner_id_for_create(user)
        session, _thread = await bootstrap_session(
            user_id=owner_id,
            title=data.title,
            thread_title=data.initial_thread_title,
            chat_model=data.initial_chat_model,
            suggest_title=True,
            workspace_id=user.workspace_id,
        )
        return session

    async def update_item(
        self,
        request: Request,
        uid: str,
        data: ChatSessionUpdate,
    ) -> ChatSession:
        """Patch session metadata; validate active_thread_uid belongs to session."""
        user = await self.get_user(request)
        patch = data.model_dump(exclude_unset=True)
        session = await self.get_item(
            uid=uid,
            user_id=None,
            ignore_user_id=True,
        )
        await self.authorize(
            action="update",
            user=user,
            filter_data=session.model_dump(),
        )
        active_uid = patch.get("active_thread_uid")
        if active_uid is not None:
            thread = await ChatThread.get_item(
                uid=active_uid,
                user_id=None,
                ignore_user_id=True,
            )
            if thread is None or thread.session_uid != session.uid:
                raise ThreadNotFoundError()
        return await ChatSession.update_item(session, patch)

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
            "workspace_id": user.workspace_id,
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
            user_id=None,
            ignore_user_id=True,
        )
        if thread is None or thread.session_uid != session.uid:
            raise ThreadNotFoundError()
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

    async def _message_reply(
        self,
        *,
        user: UserData,
        thread: ChatThread,
        data: ChatMessageCreate,
        user_msg: ChatMessage,
        stream_done_extra: dict | None = None,
        after_reply: Callable[[], Awaitable[None]] | None = None,
    ) -> ChatCompletionResponse | StreamingResponse:
        """Generate assistant reply for an already-persisted user message."""
        if not data.generate_reply:
            return ChatCompletionResponse(
                user_message=ChatMessageSchema.model_validate(user_msg),
                assistant_message=None,
            )

        owner_id = self._owner_id_for_create(user)

        if data.stream:

            async def sse() -> AsyncIterator[str]:
                chunks: list[str] = []
                async for delta in iter_billed_reply_stream(
                    thread=thread, user_id=owner_id
                ):
                    chunks.append(delta)
                    chunk_evt = json.dumps(
                        {"choices": [{"delta": {"content": delta}}]},
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk_evt}\n\n"
                full = "".join(chunks)
                assistant = await ChatMessage.create_item({
                    "thread_uid": thread.uid,
                    "user_id": owner_id,
                    "workspace_id": user.workspace_id,
                    "role": "assistant",
                    "content": full.strip(),
                    "completion_extra": {
                        "model": thread_model(thread),
                        "streamed": True,
                    },
                })
                if after_reply is not None:
                    await after_reply()
                done_evt = json.dumps(
                    {
                        "assistant_message_uid": assistant.uid,
                        **(stream_done_extra or {}),
                    },
                    ensure_ascii=False,
                )
                yield f"data: {done_evt}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        assistant = await complete_assistant_message(
            thread=thread,
            user_id=owner_id,
        )
        if after_reply is not None:
            await after_reply()
        return ChatCompletionResponse(
            user_message=ChatMessageSchema.model_validate(user_msg),
            assistant_message=ChatMessageSchema.model_validate(assistant),
        )

    async def quick_start(
        self,
        request: Request,
        data: ChatQuickStartCreate,
    ) -> ChatQuickStartResponse | StreamingResponse:
        """Create session + thread and handle the first message in one call."""
        user = await self.get_user(request)
        await self.authorize(
            action="create",
            user=user,
            filter_data=data.model_dump(exclude_none=True),
        )
        owner_id = self._owner_id_for_create(user)
        session, thread = await bootstrap_session(
            user_id=owner_id,
            title=data.title,
            thread_title=data.thread_title,
            chat_model=data.chat_model,
            suggest_title=data.suggest_title,
            workspace_id=user.workspace_id,
        )
        user_msg = await ChatMessage.create_item({
            "thread_uid": thread.uid,
            "user_id": owner_id,
            "workspace_id": user.workspace_id,
            "role": data.role,
            "content": data.content,
            "reply_to_uid": data.reply_to_uid,
        })

        if not data.generate_reply:
            session = await maybe_apply_session_title_if_ready(
                session=session,
                thread=thread,
                user_id=owner_id,
            )
            return ChatQuickStartResponse(
                session=ChatSessionSchema.model_validate(session),
                thread=ChatThreadSchema.model_validate(thread),
                user_message=ChatMessageSchema.model_validate(user_msg),
                assistant_message=None,
            )

        if data.stream:

            async def sse() -> AsyncIterator[str]:
                chunks: list[str] = []
                async for delta in iter_billed_reply_stream(
                    thread=thread, user_id=owner_id
                ):
                    chunks.append(delta)
                    chunk_evt = json.dumps(
                        {"choices": [{"delta": {"content": delta}}]},
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk_evt}\n\n"
                full = "".join(chunks)
                assistant = await ChatMessage.create_item({
                    "thread_uid": thread.uid,
                    "user_id": owner_id,
                    "workspace_id": user.workspace_id,
                    "role": "assistant",
                    "content": full.strip(),
                    "completion_extra": {
                        "model": thread_model(thread),
                        "streamed": True,
                    },
                })
                session_after = await maybe_apply_session_title_if_ready(
                    session=session,
                    thread=thread,
                    user_id=owner_id,
                )
                done_evt = json.dumps(
                    {
                        "assistant_message_uid": assistant.uid,
                        "session_uid": session_after.uid,
                        "thread_uid": thread.uid,
                        "session_title": session_after.title,
                    },
                    ensure_ascii=False,
                )
                yield f"data: {done_evt}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        assistant = await complete_assistant_message(
            thread=thread,
            user_id=owner_id,
        )
        session = await maybe_apply_session_title_if_ready(
            session=session,
            thread=thread,
            user_id=owner_id,
        )
        return ChatQuickStartResponse(
            session=ChatSessionSchema.model_validate(session),
            thread=ChatThreadSchema.model_validate(thread),
            user_message=ChatMessageSchema.model_validate(user_msg),
            assistant_message=ChatMessageSchema.model_validate(assistant),
        )

    async def quick_new_thread(
        self,
        request: Request,
        session_uid: str,
        data: ChatQuickThreadCreate,
    ) -> ChatQuickThreadResponse | StreamingResponse:
        """Create a new thread in a session and handle its first message."""
        user = await self.get_user(request)
        session = await self.get_item(
            uid=session_uid,
            user_id=None,
            ignore_user_id=True,
        )
        await self.authorize(
            action="read",
            user=user,
            filter_data=session.model_dump(),
        )
        owner_id = self._owner_id_for_create(user)
        thread = await ChatThread.create_item({
            "session_uid": session.uid,
            "title": data.thread_title,
            "chat_model": data.chat_model,
            "user_id": owner_id,
            "workspace_id": user.workspace_id,
        })
        session.active_thread_uid = thread.uid
        await session.save()

        user_msg = await ChatMessage.create_item({
            "thread_uid": thread.uid,
            "user_id": owner_id,
            "workspace_id": user.workspace_id,
            "role": data.role,
            "content": data.content,
            "reply_to_uid": data.reply_to_uid,
        })

        if not data.generate_reply:
            thread = await maybe_apply_suggested_thread_title(
                thread=thread,
                user_id=owner_id,
                user_content=data.content,
                assistant_content=None,
                title=data.thread_title,
                suggest_title=data.suggest_thread_title,
            )
            return ChatQuickThreadResponse(
                thread=ChatThreadSchema.model_validate(thread),
                user_message=ChatMessageSchema.model_validate(user_msg),
                assistant_message=None,
            )

        if data.stream:

            async def sse() -> AsyncIterator[str]:
                chunks: list[str] = []
                async for delta in iter_billed_reply_stream(
                    thread=thread, user_id=owner_id
                ):
                    chunks.append(delta)
                    chunk_evt = json.dumps(
                        {"choices": [{"delta": {"content": delta}}]},
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk_evt}\n\n"
                full = "".join(chunks)
                assistant = await ChatMessage.create_item({
                    "thread_uid": thread.uid,
                    "user_id": owner_id,
                    "workspace_id": user.workspace_id,
                    "role": "assistant",
                    "content": full.strip(),
                    "completion_extra": {
                        "model": thread_model(thread),
                        "streamed": True,
                    },
                })
                thread_after = await maybe_apply_suggested_thread_title(
                    thread=thread,
                    user_id=owner_id,
                    user_content=data.content,
                    assistant_content=full,
                    title=data.thread_title,
                    suggest_title=data.suggest_thread_title,
                    model=thread_model(thread),
                )
                done_evt = json.dumps(
                    {
                        "assistant_message_uid": assistant.uid,
                        "thread_uid": thread_after.uid,
                        "thread_title": thread_after.title,
                    },
                    ensure_ascii=False,
                )
                yield f"data: {done_evt}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        assistant = await complete_assistant_message(
            thread=thread,
            user_id=owner_id,
        )
        thread = await maybe_apply_suggested_thread_title(
            thread=thread,
            user_id=owner_id,
            user_content=data.content,
            assistant_content=assistant.content,
            title=data.thread_title,
            suggest_title=data.suggest_thread_title,
        )
        return ChatQuickThreadResponse(
            thread=ChatThreadSchema.model_validate(thread),
            user_message=ChatMessageSchema.model_validate(user_msg),
            assistant_message=ChatMessageSchema.model_validate(assistant),
        )

    async def post_message(
        self,
        request: Request,
        session_uid: str,
        thread_uid: str,
        data: ChatMessageCreate,
    ) -> ChatCompletionResponse | StreamingResponse:
        """Append a message; optionally run assistant completion."""
        """Post a user message and optionally generate an assistant reply."""
        user = await self.get_user(request)
        thread = await self.retrieve_thread(request, session_uid, thread_uid)
        session = await self.get_item(
            uid=session_uid,
            user_id=None,
            ignore_user_id=True,
        )
        owner_id = self._owner_id_for_create(user)

        user_msg = await ChatMessage.create_item({
            "thread_uid": thread.uid,
            "user_id": owner_id,
            "workspace_id": user.workspace_id,
            "role": data.role,
            "content": data.content,
            "reply_to_uid": data.reply_to_uid,
        })

        async def after_title() -> None:
            nonlocal session
            session = await maybe_apply_session_title_if_ready(
                session=session,
                thread=thread,
                user_id=owner_id,
            )

        if not data.generate_reply:
            session = await maybe_apply_session_title_if_ready(
                session=session,
                thread=thread,
                user_id=owner_id,
            )
            return ChatCompletionResponse(
                user_message=ChatMessageSchema.model_validate(user_msg),
                assistant_message=None,
            )

        return await self._message_reply(
            user=user,
            thread=thread,
            data=data,
            user_msg=user_msg,
            after_reply=after_title,
        )


chat_session_router = ChatSessionRouter()
router = APIRouter(prefix="/chat", tags=["Chat"])
router.add_api_route(
    "/messages",
    chat_session_router.quick_start,
    methods=["POST"],
    status_code=201,
    response_model=None,
)
router.include_router(chat_session_router.router)
