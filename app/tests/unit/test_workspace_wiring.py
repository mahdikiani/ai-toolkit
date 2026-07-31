"""
Representative create_item tests for workspace_id stamping.

Covers a simple task app (websearch), a chunking/multi-call app
(promptic), and chat (session/thread bootstrap) -- the pattern is
identical across all ~12 wired apps, so these three stand in for the
rest per the workspace billing plan.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestWebSearchWorkspaceStamping:
    """create_item should stamp workspace_id from the requesting user."""

    async def test_stamps_workspace_id_when_present(self) -> None:
        from apps.websearch.routes import WebSearchRouter

        router = WebSearchRouter.__new__(WebSearchRouter)
        user = SimpleNamespace(uid="u1", tenant_id="t1", workspace_id="ws1")
        request = MagicMock()
        data = SimpleNamespace(
            user_id=None,
            model_dump=lambda exclude_none=True: {"query": "q"},
        )
        created_item = MagicMock()
        created_item.start_processing = AsyncMock()

        router.get_user = AsyncMock(return_value=user)
        router.model = MagicMock()
        router.model.create_item = AsyncMock(return_value=created_item)

        with patch(
            "apps.websearch.routes.authorize_create_on_behalf",
            new_callable=AsyncMock,
        ):
            await router.create_item(request, data, background_tasks=MagicMock())

        call_dict = router.model.create_item.await_args.args[0]
        assert call_dict["workspace_id"] == "ws1"

    async def test_workspace_id_none_when_absent(self) -> None:
        from apps.websearch.routes import WebSearchRouter

        router = WebSearchRouter.__new__(WebSearchRouter)
        user = SimpleNamespace(uid="u1", tenant_id="t1", workspace_id=None)
        request = MagicMock()
        data = SimpleNamespace(
            user_id=None,
            model_dump=lambda exclude_none=True: {"query": "q"},
        )
        created_item = MagicMock()
        created_item.start_processing = AsyncMock()

        router.get_user = AsyncMock(return_value=user)
        router.model = MagicMock()
        router.model.create_item = AsyncMock(return_value=created_item)

        with patch(
            "apps.websearch.routes.authorize_create_on_behalf",
            new_callable=AsyncMock,
        ):
            await router.create_item(request, data, background_tasks=MagicMock())

        call_dict = router.model.create_item.await_args.args[0]
        assert call_dict["workspace_id"] is None


class TestPrompticWorkspaceStamping:
    """Promptic's create_item should stamp workspace_id too."""

    async def test_stamps_workspace_id(self) -> None:
        from apps.language.promptic.routes import PrompticRouter

        router = PrompticRouter.__new__(PrompticRouter)
        user = SimpleNamespace(uid="u1", tenant_id="t1", workspace_id="ws2")
        request = MagicMock()
        data = SimpleNamespace(
            user_id=None,
            input_variables={},
            model_dump=lambda exclude_none=True: {},
            model_dump_json=lambda: "{}",
        )
        created_item = MagicMock()
        created_item.save = AsyncMock()

        router.get_user = AsyncMock(return_value=user)
        router.model = MagicMock()
        router.model.create_item = AsyncMock(return_value=created_item)

        with (
            patch(
                "apps.language.promptic.routes.authorize_create_on_behalf",
                new_callable=AsyncMock,
            ),
            patch("apps.language.promptic.routes.services.check_schemas"),
        ):
            await router.create_item(
                request,
                "prompt_x",
                data,
                background_tasks=MagicMock(),
            )

        call_dict = router.model.create_item.await_args.args[0]
        assert call_dict["workspace_id"] == "ws2"


class TestChatWorkspaceStamping:
    """bootstrap_session should stamp workspace_id onto session and thread."""

    async def test_bootstrap_session_stamps_workspace_id(self) -> None:
        from apps.language.chat.services import bootstrap_session

        session = SimpleNamespace(uid="s1", active_thread_uid=None, save=AsyncMock())
        thread = SimpleNamespace(uid="th1")

        with (
            patch(
                "apps.language.chat.services.ChatSession.create_item",
                new_callable=AsyncMock,
                return_value=session,
            ) as create_session,
            patch(
                "apps.language.chat.services.ChatThread.create_item",
                new_callable=AsyncMock,
                return_value=thread,
            ) as create_thread,
        ):
            await bootstrap_session(user_id="u1", workspace_id="ws3")

        assert create_session.await_args.args[0]["workspace_id"] == "ws3"
        assert create_thread.await_args.args[0]["workspace_id"] == "ws3"

    async def test_bootstrap_session_workspace_id_defaults_to_none(self) -> None:
        from apps.language.chat.services import bootstrap_session

        session = SimpleNamespace(uid="s1", active_thread_uid=None, save=AsyncMock())
        thread = SimpleNamespace(uid="th1")

        with (
            patch(
                "apps.language.chat.services.ChatSession.create_item",
                new_callable=AsyncMock,
                return_value=session,
            ) as create_session,
            patch(
                "apps.language.chat.services.ChatThread.create_item",
                new_callable=AsyncMock,
                return_value=thread,
            ),
        ):
            await bootstrap_session(user_id="u1")

        assert create_session.await_args.args[0]["workspace_id"] is None
