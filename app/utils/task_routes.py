"""
Combined task router with USSO authentication.

Syntactic sugar that merges ``AbstractTaskRouter`` (background-task CRUD
with ``/start``, ``/webhook``, ``draftable``, and ``blocking`` support)
and ``AbstractTenantUSSORouter`` (USSO JWT/API-key auth, owner + scope
authorisation, tenant-scoped queries) into a single base class so that
concrete task routers can use single inheritance.
"""

from fastapi_mongo_base.routes import AbstractTaskRouter
from fastapi_mongo_base.utils.usso_routes import AbstractTenantUSSORouter


class AbstractTaskUSSORouter(AbstractTaskRouter, AbstractTenantUSSORouter):
    """
    Task router with USSO authentication.

    All task-based apps (OCR, transcribe, webpage, YouTube, translate,
    promptic) should inherit from this class instead of using multiple
    inheritance.

    Inherited method resolution order (C3 linearisation):

    ``AbstractTaskUSSORouter``
    → ``AbstractTaskRouter``  (task lifecycle: ``/start``, ``/webhook``,
       ``draftable``, ``blocking``, ``background_tasks``)
    → ``AbstractTenantUSSORouter``  (tenant-scoped, ``user_id`` ownership)
    → ``AbstractUSSORouterBase``  (auth, ``authorize()``, ``get_user()``)
    → ``AbstractBaseRouter``  (raw CRUD, ``config_routes``, ``config_schemas``)
    """
