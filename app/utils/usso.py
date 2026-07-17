"""USSO authentication factory for FastAPI dependency injection."""

import os

from usso import APIHeaderConfig, AuthConfig
from usso.integrations.fastapi import USSOAuthentication


def get_usso(raise_exception: bool = False) -> USSOAuthentication:
    """
    Create a configured USSOAuthentication FastAPI dependency.

    Supports two authentication methods:
    - JWT Bearer tokens verified via JWKS endpoint
    - API Key via ``x-api-key`` header verified via the verify endpoint

    Args:
        raise_exception: Whether to raise on auth failure (default False).

    Returns:
        A callable USSOAuthentication instance usable as FastAPI
        ``Depends``.
    """
    usso_base_url = os.getenv("USSO_BASE_URL") or "https://usso.uln.me"
    return USSOAuthentication(
        jwt_config=AuthConfig(
            jwks_url=f"{usso_base_url}/.well-known/jwks.json",
            api_key_header=APIHeaderConfig(
                header_name="x-api-key",
                verify_endpoint=f"{usso_base_url}/api/sso/v1/apikeys/verify",
            ),
        ),
        from_usso_base_url=usso_base_url,
        raise_exception=raise_exception,
    )
