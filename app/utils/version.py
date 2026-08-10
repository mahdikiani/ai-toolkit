"""Installed application version, for startup logs and health checks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

_PACKAGE_NAME = "ai-tools"


def app_version() -> str:
    """Return the installed ai-toolkit package version (pyproject.toml)."""
    try:
        return _pkg_version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return "unknown"
