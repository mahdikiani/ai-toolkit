"""SSRF guards for outbound URL fetches."""

from __future__ import annotations

import ipaddress
import socket
from typing import Self
from urllib.parse import urlparse

from fastapi_mongo_base.core.exceptions import BaseHTTPException

_BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


class UnsafeUrlError(BaseHTTPException):
    """Raised when a download URL is not allowed for SSRF reasons."""

    def __init__(self, detail: str = "URL is not allowed") -> None:
        """Initialize with an English detail message."""
        super().__init__(
            status_code=400,
            error_code="unsafe_url",
            detail=detail,
            message={"en": detail, "fa": "آدرس فایل مجاز نیست"},
        )

    @classmethod
    def unable_to_resolve(cls, host: str) -> Self:
        """Host DNS resolution failed."""
        return cls(detail=f"Unable to resolve host: {host}")

    @classmethod
    def private_or_reserved(cls) -> Self:
        """Reject private, loopback, or reserved resolved addresses."""
        return cls(detail="URL resolves to a private or reserved address")

    @classmethod
    def blocked_host(cls) -> Self:
        """Reject localhost and cloud-metadata hostnames."""
        return cls(detail="Localhost and metadata hosts are blocked")

    @classmethod
    def private_ip(cls) -> Self:
        """Reject literal private or reserved IP addresses."""
        return cls(detail="Private or reserved IP addresses are blocked")

    @classmethod
    def unsupported_scheme(cls, scheme: str) -> Self:
        """URL scheme is not http(s)."""
        return cls(detail=f"Unsupported URL scheme: {scheme or 'missing'}")

    @classmethod
    def missing_host(cls) -> Self:
        """URL has no hostname."""
        return cls(detail="URL host is missing")


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True when *ip* is private, loopback, link-local, or reserved."""
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        return True
    if ip.is_reserved or ip.is_unspecified:
        return True
    return any(ip in network for network in _BLOCKED_NETWORKS)


def _reject_if_blocked(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    error: UnsafeUrlError,
) -> None:
    """Raise *error* when *ip* is blocked."""
    if _is_blocked_ip(ip):
        raise error


def _validate_resolved_addresses(host: str) -> None:
    """Resolve *host* and reject any blocked addresses."""
    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError.unable_to_resolve(host) from exc

    if not addr_infos:
        raise UnsafeUrlError.unable_to_resolve(host)

    for info in addr_infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        try:
            resolved = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        _reject_if_blocked(resolved, UnsafeUrlError.private_or_reserved())


def _validate_hostname(host: str) -> None:
    """Reject localhost names and hosts that resolve to blocked addresses."""
    lowered = host.lower().rstrip(".")
    if lowered in {"localhost", "metadata.google.internal"}:
        raise UnsafeUrlError.blocked_host()

    try:
        literal = ipaddress.ip_address(lowered)
    except ValueError:
        _validate_resolved_addresses(host)
        return

    _reject_if_blocked(literal, UnsafeUrlError.private_ip())


def assert_safe_url(url: str, *, allow_data: bool = True) -> None:
    """Reject non-http(s) schemes and hosts that resolve to private/metadata IPs."""
    if allow_data and url.startswith("data:"):
        return

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise UnsafeUrlError.unsupported_scheme(scheme)

    host = parsed.hostname
    if not host:
        raise UnsafeUrlError.missing_host()

    _validate_hostname(host)
