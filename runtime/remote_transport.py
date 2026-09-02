"""Exact-origin and pre-request DNS/TLS policy for stateless remote inference."""

from __future__ import annotations

import ipaddress
import re
import socket
import ssl
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

Resolver = Callable[[str, int], Sequence[str]]
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


class DNSPolicyError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Remote DNS policy failed")


@dataclass(frozen=True)
class RemoteOrigin:
    scheme: str
    hostname: str
    port: int
    loopback: bool

    @property
    def url(self) -> str:
        host = f"[{self.hostname}]" if ":" in self.hostname else self.hostname
        default_port = 443 if self.scheme == "https" else 80
        suffix = "" if self.port == default_port else f":{self.port}"
        return f"{self.scheme}://{host}{suffix}"


def _parse_origin(value: str) -> RemoteOrigin:
    """Reject ambiguous URLs; V1 supports standard ASCII DNS names and loopback IPs."""

    try:
        if not isinstance(value, str) or not value or any(
            ord(char) <= 32 or ord(char) >= 127 for char in value
        ):
            raise ValueError
        if any(char in value for char in ("?", "#", "\\", "%", "*")):
            raise ValueError
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.netloc.endswith(":")
        ):
            raise ValueError
        authority_pattern = (
            r"\[[0-9a-fA-F:.]+\](?::[0-9]+)?" if parsed.netloc.startswith("[")
            else r"[a-zA-Z0-9.-]+(?::[0-9]+)?"
        )
        if re.fullmatch(authority_pattern, parsed.netloc) is None:
            raise ValueError
        host = parsed.hostname
        if not host:
            raise ValueError
        host = host.lower().removesuffix(".")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if parsed.port == 0 or not 1 <= port <= 65535:
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None:
            if not address.is_loopback:
                raise ValueError  # Public identities must be hostnames, not IP literals.
            host = address.compressed
            loopback = True
        else:
            labels = host.split(".")
            if len(host) > 253 or not all(_DNS_LABEL.fullmatch(label) for label in labels):
                raise ValueError
            loopback = host == "localhost"
            if not loopback and (len(labels) < 2 or not any(c.isalpha() for c in labels[-1])):
                raise ValueError  # No single-label or alternative numeric IP spellings.
        if parsed.scheme != "https" and not loopback:
            raise ValueError
        return RemoteOrigin(parsed.scheme, host, port, loopback)
    except ValueError:
        raise ValueError(
            "Remote inference requires HTTPS with an exact hostname origin; "
            "HTTP is allowed only for loopback development"
        ) from None


def resolve_addresses(hostname: str, port: int) -> Sequence[str]:
    """Resolve A and AAAA answers without silently discarding malformed results."""

    answers = socket.getaddrinfo(
        hostname, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    addresses = []
    for family, socktype, protocol, _canonical, sockaddr in answers:
        if (
            family not in {socket.AF_INET, socket.AF_INET6}
            or socktype != socket.SOCK_STREAM
            or protocol != socket.IPPROTO_TCP
            or not isinstance(sockaddr, tuple)
            or len(sockaddr) != (2 if family == socket.AF_INET else 4)
            or not isinstance(sockaddr[0], str)
            or type(sockaddr[1]) is not int
            or sockaddr[1] != port
        ):
            raise DNSPolicyError()
        address = ipaddress.ip_address(sockaddr[0])
        if address.version != (4 if family == socket.AF_INET else 6):
            raise DNSPolicyError()
        addresses.append(sockaddr[0])
    return addresses


def create_remote_ssl_context() -> ssl.SSLContext:
    # HTTPX's public helper loads its certifi CA roots explicitly when trust_env=False.
    # Unlike ssl.load_default_certs(), this does not honor SSL_CERT_FILE/SSL_CERT_DIR.
    context = httpx.create_ssl_context(verify=True, trust_env=False)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.minimum_version = max(context.minimum_version, ssl.TLSVersion.TLSv1_2)
    return context


@dataclass(frozen=True)
class RemoteTransportPolicy:
    origin: RemoteOrigin
    allowed_origins: frozenset[RemoteOrigin]

    @classmethod
    def from_config(
        cls, endpoint: str, allowed_origins: str | Sequence[str] | None = None,
    ) -> RemoteTransportPolicy:
        origin = _parse_origin(endpoint)
        if isinstance(allowed_origins, str):
            entries = allowed_origins.split(",") if allowed_origins.strip() else []
        else:
            entries = [] if allowed_origins is None else allowed_origins
        allowed = frozenset(_parse_origin(entry.strip()) for entry in entries)
        if not origin.loopback and origin not in allowed:
            raise ValueError(
                "Public remote inference requires an exact approved origin in "
                "AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS"
            )
        return cls(origin, allowed)

    def validate_dns(self, resolver: Resolver) -> None:
        """Recheck all answers per invocation; this is preflight, not connection pinning."""

        try:
            answers = resolver(self.origin.hostname, self.origin.port)
            if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)) or not answers:
                raise DNSPolicyError()
            for text in answers:
                if not isinstance(text, str) or "%" in text:
                    raise DNSPolicyError()
                address = ipaddress.ip_address(text)
                if self.origin.loopback:
                    allowed = address.is_loopback
                else:
                    allowed = address.is_global and not (
                        address.is_private or address.is_loopback or address.is_link_local
                        or address.is_multicast or address.is_unspecified or address.is_reserved
                        or (isinstance(address, ipaddress.IPv6Address) and address.is_site_local)
                    )
                if not allowed:
                    raise DNSPolicyError()
        except Exception:  # noqa: BLE001 - Every resolver failure must prevent transmission.
            # Resolver exceptions may contain private addresses or local configuration.
            raise DNSPolicyError() from None
