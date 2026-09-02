"""Local control-plane authentication without buffering application responses."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

MIN_LOCAL_API_TOKEN_CHARS = 32
_LOCKED_TOKEN = "0" * 64
_PROTECTED_ROOTS = ("/api/chat", "/api/conversations", "/api/memory", "/api/assets", "/api/capabilities")


def environment_flag(
    name: str,
    *,
    environ: Mapping[str, str],
    default: bool = False,
) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"{name} must be '0' or '1'")


def validate_local_api_token(token: str | None) -> str | None:
    if token is None:
        return None
    normalized = token.strip()
    if len(normalized) < MIN_LOCAL_API_TOKEN_CHARS:
        raise ValueError(
            f"Local API token must contain at least {MIN_LOCAL_API_TOKEN_CHARS} characters"
        )
    return normalized


def _bearer_value(authorization: str | None) -> str:
    if authorization is None:
        return ""
    scheme, separator, value = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return value


def verify_local_api_token(
    authorization: str | None,
    expected_token: str | None,
) -> bool:
    """Compare every candidate in constant time, including locked/missing-token mode."""

    candidate = _bearer_value(authorization)
    comparison_token = expected_token if expected_token is not None else _LOCKED_TOKEN
    matches = secrets.compare_digest(candidate, comparison_token)
    return expected_token is not None and matches


def _is_protected_path(path: str) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in _PROTECTED_ROOTS)


class LocalApiAuthMiddleware:
    """Protect private local routes while preserving native streaming ASGI behavior."""

    def __init__(self, app: ASGIApp, *, token: str | None) -> None:
        self.app = app
        self._token = validate_local_api_token(token)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and _is_protected_path(str(scope.get("path", ""))):
            headers: dict[bytes, bytes] = dict(scope.get("headers", []))
            raw_authorization = headers.get(b"authorization")
            authorization = (
                raw_authorization.decode("latin-1")
                if raw_authorization is not None
                else None
            )
            if not verify_local_api_token(authorization, self._token):
                response = JSONResponse(
                    {"detail": "Unauthorized"},
                    status_code=401,
                    headers={
                        "Cache-Control": "no-store",
                        "WWW-Authenticate": "Bearer",
                    },
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def security_state(*, auth_enabled: bool, docs_enabled: bool) -> dict[str, Any]:
    """Expose only non-secret configuration state for diagnostics and tests."""

    return {
        "local_api_auth_enabled": auth_enabled,
        "dev_docs_enabled": docs_enabled,
    }
