"""Explicit inference trust scope and content-free remote disclosure failures."""

import json
from enum import Enum

from backend.secret_detection import contains_credential_like_text


class InferenceExecutionScope(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class RemoteDisclosureBlockedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Remote inference blocked by local privacy policy")


def require_execution_scope(provider: object) -> InferenceExecutionScope:
    scope = getattr(provider, "execution_scope", None)
    if not isinstance(scope, InferenceExecutionScope):
        raise TypeError("Inference provider must declare a valid execution scope")
    return scope


def guarded_request_body(payload: dict[str, object], *, transport_token: str) -> bytes:
    """Scan the exact serialized request plus decoded strings, then send that snapshot.

    The header token is deliberately not part of the body. Decoded traversal catches
    JSON escaping, including strings/keys nested inside generation configuration.
    No environment, database, logs, or matched-text diagnostics are involved.
    """

    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def check(value: object) -> None:
        if isinstance(value, str):
            if transport_token in value or contains_credential_like_text(value):
                raise RemoteDisclosureBlockedError()
        elif isinstance(value, dict):
            for key, item in value.items():
                check(key)
                check(item)
                if isinstance(key, str) and isinstance(item, (str, int, float)):
                    check(f"{key}: {item}")
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(serialized)
    check(json.loads(serialized))
    return serialized.encode("utf-8")
