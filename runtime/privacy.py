"""Explicit inference trust scope and content-free remote disclosure failures."""

import json
from copy import deepcopy
from enum import Enum

from backend.secret_detection import contains_credential_like_data


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
    """Own and validate semantic data, then serialize that checked snapshot once.

    The header token is deliberately not part of the body. Decoded traversal catches
    JSON escaping, including strings/keys nested inside generation configuration.
    No environment, database, logs, or matched-text diagnostics are involved.
    """

    snapshot = deepcopy(payload)
    if contains_credential_like_data(snapshot, forbidden_text=transport_token):
        raise RemoteDisclosureBlockedError()
    serialized = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Additional exact-token protection only; never run credential patterns over JSON syntax.
    if transport_token and transport_token in serialized:
        raise RemoteDisclosureBlockedError()
    return serialized.encode("utf-8")
