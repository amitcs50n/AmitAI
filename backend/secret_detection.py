"""Shared, deliberately narrow credential heuristic; never returns matched text."""

import re
import unicodedata

_CREDENTIAL_PATTERNS = (
    re.compile(
        r"\b(?:password|passcode|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
        r"auth[_ -]?token|client[_ -]?secret)\b\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bauthorization\s*:\s*bearer\s+\S+", re.IGNORECASE),
)


def contains_credential_like_text(text: str) -> bool:
    """Detect known credential forms, not general PII or every possible secret."""

    normalized = unicodedata.normalize("NFKC", text)
    return any(pattern.search(normalized) for pattern in _CREDENTIAL_PATTERNS)
