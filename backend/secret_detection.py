"""Narrow semantic credential checks. No matched contents leave this module."""

import json
import re
import unicodedata

_BASE_LABELS = (
    "password", "passcode", "api_key", "access_token", "refresh_token", "auth_token",
    "client_secret",
)
# Preserve the previously supported compact spellings (apikey, accesstoken, ...).
_LABELS = frozenset((*_BASE_LABELS, *(label.replace("_", "") for label in _BASE_LABELS)))
_SEPARATORS = re.compile(r"[\s_.-]+")
_LABEL = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")
_STANDALONE = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)
_QUOTED = r'"(?:\\.|[^"\\])*"' + r"|'(?:\\.|[^'\\])*'"
# Consume whole quoted strings before looking for unquoted assignments. A closing
# quote in 'Explain "password:"' must never become an assignment's value.
_TEXT_TOKEN = re.compile(
    rf"(?P<json>[\[{{])|(?P<quoted>{_QUOTED})|"
    r"(?<![\w.-])(?P<label>[a-zA-Z0-9]+(?:[_. \t-]+[a-zA-Z0-9]+)*)"
)
# The unquoted token does not require a trailing assignment separator: consuming
# ordinary prose once avoids repeatedly backtracking over long non-assignment text.
_QUOTED_VALUE = re.compile(_QUOTED)
_ASSIGNMENT = re.compile(r"[ \t]*[:=][ \t]*")
_BARE_VALUE = re.compile(r'''[^\r\n"',;{}\[\]]+''')
_BEARER = re.compile(r"bearer\s+\S+", re.IGNORECASE)


class _JsonPairs(list):
    """Keep duplicate JSON keys visible to the detector instead of overwriting them."""


_JSON = json.JSONDecoder(object_pairs_hook=_JsonPairs)


def _normalize_label(label: str) -> str:
    return _SEPARATORS.sub("_", unicodedata.normalize("NFKC", label).casefold().strip())


def is_credential_like_label(label: str) -> bool:
    """Match a known family, optionally preceded by separator-delimited names."""

    normalized = _normalize_label(label)
    return _LABEL.fullmatch(normalized) is not None and any(
        normalized == base or normalized.endswith("_" + base) for base in _LABELS
    )


def contains_credential_like_pair(label: str, value: str) -> bool:
    """Check an actual decoded label/value pair, not serialization punctuation."""

    normalized_value = unicodedata.normalize("NFKC", value).strip()
    if not normalized_value:
        return False
    if is_credential_like_label(label):
        return True
    normalized_label = _normalize_label(label)
    return (
        normalized_label == "authorization" or normalized_label.endswith("_authorization")
    ) and _BEARER.match(normalized_value) is not None


def _unquote(quoted: str) -> str:
    if quoted.startswith('"'):
        try:
            return json.loads(quoted)
        except ValueError:
            pass  # Non-JSON prose quoting is still inspected, without evaluation.
    return re.sub(r"\\(['\"\\])", r"\1", quoted[1:-1])


def _assignment_value(text: str, start: int) -> str:
    start += len(text[start:]) - len(text[start:].lstrip(" \t"))
    quoted = _QUOTED_VALUE.match(text, start)
    if quoted is not None:
        return _unquote(quoted.group())
    if text[start:start + 1] in ("'", '"'):
        start += 1  # An unterminated quote with actual content is not an empty value.
    bare = _BARE_VALUE.match(text, start)
    return bare.group().strip() if bare is not None else ""


def _scalar_text(value: object) -> str:
    return str(value) if isinstance(value, (str, int, float, bool)) else ""


def contains_credential_like_data(value: object, *, forbidden_text: str | None = None) -> bool:
    """Inspect decoded JSON-compatible data and JSON embedded in actual text.

    The optional exact secret is supplied by the caller, never read from process
    state. Traversal is iterative, including decoded quoting/embedded JSON. Objects
    with sibling key/value fields are checked as structured memory-style records.
    """

    pending = [value]
    seen_containers: dict[int, object] = {}
    while pending:
        item = pending.pop()
        if isinstance(item, (dict, list, tuple)):
            if id(item) in seen_containers:
                continue
            # Keep references so object IDs cannot be reused during embedded JSON decoding.
            # Cyclic configuration still fails JSON serialization, without hanging this scan.
            seen_containers[id(item)] = item
        if isinstance(item, (dict, _JsonPairs)):
            pairs = list(item.items()) if isinstance(item, dict) else item
            record_labels = [v for k, v in pairs if k == "key" and isinstance(v, str)]
            record_values = [v for k, v in pairs if k == "value"]
            if any(contains_credential_like_pair(k, _scalar_text(v))
                   for k in record_labels for v in record_values):
                return True
            for key, child in pairs:
                if isinstance(key, str) and contains_credential_like_pair(key, _scalar_text(child)):
                    return True
                pending.extend((key, child))
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, str):
            if forbidden_text and forbidden_text in item:
                return True
            text = unicodedata.normalize("NFKC", item)
            if any(pattern.search(text) for pattern in _STANDALONE):
                return True
            position = 0
            while match := _TEXT_TOKEN.search(text, position):
                position = match.end()
                if match.group("json"):
                    try:
                        decoded, position = _JSON.raw_decode(text, match.start())
                    except (ValueError, RecursionError):
                        continue  # Still inspect subsequent assignments in malformed JSON.
                    pending.append(decoded)
                elif match.group("quoted"):
                    label = _unquote(match.group())
                    pending.append(label)
                    separator = _ASSIGNMENT.match(text, position)
                    if separator and contains_credential_like_pair(
                        label, _assignment_value(text, separator.end()),
                    ):
                        return True
                else:
                    separator = _ASSIGNMENT.match(text, position)
                    if separator and contains_credential_like_pair(
                        match.group("label"), _assignment_value(text, separator.end()),
                    ):
                        return True
    return False


def contains_credential_like_text(text: str) -> bool:
    """Detect known credential forms, not general PII or every possible secret."""

    return contains_credential_like_data(text)
