"""Benchmark-local durable artifacts and conservative JSONL crash recovery.

Newline terminates a committed JSONL frame. Recovery never edits a valid row or
repairs JSON: only an unfinished object at EOF can be discarded, after the caller
has validated every complete row. Local advisory locking excludes concurrent runs.
"""

from __future__ import annotations

import codecs
import errno
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

MAX_MANIFEST_BYTES = 1_048_576
MAX_RESULT_LINE_BYTES = 2_097_152
MAX_RESULTS_BYTES = 67_108_864
MAX_JSON_DEPTH = 64


class RunArtifactError(ValueError):
    """Only constant, content-free diagnostics belong in this exception."""


def flush_durable(handle: BinaryIO) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP}:
            raise


def _sync_directory(directory: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP}:
                raise
        finally:
            os.close(descriptor)


def _regular_file(path: Path, *, missing_ok: bool = False) -> None:
    if path.is_symlink() or (not path.is_file() and not (missing_ok and not path.exists())):
        raise RunArtifactError("Benchmark artifact must be a regular local file")


@contextmanager
def exclusive_run(directory: Path, *, resume: bool) -> Iterator[None]:
    path = directory / ".run.lock"
    _regular_file(path, missing_ok=not resume)
    with path.open("r+b" if resume else "x+b") as handle:
        if not resume:
            handle.write(b"\0")
            flush_durable(handle)
            _sync_directory(directory)
        if path.stat().st_size != 1:
            raise RunArtifactError("Invalid benchmark run lock")
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RunArtifactError("Benchmark run is already in use") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    _regular_file(path, missing_ok=True)
    payload = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            flush_durable(handle)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def durable_append(path: Path, row: dict[str, Any]) -> None:
    _regular_file(path, missing_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(payload) > MAX_RESULT_LINE_BYTES:
        raise RunArtifactError("Benchmark result exceeds the recovery size limit")
    with path.open("a+b") as handle:
        size = handle.tell()
        separator = b""
        if size:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                separator = b"\n"  # Preserve a valid final object lacking only its delimiter.
        if size + len(separator) + len(payload) > MAX_RESULTS_BYTES:
            raise RunArtifactError("Benchmark results exceed the recovery size limit")
        handle.write(separator + payload)
        flush_durable(handle)
    if not size:
        _sync_directory(path.parent)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON number")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def strict_json(raw: bytes) -> Any:
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs,
                          parse_constant=_reject_constant, parse_float=_finite_float)
    pending = [(document, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError("JSON nesting limit exceeded")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    return document


def read_manifest(path: Path) -> dict[str, Any]:
    _regular_file(path)
    with path.open("rb") as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise RunArtifactError("Benchmark manifest exceeds the size limit")
    try:
        manifest = strict_json(raw)
    except (ValueError, RecursionError):
        raise RunArtifactError("Invalid benchmark manifest JSON") from None
    if not isinstance(manifest, dict):
        raise RunArtifactError("Benchmark manifest must be an object")
    return manifest


class _IncompleteJSON(Exception):
    def __init__(self, *, in_string: bool = False) -> None:
        self.in_string = in_string


class _JSONPrefix:
    """Recognize syntactically extendable JSON prefixes, never manufacture JSON.

    A bounded recursive-descent recognizer avoids guesses based on decoder error
    messages. Invalid escapes, duplicate keys, trailing commas and bad tokens are
    rejected even at EOF. Only EOF where more syntax could be valid is incomplete.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def whitespace(self) -> None:
        while self.index < len(self.text) and self.text[self.index] in " \r\n\t":
            self.index += 1

    def peek(self) -> str:
        self.whitespace()
        if self.index == len(self.text):
            raise _IncompleteJSON()
        return self.text[self.index]

    def require(self, expected: str) -> None:
        if self.peek() != expected:
            raise ValueError("Invalid JSON syntax")
        self.index += 1

    def string(self) -> str:
        self.require('"')
        start = self.index - 1
        while self.index < len(self.text):
            character = self.text[self.index]
            self.index += 1
            if character == '"':
                return json.loads(self.text[start:self.index])
            if ord(character) < 32:
                raise ValueError("Invalid string character")
            if character == "\\":
                if self.index == len(self.text):
                    raise _IncompleteJSON()
                escaped = self.text[self.index]
                self.index += 1
                if escaped == "u":
                    for _ in range(4):
                        if self.index == len(self.text):
                            raise _IncompleteJSON()
                        if self.text[self.index] not in "0123456789abcdefABCDEF":
                            raise ValueError("Invalid unicode escape")
                        self.index += 1
                elif escaped not in '"\\/bfnrt':
                    raise ValueError("Invalid escape")
        raise _IncompleteJSON(in_string=True)

    def value(self, depth: int = 0) -> None:
        if depth > MAX_JSON_DEPTH:
            raise ValueError("JSON nesting limit exceeded")
        character = self.peek()
        if character == '"':
            self.string()
        elif character in "{[":
            self.index += 1
            closing = "}" if character == "{" else "]"
            keys: set[str] = set()
            if self.peek() == closing:
                self.index += 1
                return
            while True:
                if character == "{":
                    key = self.string()
                    if key in keys:
                        raise ValueError("Duplicate JSON key")
                    keys.add(key)
                    self.require(":")
                self.value(depth + 1)
                delimiter = self.peek()
                self.index += 1
                if delimiter == closing:
                    return
                if delimiter != ",":
                    raise ValueError("Invalid JSON delimiter")
        elif character in "tfn":
            literal = {"t": "true", "f": "false", "n": "null"}[character]
            remaining = self.text[self.index:]
            if remaining.startswith(literal):
                self.index += len(literal)
            elif literal.startswith(remaining):
                raise _IncompleteJSON()
            else:
                raise ValueError("Invalid JSON literal")
        elif character in "-0123456789":
            start = self.index
            while self.index < len(self.text) and self.text[self.index] in "-+0123456789.eE":
                self.index += 1
            number = self.text[start:self.index]
            if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", number):
                _finite_float(number)
            elif self.index == len(self.text) and re.fullmatch(
                r"-|(?:-?(?:0|[1-9][0-9]*))(?:\.|(?:\.[0-9]+)?[eE][+-]?)", number,
            ):
                raise _IncompleteJSON()
            else:
                raise ValueError("Invalid JSON number")
        else:
            raise ValueError("Invalid JSON token")


def is_torn_object(raw: bytes) -> bool:
    """Only a valid-but-incomplete object prefix, including a torn UTF-8 string."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        text = decoder.decode(raw, final=False)
        recognizer = _JSONPrefix(text)
        if recognizer.peek() != "{":
            return False
        recognizer.value()
    except _IncompleteJSON as exc:
        return bool(text.strip()) and (not decoder.getstate()[0] or exc.in_string)
    except (ValueError, RecursionError):
        return False
    return False  # A complete object, even with trailing garbage, is NOT torn.


def read_results(path: Path, *, expected_count: int) -> tuple[list[dict[str, Any]], int | None]:
    """Read only; caller must validate rows and manifest before applying a repair."""
    _regular_file(path)
    if path.stat().st_size > MAX_RESULTS_BYTES:
        raise RunArtifactError("Benchmark results exceed the recovery size limit")
    rows = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline(MAX_RESULT_LINE_BYTES + 1)
            if not raw:
                return rows, None
            if offset + len(raw) > MAX_RESULTS_BYTES or len(raw) > MAX_RESULT_LINE_BYTES:
                raise RunArtifactError("Benchmark results exceed the recovery size limit")
            if len(rows) >= expected_count:
                raise RunArtifactError("Unexpected extra benchmark result")
            try:
                row = strict_json(raw)
            except (ValueError, RecursionError):
                # Only an unterminated EOF frame can be torn, never a newline-framed row.
                if not raw.endswith(b"\n") and not handle.read(1) and is_torn_object(raw):
                    return rows, offset
                raise RunArtifactError("Invalid benchmark results JSONL") from None
            if not isinstance(row, dict):
                raise RunArtifactError("Benchmark result must be an object")
            rows.append(row)


def truncate_torn_tail(path: Path, offset: int) -> None:
    _regular_file(path)
    with path.open("r+b") as handle:
        if not 0 <= offset < handle.seek(0, os.SEEK_END):
            raise RunArtifactError("Invalid torn-result boundary")
        if offset:
            handle.seek(offset - 1)
            if handle.read(1) != b"\n":
                raise RunArtifactError("Invalid torn-result boundary")
        handle.truncate(offset)
        flush_durable(handle)
