"""Private, bounded staging and no-overwrite publication for local backups only."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from .paths import (
    _fsync_directory,
    _write_posix_private_file,
    _write_windows_private_file,
    assert_owner_only,
    ensure_private_directory,
)

CHUNK_BYTES = 1024 * 1024


def checked_path(path: Path, *, absolute: bool = False) -> Path:
    expanded = path.expanduser()
    if absolute and not expanded.is_absolute():
        raise ValueError("An absolute backup path is required")
    expanded = expanded.absolute()
    for item in (expanded, *expanded.parents):
        try:
            details = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or (
            getattr(details, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError("Backup paths must not contain links")
    return expanded.resolve(strict=False)


def regular_file(path: Path, limit: int) -> int:
    checked_path(path)
    details = path.stat()
    if not stat.S_ISREG(details.st_mode) or not 0 < details.st_size <= limit:
        raise ValueError("Backup file is invalid or exceeds V1 limits")
    return details.st_size


def new_private_file(path: Path, content: bytes = b"") -> None:
    checked_path(path)
    ensure_private_directory(path.parent)
    if os.name == "nt":
        _write_windows_private_file(path, content)
    else:
        _write_posix_private_file(path, content)
    assert_owner_only(path, directory=False)


@contextmanager
def private_stage(parent: Path) -> Iterator[Path]:
    parent = checked_path(parent)
    # The parent may be a user-selected backup directory. Only our fresh child
    # and files must be private; never chmod or change ACLs on a user directory.
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / f".amitai-backup-{uuid4().hex}"
    if root.exists():
        raise FileExistsError
    ensure_private_directory(root)
    try:
        yield root
    finally:
        # Only this freshly allocated directory, never a supplied target root.
        if root.exists():
            if checked_path(root) != root:
                raise ValueError("Backup staging path changed")
            assert_owner_only(root, directory=True)
            shutil.rmtree(root)


def copy_and_hash(source: BinaryIO, target: BinaryIO | None, limit: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(min(CHUNK_BYTES, limit - size + 1)):
        size += len(chunk)
        if size > limit:
            raise ValueError("Backup member exceeds V1 limits")
        digest.update(chunk)
        if target is not None:
            target.write(chunk)
    return size, digest.hexdigest()


def file_fingerprint(path: Path, limit: int) -> tuple[int, str]:
    regular_file(path, limit)
    with path.open("rb") as source:
        return copy_and_hash(source, None, limit)


def stage_copy(source: BinaryIO, path: Path, limit: int) -> tuple[int, str]:
    new_private_file(path)
    with path.open("r+b") as target:
        result = copy_and_hash(source, target, limit)
        target.flush()
        os.fsync(target.fileno())
    return result


def install_new(source: Path, target: Path) -> None:
    """Atomic no-clobber publication; staging and target must share a filesystem.

    Unlike POSIX rename/replace, link fails if any target already exists. The
    stage is later removed. No chmod/chown or unverified overwrite is performed.
    """
    checked_path(source)
    checked_path(target)
    assert_owner_only(source, directory=False)
    os.link(source, target)
    _fsync_directory(target.parent)


@contextmanager
def journal_lock(path: Path) -> Iterator[BinaryIO]:
    checked_path(path)
    assert_owner_only(path, directory=False)
    with path.open("r+b") as handle:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield handle
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
