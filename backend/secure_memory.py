"""Best-effort locked and explicitly zeroized secret memory."""

from __future__ import annotations

import ctypes
import mmap
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Self


class SecureMemoryError(RuntimeError):
    """A sanitized secure-memory allocation or use failure."""


class SecretHandle:
    """Own exactly 32 mutable key bytes without revealing them in object text."""

    __slots__ = ("_address", "_buffer", "_closed", "_locked")

    def __init__(self, secret: bytes | bytearray, *, require_lock: bool = True) -> None:
        if len(secret) != 32:
            raise SecureMemoryError("Secret must contain exactly 32 bytes")
        self._buffer = bytearray(secret)
        view = (ctypes.c_ubyte * len(self._buffer)).from_buffer(self._buffer)
        self._address = ctypes.addressof(view)
        self._closed = False
        self._locked = self._lock_memory()
        if require_lock and not self._locked:
            self.close()
            raise SecureMemoryError("Secret memory could not be locked")

    def _lock_memory(self) -> bool:
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.VirtualLock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            kernel32.VirtualLock.restype = ctypes.c_int
            return bool(kernel32.VirtualLock(self._address, len(self._buffer)))

        libc = ctypes.CDLL(None, use_errno=True)
        mlock = getattr(libc, "mlock", None)
        if mlock is None:
            return False
        mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        mlock.restype = ctypes.c_int
        if mlock(self._address, len(self._buffer)) != 0:
            return False
        if sys.platform.startswith("linux"):
            madvise = getattr(libc, "madvise", None)
            madv_dontdump = getattr(mmap, "MADV_DONTDUMP", 16)
            if madvise is not None:
                page_size = mmap.PAGESIZE
                page_start = self._address - (self._address % page_size)
                page_end = (
                    (self._address + len(self._buffer) + page_size - 1) // page_size
                ) * page_size
                madvise(
                    ctypes.c_void_p(page_start),
                    ctypes.c_size_t(page_end - page_start),
                    ctypes.c_int(madv_dontdump),
                )
        return True

    def _unlock_memory(self) -> None:
        if not self._locked:
            return
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.VirtualUnlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            kernel32.VirtualUnlock.restype = ctypes.c_int
            kernel32.VirtualUnlock(self._address, len(self._buffer))
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            munlock = getattr(libc, "munlock", None)
            if munlock is not None:
                munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
                munlock.restype = ctypes.c_int
                munlock(self._address, len(self._buffer))
        self._locked = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise SecureMemoryError("Secret handle is closed")

    @contextmanager
    def temporary_hex(self) -> Iterator[str]:
        self._ensure_open()
        value = self._buffer.hex()
        try:
            yield value
        finally:
            value = ""

    def copy_bytes(self) -> bytes:
        """Return an unavoidable short-lived cryptographic-library input copy."""

        self._ensure_open()
        return bytes(self._buffer)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def locked(self) -> bool:
        return self._locked

    def close(self) -> None:
        if self._closed:
            return
        ctypes.memset(self._address, 0, len(self._buffer))
        self._unlock_memory()
        self._closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"<{type(self).__name__} {state}>"

    __str__ = __repr__

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - destructors must never escape
            return
