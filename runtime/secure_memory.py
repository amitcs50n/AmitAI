"""Database-key interface backed by the shared locked/zeroized secret handle."""

from backend.secure_memory import SecretHandle, SecureMemoryError

__all__ = ["DatabaseKeyHandle", "SecureMemoryError"]


class DatabaseKeyHandle(SecretHandle):
    """Keep the existing SQLCipher key-source API and sanitized representation."""

    __slots__ = ()
