"""One independent asset key, protected at rest by the application SQLCipher DB."""

from __future__ import annotations

import secrets

from sqlalchemy import Engine

from .asset_crypto import ASSET_FORMAT_VERSION
from .asset_storage import AssetStorage
from .models import utc_now
from .secure_memory import SecretHandle


class AssetKeyError(RuntimeError):
    """An unavailable/inconsistent key must never cause silent replacement."""


def load_asset_key(
    engine: Engine, storage: AssetStorage, *, create_if_missing: bool = True,
) -> SecretHandle:
    """Commit the singleton before file migration; retain only a locked handle.

    This private SQLite/SQLCipher DBAPI path deliberately avoids SQLAlchemy's
    parameter AND debug result-row logging of key bytes. No ORM object or key
    value is added to an application session, DTO, engine URL, or app-state repr.
    DBAPI/cryptography can still create short-lived immutable Python copies.
    """
    connection = None
    handle = None
    rows = []
    material = None
    try:
        connection = engine.raw_connection()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                cursor.execute("SELECT id, format_version, key_material FROM asset_encryption_state")
                rows = cursor.fetchmany(2)
                if not rows:
                    if not create_if_missing:
                        raise ValueError
                    # Includes encrypted orphans and interrupted-write files, not just DB rows.
                    storage.assert_key_creation_safe()
                    material = secrets.token_bytes(32)
                    handle = SecretHandle(material)
                    cursor.execute(
                        "INSERT INTO asset_encryption_state "
                        "(id, format_version, key_material, created_at) VALUES (1, ?, ?, ?)",
                        (ASSET_FORMAT_VERSION, material, utc_now().isoformat()),
                    )
                else:
                    if len(rows) != 1 or rows[0][:2] != (1, ASSET_FORMAT_VERSION):
                        raise ValueError
                    material = rows[0][2]
                    if not isinstance(material, bytes) or len(material) != 32:
                        raise ValueError
                    handle = SecretHandle(material)
                connection.commit()
            finally:
                cursor.close()
        finally:
            material = None
            rows.clear()
            connection.close()  # pool reset rolls back failed creation
        return handle
    except Exception:  # noqa: BLE001 - never expose DBAPI parameters/results
        if handle is not None:
            handle.close()
        raise AssetKeyError("Asset encryption key unavailable") from None
