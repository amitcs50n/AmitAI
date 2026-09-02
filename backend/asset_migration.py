"""Offline, idempotent image migration before cleanup or request serving."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from functools import partial

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .asset_crypto import MAX_ASSET_BYTES, PNG_SIGNATURE
from .asset_storage import AssetStorage, AssetStorageError, _safe_storage
from .assets import MAX_IMAGE_DIMENSION, MAX_IMAGE_PIXELS, _verify_png_container
from .models import UploadedAsset


def validate_normalized_png(
    data: bytes,
    *,
    expected: tuple[int, str, int, int] | None = None,
) -> None:
    """Verify the legacy canonical container; no re-encoding, metadata change, or logging."""
    if not 0 < len(data) <= MAX_ASSET_BYTES or not data.startswith(PNG_SIGNATURE):
        raise AssetStorageError("Stored image is unavailable")
    if expected is not None:
        size, digest, _width, _height = expected
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise AssetStorageError("Stored image is unavailable")
    _verify_png_container(data)
    # Our normalized writer emits only IHDR, IDAT(s), IEND: 8-bit noninterlaced RGBA.
    if data[8:16] != b"\x00\x00\x00\rIHDR" or data[24:29] != b"\x08\x06\x00\x00\x00":
        raise AssetStorageError("Stored image is unavailable")
    width, height = struct.unpack(">II", data[16:24])
    if (
        not 0 < width <= MAX_IMAGE_DIMENSION
        or not 0 < height <= MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
        or (expected is not None and (width, height) != expected[2:])
    ):
        raise AssetStorageError("Stored image is unavailable")
    offset = 33
    has_pixels = False
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        if kind == b"IDAT":
            has_pixels = True
        elif kind != b"IEND" or not has_pixels:
            raise AssetStorageError("Stored image is unavailable")
        offset += 12 + length


@_safe_storage
def migrate_assets(
    session_factory: sessionmaker[Session],
    storage: AssetStorage,
    *,
    phase_hook: Callable[[str], None] | None = None,
) -> None:
    # Snapshot only the verification fields; no open SQL transaction during file I/O.
    with session_factory() as session:
        records = {
            row.id: (row.byte_size, row.sha256, row.width, row.height)
            for row in session.execute(
                select(
                    UploadedAsset.id,
                    UploadedAsset.byte_size,
                    UploadedAsset.sha256,
                    UploadedAsset.width,
                    UploadedAsset.height,
                )
            )
        }
    # Include complete generated orphans: no plaintext residue while they await TTL cleanup.
    identifiers = set(records)
    identifiers.update(path.stem for path in storage.owned_files() if not path.name.startswith("."))
    for asset_id in sorted(identifiers):
        storage.migrate_asset(
            asset_id,
            partial(validate_normalized_png, expected=records.get(asset_id)),
            phase_hook=phase_hook,
        )
    # A temp never was a committed asset. Safe even if a prior process died mid-write.
    storage.clean_interrupted_temps()
