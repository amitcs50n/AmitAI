"""Small, bounded V1 asset framing; all encryption uses the library AEAD API."""

from __future__ import annotations

import secrets
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAX_ASSET_BYTES = 20 * 1024 * 1024
ASSET_MAGIC = b"AMITASST"
ASSET_FORMAT_VERSION = 1
ASSET_HEADER = ASSET_MAGIC + bytes([ASSET_FORMAT_VERSION])
ASSET_NONCE_BYTES = 12
ASSET_TAG_BYTES = 16
ASSET_OVERHEAD = len(ASSET_HEADER) + ASSET_NONCE_BYTES + ASSET_TAG_BYTES
MAX_ASSET_CIPHERTEXT_BYTES = MAX_ASSET_BYTES + ASSET_OVERHEAD
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class AssetCryptoError(RuntimeError):
    """Public-safe failure; never include crypto diagnostics or supplied input."""


def _aad(asset_id: str) -> bytes:
    identifier = UUID(asset_id)
    if str(identifier) != asset_id:
        raise ValueError
    # Stable domain + exact header/version + binary UUID. No paths/DB namespace.
    return b"amitai:image-asset\x00" + ASSET_HEADER + identifier.bytes


def encrypt_asset(asset_id: str, plaintext: bytes, key: bytes | bytearray) -> bytes:
    try:
        if len(key) != 32 or not 0 < len(plaintext) <= MAX_ASSET_BYTES:
            raise ValueError
        nonce = secrets.token_bytes(ASSET_NONCE_BYTES)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, _aad(asset_id))
        return ASSET_HEADER + nonce + ciphertext
    except Exception:  # noqa: BLE001 - sanitize library and input failures
        raise AssetCryptoError("Stored image is unavailable") from None


def decrypt_asset(asset_id: str, envelope: bytes, key: bytes | bytearray) -> bytes:
    try:
        if (
            len(key) != 32
            or not ASSET_OVERHEAD < len(envelope) <= MAX_ASSET_CIPHERTEXT_BYTES
            or not envelope.startswith(ASSET_HEADER)
        ):
            raise ValueError
        nonce_end = len(ASSET_HEADER) + ASSET_NONCE_BYTES
        plaintext = AESGCM(key).decrypt(
            envelope[len(ASSET_HEADER) : nonce_end],
            envelope[nonce_end:],
            _aad(asset_id),
        )
        if not 0 < len(plaintext) <= MAX_ASSET_BYTES:
            raise ValueError
        return plaintext
    except Exception:  # noqa: BLE001 - no InvalidTag, nonce, key, or parser details
        raise AssetCryptoError("Stored image is unavailable") from None
