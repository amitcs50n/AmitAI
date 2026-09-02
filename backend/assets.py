"""Explicit upload validation, local lifecycle and future processing boundary."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import warnings
import zlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from threading import Lock
from uuid import uuid4

from PIL import Image, ImageOps
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from .asset_storage import ASSET_ID_PATTERN, MAX_ASSET_BYTES, AssetStorage
from .models import Conversation, Message, UploadedAsset, message_assets, utc_now
from .vision_grant import RemoteVisionGrant, require_remote_vision_grant

MAX_IMAGE_DIMENSION = 8192
MAX_IMAGE_PIXELS = 24_000_000
MAX_ATTACHMENTS = 4
TEMPORARY_TTL = timedelta(hours=24)
IMAGE_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_DECODE_LOCK = Lock()
_PRIVATE_DECODE = ContextVar("private_image_decode", default=False)


class _DecoderLogFilter(logging.Filter):
    def filter(self, _record: logging.LogRecord) -> bool:
        return not _PRIVATE_DECODE.get()


# These are the decoders allowed below, plus their shared EXIF reader. Filters
# affect only the current decode context, not unrelated application logging.
for _name in ("Image", "TiffImagePlugin", "PngImagePlugin", "JpegImagePlugin", "WebPImagePlugin"):
    logging.getLogger(f"PIL.{_name}").addFilter(_DecoderLogFilter())


@contextmanager
def _private_decoder_logging():
    token = _PRIVATE_DECODE.set(True)
    try:
        yield
    finally:
        _PRIVATE_DECODE.reset(token)


VISION_NOT_ENABLED = (
    "Your images are attached locally. Vision, OCR, image editing and image generation "
    "are not enabled yet. I haven't analyzed these images or sent them to an inference provider."
)


class AssetError(ValueError):
    def __init__(self, detail: str = "Asset request is invalid", status_code: int = 422) -> None:
        super().__init__(detail)
        self.status_code = status_code


def validate_asset_ids(ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if len(ids) > MAX_ATTACHMENTS or len(ids) != len(set(ids)):
        raise AssetError()
    if any(ASSET_ID_PATTERN.fullmatch(item) is None for item in ids):
        raise AssetError()
    return tuple(ids)


def sanitized_filename(value: str | None) -> str:
    # Paths/controls/shell syntax are not display metadata. An ASCII leaf name is
    # sufficient for V1; never preserve the original path or normalize it to a path.
    leaf = unicodedata.normalize("NFKC", value or "image").replace("\\", "/").rsplit("/", 1)[-1]
    leaf = re.sub(r"[^A-Za-z0-9 ._-]", "_", leaf).strip(" .")[:120].rstrip(" .")
    return leaf or "image"


@dataclass(frozen=True)
class NormalizedImage:
    content: bytes
    width: int
    height: int


def _verify_png_container(content: bytes) -> None:
    # Pillow permits a missing IEND CRC; V1 rejects truncated containers too.
    offset = 8
    while offset + 12 <= len(content):
        size = int.from_bytes(content[offset : offset + 4], "big")
        end = offset + size + 12
        if end > len(content):
            break
        if zlib.crc32(content[offset + 4 : end - 4]) != int.from_bytes(
            content[end - 4 : end], "big"
        ):
            break
        if content[offset + 4 : offset + 8] == b"IEND":
            if size == 0 and end == len(content):
                return
            break
        offset = end
    raise AssetError("Image is invalid or unsupported")


def normalize_image(content: bytes, content_type: str) -> NormalizedImage:
    if not content or len(content) > MAX_ASSET_BYTES:
        raise AssetError("Image must be between 1 byte and 20 MiB", 413 if content else 422)
    if content_type not in IMAGE_TYPES.values():
        raise AssetError("Supported images are PNG, JPEG and WebP", 415)
    # Bounded decoded pixels and serialized decoding prevent concurrent large
    # allocations. Warning/error bodies can contain untrusted image metadata.
    with _DECODE_LOCK, warnings.catch_warnings(), _private_decoder_logging():
        warnings.simplefilter("error")
        try:
            with Image.open(BytesIO(content), formats=list(IMAGE_TYPES)) as source:
                if IMAGE_TYPES.get(source.format) != content_type:
                    raise AssetError("Image type does not match its contents")
                if source.format == "PNG":
                    _verify_png_container(content)
                width, height = source.size
                if (
                    width > MAX_IMAGE_DIMENSION
                    or height > MAX_IMAGE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise AssetError("Image dimensions exceed the supported limits")
                if getattr(source, "n_frames", 1) != 1:
                    raise AssetError("Animated images are not supported")
                source.verify()
            with Image.open(BytesIO(content), formats=list(IMAGE_TYPES)) as source:
                source.load()  # Full decode, not just a plausible header.
                oriented = ImageOps.exif_transpose(source)
                rgba = oriented.convert("RGBA")
                # A fresh pixels-only image cannot inherit EXIF, ICC, text or XMP.
                clean = Image.new("RGBA", rgba.size)
                clean.paste(rgba)
                output = BytesIO()
                clean.save(output, format="PNG")
                normalized = output.getvalue()
                if len(normalized) > MAX_ASSET_BYTES:
                    raise AssetError("Normalized image exceeds 20 MiB", 413)
                return NormalizedImage(normalized, clean.width, clean.height)
        except AssetError:
            raise
        except Exception:  # noqa: BLE001 - decoder exceptions must not disclose metadata
            raise AssetError("Image is invalid or unsupported") from None


class AssetService:
    def __init__(self, session: Session, storage: AssetStorage) -> None:
        self.session = session
        self.storage = storage

    def get(self, asset_id: str) -> UploadedAsset:
        validate_asset_ids([asset_id])
        asset = self.session.get(UploadedAsset, asset_id, populate_existing=True)
        if asset is None or (
            asset.persistence_mode == "temporary" and asset.created_at <= utc_now() - TEMPORARY_TTL
        ):
            raise AssetError("Asset not found or expired", 404)
        return asset

    def create(
        self,
        image: NormalizedImage,
        *,
        filename: str | None,
        persistence_mode: str = "temporary",
        conversation_id: str | None = None,
    ) -> UploadedAsset:
        if persistence_mode not in {"temporary", "conversation"} or (
            (persistence_mode == "conversation") != (conversation_id is not None)
        ):
            raise AssetError()
        asset_id = str(uuid4())
        written = False
        try:
            # Decode happens before this short transaction. Row + complete file
            # become available together. Crash leftovers are swept after the TTL.
            with self.session.begin():
                if (
                    conversation_id is not None
                    and self.session.get(Conversation, conversation_id) is None
                ):
                    raise AssetError("Conversation not found", 404)
                asset = UploadedAsset(
                    id=asset_id,
                    original_filename=sanitized_filename(filename),
                    content_type="image/png",
                    byte_size=len(image.content),
                    width=image.width,
                    height=image.height,
                    sha256=hashlib.sha256(image.content).hexdigest(),
                    persistence_mode=persistence_mode,
                    conversation_id=conversation_id,
                )
                self.session.add(asset)
                self.session.flush()
                self.storage.write(asset_id, image.content)
                written = True
            return asset
        except Exception:
            self.session.rollback()
            if written:
                self.storage.delete(asset_id)
            raise

    def validate_links(self, ids: tuple[str, ...], conversation_id: str | None) -> None:
        validate_asset_ids(ids)
        for asset_id in ids:
            asset = self.get(asset_id)
            if asset.kind != "image" or (
                asset.conversation_id is not None and asset.conversation_id != conversation_id
            ):
                raise AssetError("Asset belongs to another conversation", 409)

    def attach(self, ids: tuple[str, ...], message: Message) -> None:
        self.validate_links(ids, message.conversation_id)
        for asset_id in ids:
            # Conditional update prevents competing requests from reassigning an
            # upload to another conversation after a stale preparation snapshot.
            changed = self.session.execute(
                update(UploadedAsset)
                .where(
                    UploadedAsset.id == asset_id,
                    (UploadedAsset.conversation_id.is_(None))
                    | (UploadedAsset.conversation_id == message.conversation_id),
                )
                .values(persistence_mode="conversation", conversation_id=message.conversation_id)
            )
            if changed.rowcount != 1:
                raise AssetError("Asset changed; retry", 409)
            self.session.execute(
                message_assets.insert().values(
                    message_id=message.id,
                    asset_id=asset_id,
                )
            )

    def delete(self, asset_id: str) -> None:
        with self.session.begin():
            validate_asset_ids([asset_id])
            asset = self.session.get(UploadedAsset, asset_id)
            if asset is None:
                raise AssetError("Asset not found", 404)
            self.session.delete(asset)
        # Detach via FK cascade before removing content. Failed disk removal leaves
        # inaccessible orphan bytes for the periodic sweep, not a readable asset.
        self.storage.delete(asset_id)

    def cleanup(self, *, now: datetime | None = None) -> int:
        cutoff = (now or utc_now()) - TEMPORARY_TTL
        with self.session.begin():
            expired = list(
                self.session.scalars(
                    select(UploadedAsset.id).where(
                        UploadedAsset.persistence_mode == "temporary",
                        UploadedAsset.created_at <= cutoff,
                    )
                )
            )
            self.session.execute(
                delete(UploadedAsset).where(
                    UploadedAsset.persistence_mode == "temporary",
                    UploadedAsset.created_at <= cutoff,
                )
            )
            active = set(self.session.scalars(select(UploadedAsset.id)))
        for asset_id in expired:
            self.storage.delete(asset_id)
        return len(expired) + self.storage.clean_orphans(active, older_than=cutoff.timestamp())

    def processing_bytes(
        self, asset_id: str, *, remote: bool = False,
        remote_grant: RemoteVisionGrant | None = None,
    ) -> bytes:
        if remote or remote_grant is not None:
            try:
                require_remote_vision_grant(remote_grant, asset_id)
            except PermissionError:
                raise AssetError("Remote image processing is not enabled", 403) from None
        asset = self.get(asset_id)
        content = self.storage.read(asset.id)
        if hashlib.sha256(content).hexdigest() != asset.sha256:
            raise AssetError("Stored image is unavailable", 503)
        return content
