"""Narrow, bounded, RAM-only multipart protocol. Never use upload spooling."""

import json
import re
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from backend.assets import MAX_ASSET_BYTES, normalize_image

MAX_VISION_METADATA_BYTES = 1024 * 1024
MAX_VISION_BODY_BYTES = MAX_ASSET_BYTES + MAX_VISION_METADATA_BYTES + 4096
_CONTENT_TYPE = re.compile(
    r'multipart/form-data; boundary=(?:"([a-zA-Z0-9_-]{1,70})"|([a-zA-Z0-9_-]{1,70}))'
)


class VisionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=200_000)


class VisionGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    max_new_tokens: int = Field(ge=1, le=512)
    enable_thinking: Literal[False]
    do_sample: bool
    repetition_penalty: float = Field(ge=0.1, le=10)
    seed: int = Field(ge=0, le=2**32 - 1)
    temperature: float | None = Field(default=None, gt=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0, le=100)


class VisionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    request_id: UUID
    messages: list[VisionMessage] = Field(min_length=1, max_length=40)
    generation_config: VisionGenerationConfig


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Invalid vision request")
        result[key] = value
    return result


def decode_metadata(data: bytes) -> VisionMetadata:
    if not data or len(data) > MAX_VISION_METADATA_BYTES:
        raise ValueError("Invalid vision request")
    payload = VisionMetadata.model_validate(json.loads(data, object_pairs_hook=_unique_object))
    if not any(message.role == "user" for message in payload.messages):
        raise ValueError("Invalid vision request")
    return payload


def encode_vision_body(metadata: bytes, image_png: bytes) -> tuple[bytes, str]:
    # The caller MUST guard metadata before passing it here; never scan binary pixels.
    decode_metadata(metadata)
    if not image_png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid vision image")
    image_png = normalize_image(image_png, "image/png").content
    boundary = "amitai_" + uuid4().hex
    delimiter = boundary.encode("ascii")
    body = bytearray()
    for name, media_type, value in (
        (b"metadata", b"application/json", metadata),
        (b"image", b"image/png", image_png),
    ):
        body.extend(b"--" + delimiter + b'\r\nContent-Disposition: form-data; name="' + name)
        body.extend(b'"\r\nContent-Type: ' + media_type + b"\r\n\r\n" + value + b"\r\n")
    body.extend(b"--" + delimiter + b"--\r\n")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def decode_vision_body(body: bytes, content_type: str) -> tuple[VisionMetadata, bytes]:
    """Accept exactly two unnamed-file parts, no extra headers/preamble/epilogue.

    Total input is bounded before this function. Delimiter collisions fail closed.
    This deliberately implements only our small protocol, not arbitrary form uploads.
    """
    match = _CONTENT_TYPE.fullmatch(content_type)
    if not match or len(body) > MAX_VISION_BODY_BYTES:
        raise ValueError("Invalid vision request")
    boundary = (match[1] or match[2]).encode("ascii")
    segments = body.split(b"--" + boundary)
    if len(segments) != 4 or segments[0] or segments[-1] != b"--\r\n":
        raise ValueError("Invalid vision request")
    parts = {}
    for segment in segments[1:-1]:
        if not segment.startswith(b"\r\n") or not segment.endswith(b"\r\n"):
            raise ValueError("Invalid vision request")
        headers, separator, data = segment[2:-2].partition(b"\r\n\r\n")
        if not separator or len(headers) > 2048:
            raise ValueError("Invalid vision request")
        values = {}
        for line in headers.split(b"\r\n"):
            name, separator, value = line.partition(b":")
            name = name.lower()
            if (
                not separator
                or name in values
                or name not in {b"content-type", b"content-disposition"}
            ):
                raise ValueError("Invalid vision request")
            values[name] = value.strip()
        disposition = values.get(b"content-disposition", b"")
        part_match = re.fullmatch(rb'form-data; name="(metadata|image)"', disposition)
        if part_match is None:
            raise ValueError("Invalid vision request")
        name = part_match[1]
        expected_type = b"image/png" if name == b"image" else b"application/json"
        limit = MAX_ASSET_BYTES if name == b"image" else MAX_VISION_METADATA_BYTES
        if (
            name in parts
            or values.get(b"content-type") != expected_type
            or not data
            or len(data) > limit
        ):
            raise ValueError("Invalid vision request")
        parts[name] = data
    if set(parts) != {b"metadata", b"image"}:
        raise ValueError("Invalid vision request")
    return decode_metadata(parts[b"metadata"]), parts[b"image"]
