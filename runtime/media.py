"""Request-local media contracts. No paths, storage handles, or transport credentials.

Remote vision requires a current-request RemoteVisionGrant before asset decryption
and controlled transport. Upload consent alone is not remote consent.
"""

import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from typing import Literal

from PIL import Image

from backend.assets import (
    MAX_ASSET_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    _private_decoder_logging,
)

ProcessingScope = Literal["local_only", "remote_allowed"]
MIN_VISION_PIXELS = 65_536
MAX_VISION_PIXELS = 1_048_576
MAX_VISION_ASPECT_RATIO = 200


@dataclass(frozen=True)
class VisionGenerationRequest:
    """Already-compiled text plus one borrowed image, valid only during this request."""

    messages: Sequence[Mapping[str, str]] = field(repr=False)
    image: Image.Image = field(repr=False)
    content_type: Literal["image/png"] = "image/png"

    def model_messages(self) -> list[dict[str, object]]:
        if self.content_type != "image/png" or not isinstance(self.image, Image.Image):
            raise ValueError("Invalid local vision input")
        messages: list[dict[str, object]] = [dict(message) for message in self.messages]
        # Tool followups may follow the current user; historical images are never present.
        current = next((m for m in reversed(messages) if m["role"] == "user"), None)
        if current is None or not isinstance(current["content"], str):
            raise ValueError("Invalid local vision input")
        current["content"] = [
            {"type": "image", "image": self.image},
            {"type": "text", "text": current["content"]},
        ]
        return messages


@contextmanager
def decoded_vision_image(png: bytes) -> Iterator[Image.Image]:
    """Decode authenticated canonical PNG in RAM, with bounded Pillow allocation."""

    image = None
    try:
        if not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid local vision input")
        if len(png) > MAX_ASSET_BYTES:
            raise ValueError("Invalid local vision input")
        with _private_decoder_logging(), warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with BytesIO(png) as source, Image.open(source) as decoded:
                width, height = decoded.size
                if (
                    decoded.format != "PNG"
                    or getattr(decoded, "n_frames", 1) != 1
                    or max(width, height) > MAX_IMAGE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS
                    or max(width, height) / min(width, height) > MAX_VISION_ASPECT_RATIO
                ):
                    raise ValueError("Invalid local vision input")
                decoded.load()
                image = decoded.convert("RGB")
                image.info.clear()
                if width * height > MAX_VISION_PIXELS:
                    scale = math.sqrt(MAX_VISION_PIXELS / (width * height))
                    resized = image.resize(
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                    image.close()
                    image = resized
        yield image
    finally:
        if image is not None:
            image.close()


@dataclass(frozen=True)
class VisionAnalysisRequest:
    asset_ids: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class ImageEditRequest:
    asset_id: str
    prompt: str


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    reference_asset_ids: tuple[str, ...] = ()
