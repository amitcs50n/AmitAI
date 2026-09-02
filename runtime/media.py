"""Future media contracts. References are uploaded IDs, never filesystem paths.

No media executor is enabled. A future remote implementation must authorize every
referenced asset through AssetService.processing_bytes(remote=True) AND a media
disclosure policy before transport. Upload consent alone is not remote consent.
"""

from dataclasses import dataclass
from typing import Literal

ProcessingScope = Literal["local_only", "remote_allowed"]


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
