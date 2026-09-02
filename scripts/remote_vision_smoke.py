"""Explicit synthetic remote smoke. Never accepts a user's image or filesystem path."""

import os
from io import BytesIO
from threading import Event
from uuid import uuid4

from backend.chat_service import GenerationMessage
from backend.vision_grant import RemoteVisionGrant
from runtime.config import load_runtime_config
from runtime.generator import ProviderChatGenerator
from runtime.providers import RemoteInferenceProvider
from scripts.vision_smoke import synthetic_image


def main() -> int:
    provider = None
    grant = RemoteVisionGrant(str(uuid4()), explicit_consent=True)
    try:
        config = load_runtime_config()
        provider = RemoteInferenceProvider(
            os.environ["AMITAI_REMOTE_INFERENCE_URL"],
            os.environ["AMITAI_REMOTE_INFERENCE_TOKEN"],
            str(config.model["name"]),
            allowed_origins=os.environ.get("AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS"),
        )
        generator = ProviderChatGenerator(config, provider=provider)
        with synthetic_image() as image, BytesIO() as buffer:
            image.save(buffer, format="PNG")
            png = buffer.getvalue()
        messages = [GenerationMessage("user", "What text and shapes do you see?")]
        print(generator.generate_vision_response(messages, png, remote_grant=grant).response)
        grant.revoke()
        grant = RemoteVisionGrant(str(uuid4()), explicit_consent=True)
        for item in generator.stream_vision_response(
            messages, png, remote_grant=grant, cancel_event=Event()
        ):
            if hasattr(item, "delta"):
                print(item.delta, end="", flush=True)
        print()
        print(
            generator.generate_response(
                [GenerationMessage("user", "What is the capital of France?")]
            ).response
        )
        return 0
    except Exception:  # noqa: BLE001 - no URL, credentials or private errors in console
        print(
            "Synthetic remote vision smoke failed. Check inference configuration and GPU capacity."
        )
        return 1
    finally:
        grant.revoke()
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
