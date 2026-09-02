"""Synthetic-only real-weight test: python -m scripts.vision_smoke (from repo root).

Requires a suitable CUDA PyTorch/TorchVision environment, e.g. A100 80GB.
Never invoked by the test suite. No image filename or remote API is accepted.
"""

import time

from PIL import Image, ImageDraw, ImageFont

from runtime.config import load_runtime_config
from runtime.media import VisionGenerationRequest
from runtime.model import NativeQwenGenerator


def synthetic_image() -> Image.Image:
    image = Image.new("RGB", (1024, 768), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 70), "AEVON VISION 42", fill="black", font=ImageFont.load_default(size=72))
    draw.rectangle((120, 300, 400, 580), fill="red")
    draw.ellipse((600, 300, 880, 580), fill="blue")
    return image


def main() -> int:
    try:
        config = load_runtime_config()
        start = time.perf_counter()
        engine = NativeQwenGenerator(config.model, int(config.generation["seed"]))
        print(f"Model load: {time.perf_counter() - start:.2f}s")
        with synthetic_image() as image:
            request = VisionGenerationRequest(
                [
                    {"role": "system", "content": config.runtime_system_prompt},
                    {"role": "user", "content": "What text and shapes do you see?"},
                ],
                image,
            )
            start = time.perf_counter()
            result = engine.generate_detailed(request.model_messages(), config.generation)
        print(f"Generation: {time.perf_counter() - start:.2f}s")
        print(result.text)
        if engine.torch.cuda.is_available():
            for index in range(engine.torch.cuda.device_count()):
                cuda = engine.torch.cuda
                print(
                    f"CUDA {index} GiB: allocated={cuda.memory_allocated(index) / 2**30:.2f} "
                    f"reserved={cuda.memory_reserved(index) / 2**30:.2f} "
                    f"max_allocated={cuda.max_memory_allocated(index) / 2**30:.2f}"
                )
        return 0
    except Exception:  # noqa: BLE001 - keep cache paths and credentials out of console errors
        print(
            "Native vision smoke failed; check the pinned runtime compatibility and GPU capacity."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
