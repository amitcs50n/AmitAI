"""Synthetic-only real-weight test: python -m scripts.vision_smoke (from repo root).

Requires a suitable CUDA PyTorch/TorchVision environment, e.g. A100 80GB.
Never invoked by the test suite. No image filename or remote API is accepted.
Unlike production routes, this explicitly invoked developer script prints full
tracebacks for its synthetic-only generation calls. Do not add private inputs.
"""

import time
import traceback
from threading import Event

from PIL import Image, ImageDraw, ImageFont

from evaluation.hf_backend import GenerationOutput
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
    stage = "NATIVE LOAD"
    engine = None
    try:
        config = load_runtime_config()
        start = time.perf_counter()
        engine = NativeQwenGenerator(config.model, int(config.generation["seed"]))
        print(f"Model load: {time.perf_counter() - start:.2f}s")
        print(f"Runtime versions: {engine.dependency_versions}")
        prepare = engine._prepare_generation

        def measured_prepare(messages, generation_config):
            inputs, kwargs, prompt_tokens = prepare(messages, generation_config)
            if "image_grid_thw" in inputs:
                _, height, width = inputs["image_grid_thw"][0].tolist()
                print(f"Processed image dimensions: {width * 16} x {height * 16}")
            shapes = {key: tuple(value.shape) for key, value in inputs.items()}
            print(f"Input tensor shapes: {shapes}")
            print(f"Prompt tokens: {prompt_tokens}")
            return inputs, kwargs, prompt_tokens

        engine._prepare_generation = measured_prepare
        with synthetic_image() as image:
            request = VisionGenerationRequest(
                [
                    {"role": "system", "content": config.runtime_system_prompt},
                    {"role": "user", "content": "What text and shapes do you see?"},
                ],
                image,
            )
            stage = "NATIVE VISION NONSTREAM"
            start = time.perf_counter()
            result = engine.generate_detailed(request.model_messages(), config.generation)
            print(f"{stage}: PASS ({time.perf_counter() - start:.2f}s)")
            print(f"Vision output tokens: {result.output_tokens}")
            print(result.text)

            stage = "NATIVE VISION STREAM"
            start = time.perf_counter()
            chunks = []
            final = None
            signal = Event()
            stream = engine.generate_detailed_stream(
                request.model_messages(), config.generation, cancel_event=signal
            )
            try:
                for item in stream:
                    if final is not None:
                        raise ValueError("Native stream emitted after terminal output")
                    if isinstance(item, str) and item:
                        chunks.append(item)
                        print(item, end="", flush=True)
                    elif isinstance(item, GenerationOutput):
                        final = item
                    else:
                        raise ValueError("Native stream emitted an invalid item")
                if final is None or "".join(chunks) != final.text:
                    raise ValueError("Native stream did not reconstruct terminal output")
            finally:
                signal.set()
                stream.close()
            print(f"\n{stage}: PASS ({time.perf_counter() - start:.2f}s)")
            print(f"Stream reconstruction: PASS ({final.output_tokens} output tokens)")
        stage = "NATIVE TEXT"
        start = time.perf_counter()
        followup = engine.generate_detailed([
            {"role": "system", "content": config.runtime_system_prompt},
            {"role": "user", "content": "What is the capital of France?"},
        ], config.generation)
        print(f"Same-model text followup: {time.perf_counter() - start:.2f}s, {followup.output_tokens} tokens")
        print(followup.text)
        print(f"{stage}: PASS")
        return 0
    except Exception:  # noqa: BLE001 - synthetic developer diagnostic, not a production logger
        print(f"\n{stage}: FAIL", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if engine is not None and engine.torch.cuda.is_available():
            for index in range(engine.torch.cuda.device_count()):
                cuda = engine.torch.cuda
                print(
                    f"CUDA {index} GiB: allocated={cuda.memory_allocated(index) / 2**30:.2f} "
                    f"reserved={cuda.memory_reserved(index) / 2**30:.2f} "
                    f"max_allocated={cuda.max_memory_allocated(index) / 2**30:.2f} "
                    f"max_reserved={cuda.max_memory_reserved(index) / 2**30:.2f}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
