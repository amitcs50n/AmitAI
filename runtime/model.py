"""One native Qwen model for text and images; no weights load at module import."""

from __future__ import annotations

import logging
from typing import Any

from evaluation.hf_backend import TransformersGenerator

from .config import EXPECTED_MODEL_NAME, EXPECTED_MODEL_REVISION
from .media import MAX_VISION_PIXELS, MIN_VISION_PIXELS

LOGGER = logging.getLogger(__name__)
VISION_PATCH_SIZE = 16
VISION_MERGE_SIZE = 2

# The pinned checkpoint's tokenizer AND processor templates render content as a
# string, despite its multimodal weights. Add only Qwen's official image rendering
# branch; never replace the text template or introduce another checkpoint/template.
# Reference: https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/chat_template.jinja
_CONTENT_SLOT = "{{ message.content }}"
_VISION_CONTENT = (
    "{% if message.content is string %}{{ message.content }}"
    "{% else %}{% for item in message.content %}"
    "{% if item.type == 'image' %}"
    "{{ vision_start_token }}{{ image_token }}{{ vision_end_token }}"
    "{% elif item.type == 'text' %}{{ item.text }}"
    "{% else %}{{ raise_exception('Unsupported vision content') }}{% endif %}"
    "{% endfor %}{% endif %}"
)


def vision_chat_template(text_template: str) -> str:
    if not isinstance(text_template, str) or text_template.count(_CONTENT_SLOT) != 3:
        raise ValueError("Incompatible native vision chat template")
    return text_template.replace(_CONTENT_SLOT, _VISION_CONTENT)


def _execution_device(module: Any) -> Any:
    """Accelerate hooks are authoritative for offloaded/sharded modules, not model.device."""

    hook_device = getattr(getattr(module, "_hf_hook", None), "execution_device", None)
    if hook_device is not None:
        return hook_device
    device = next(module.parameters()).device
    if str(device) == "meta":
        raise ValueError("Native model execution device is unavailable")
    return device


class NativeQwenGenerator(TransformersGenerator):
    """Reuse the proven generation/stream worker with a shared multimodal loader."""

    supports_vision = True

    def __init__(self, model_config: dict[str, Any], seed: int) -> None:
        # Do not invoke the text-only base constructor: it would allocate a second model.
        if (
            model_config.get("name") != EXPECTED_MODEL_NAME
            or model_config.get("revision") != EXPECTED_MODEL_REVISION
            or model_config.get("dtype") != "bfloat16"
            or model_config.get("device_map") != "auto"
            or model_config.get("trust_remote_code") is not False
            or model_config.get("load_in_4bit") is not False
        ):
            raise ValueError("Native runtime requires the pinned BF16 model configuration")
        try:
            import torch
            import transformers
            from transformers import (
                AutoConfig,
                AutoProcessor,
                Qwen3_5ForConditionalGeneration,
                StoppingCriteria,
                StoppingCriteriaList,
                TextIteratorStreamer,
            )

            common = {"revision": EXPECTED_MODEL_REVISION, "trust_remote_code": False}
            config = AutoConfig.from_pretrained(EXPECTED_MODEL_NAME, **common)
            self._validate_config(config)
            self.processor = AutoProcessor.from_pretrained(
                EXPECTED_MODEL_NAME,
                **common,
                use_fast=False,
            )
            self.tokenizer = self.processor.tokenizer
            self._validate_processor(config)
            self.vision_template = vision_chat_template(self.processor.chat_template)
            # Both aliases are set: Transformers 5.2 supports legacy min/max_pixels
            # as well as size. The per-call cap below prevents accidental overrides.
            image_processor = self.processor.image_processor
            image_processor.size = {
                "shortest_edge": MIN_VISION_PIXELS,
                "longest_edge": MAX_VISION_PIXELS,
            }
            image_processor.min_pixels = MIN_VISION_PIXELS
            image_processor.max_pixels = MAX_VISION_PIXELS
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
                EXPECTED_MODEL_NAME,
                **common,
                config=config,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                output_loading_info=True,
            )
            # Full checkpoint load, not a permissive language-only conversion. A
            # missing tower must fail rather than initialize random visual weights.
            if (
                any(
                    loading.get(key)
                    for key in (
                        "missing_keys",
                        "unexpected_keys",
                        "mismatched_keys",
                        "error_msgs",
                    )
                )
                or getattr(model.model, "visual", None) is None
            ):
                raise ValueError("Incomplete native vision model")
            if not any(parameter.numel() for parameter in model.model.visual.parameters()):
                raise ValueError("Incomplete native vision model")
            model.eval()
            self.model = model
            self.torch = torch
            self.StoppingCriteria = StoppingCriteria
            self.StoppingCriteriaList = StoppingCriteriaList
            self.TextIteratorStreamer = TextIteratorStreamer
            self.resolved_revision = getattr(config, "_commit_hash", None)
            self.dependency_versions = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
        except Exception:  # noqa: BLE001 - never expose dependency/cache/token exception details
            # Loading exceptions can include cache paths, tokens or config dumps.
            raise RuntimeError(
                "Native Qwen initialization failed; check runtime compatibility"
            ) from None
        LOGGER.info("Native model initialized vision_capable=true")

    @staticmethod
    def _validate_config(config: Any) -> None:
        vision = getattr(config, "vision_config", None)
        if (
            getattr(config, "model_type", None) != "qwen3_5"
            or getattr(config, "architectures", None) != ["Qwen3_5ForConditionalGeneration"]
            or getattr(config, "language_model_only", None) is not False
            or getattr(vision, "patch_size", None) != VISION_PATCH_SIZE
            or getattr(vision, "spatial_merge_size", None) != VISION_MERGE_SIZE
            or getattr(vision, "temporal_patch_size", None) != 2
            or any(
                not isinstance(getattr(config, key, None), int)
                for key in (
                    "image_token_id",
                    "vision_start_token_id",
                    "vision_end_token_id",
                )
            )
        ):
            raise ValueError("Incompatible native vision configuration")

    def _validate_processor(self, config: Any) -> None:
        processor = self.processor
        image_processor = getattr(processor, "image_processor", None)
        if (
            processor.__class__.__name__ != "Qwen3VLProcessor"
            or processor.image_token_id != config.image_token_id
            or getattr(image_processor, "patch_size", None) != VISION_PATCH_SIZE
            or getattr(image_processor, "merge_size", None) != VISION_MERGE_SIZE
            or getattr(image_processor, "temporal_patch_size", None) != 2
            or self.tokenizer.chat_template != processor.chat_template
        ):
            raise ValueError("Incompatible native vision processor")
        self.vision_tokens = {
            "vision_start_token": self.tokenizer.convert_ids_to_tokens(
                config.vision_start_token_id
            ),
            "image_token": processor.image_token,
            "vision_end_token": self.tokenizer.convert_ids_to_tokens(config.vision_end_token_id),
        }
        if tuple(self.vision_tokens.values()) != (
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
        ):
            raise ValueError("Incompatible native vision tokens")

    def _prepare_generation(self, messages, generation_config):
        template_kwargs = {"add_generation_prompt": True}
        if "enable_thinking" in generation_config:
            template_kwargs["enable_thinking"] = bool(generation_config["enable_thinking"])
        vision = any(isinstance(message["content"], list) for message in messages)
        if vision:
            inputs = self.processor.apply_chat_template(
                messages,
                chat_template=self.vision_template,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
                **self.vision_tokens,
                min_pixels=MIN_VISION_PIXELS,
                max_pixels=MAX_VISION_PIXELS,
                # Qwen3VLProcessor's defaults in Transformers 5.2; neither is a
                # Qwen3_5 forward input. No blind removal of arbitrary model fields.
                return_token_type_ids=False,
                return_mm_token_type_ids=False,
            )
            grid = inputs["image_grid_thw"].tolist()
            if (
                len(grid) != 1
                or len(grid[0]) != 3
                or grid[0][0] != 1
                or any(n <= 0 for n in grid[0])
                or grid[0][1] * grid[0][2] * VISION_PATCH_SIZE**2 > MAX_VISION_PIXELS
                or "pixel_values" not in inputs
            ):
                raise ValueError("Invalid native vision tensor layout")
            patches = grid[0][1] * grid[0][2]
            image_tokens = int((inputs["input_ids"] == self.processor.image_token_id).sum().item())
            if (
                any(n % VISION_MERGE_SIZE for n in grid[0][1:])
                or inputs["pixel_values"].shape[0] != patches
                or image_tokens != patches // VISION_MERGE_SIZE**2
            ):
                raise ValueError("Invalid native vision tensor layout")
        else:
            # Preserve the validated text-only template and tokenization exactly.
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                **template_kwargs,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")

        text_device = _execution_device(self.model.get_input_embeddings())
        inputs = {
            key: value.to(
                _execution_device(self.model.model.visual) if key == "pixel_values" else text_device
            )
            for key, value in inputs.items()
        }
        do_sample = bool(generation_config.get("do_sample", False))
        generate_kwargs = {
            "max_new_tokens": int(generation_config.get("max_new_tokens", 512)),
            "do_sample": do_sample,
            "use_cache": True,
        }
        if "repetition_penalty" in generation_config:
            generate_kwargs["repetition_penalty"] = float(generation_config["repetition_penalty"])
        if do_sample:
            generate_kwargs["temperature"] = float(generation_config.get("temperature", 0.7))
            generate_kwargs["top_p"] = float(generation_config.get("top_p", 0.9))
        return inputs, generate_kwargs, int(inputs["input_ids"].shape[1])
