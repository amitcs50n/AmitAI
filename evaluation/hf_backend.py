from __future__ import annotations

from typing import Any


class TransformersGenerator:
    """Lazy Hugging Face backend for text-only Qwen3.5 baseline inference."""

    def __init__(self, model_config: dict[str, Any], seed: int) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Baseline inference dependencies are missing. Install with "
                "pip install -e '.[eval]' in a CUDA PyTorch environment."
            ) from exc

        if model_config.get("load_in_4bit", False):
            raise ValueError("The baseline harness is configured for BF16, not 4-bit loading")

        dtype_name = str(model_config.get("dtype", "bfloat16"))
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported model dtype: {dtype_name}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        revision = model_config.get("revision")
        common_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        }
        common_kwargs = {key: value for key, value in common_kwargs.items() if value is not None}
        model_kwargs = {
            **common_kwargs,
            "device_map": model_config.get("device_map", "auto"),
            "dtype": dtype_map[dtype_name],
            "low_cpu_mem_usage": True,
        }

        model_name = str(model_config["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        self.torch = torch
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        self.dependency_versions = {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        }

    def generate(
        self,
        messages: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> str:
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "enable_thinking" in generation_config:
            template_kwargs["enable_thinking"] = bool(
                generation_config["enable_thinking"]
            )
        prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        do_sample = bool(generation_config.get("do_sample", False))
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(generation_config.get("max_new_tokens", 512)),
            "do_sample": do_sample,
            "use_cache": True,
        }
        if "repetition_penalty" in generation_config:
            generate_kwargs["repetition_penalty"] = float(
                generation_config["repetition_penalty"]
            )
        if do_sample:
            generate_kwargs["temperature"] = float(
                generation_config.get("temperature", 0.7)
            )
            generate_kwargs["top_p"] = float(generation_config.get("top_p", 0.9))

        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = generated[:, prompt_length:]
        return self.tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
