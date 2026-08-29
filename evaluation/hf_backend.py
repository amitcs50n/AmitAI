from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    input_tokens: int
    output_tokens: int


class TransformersGenerator:
    """Lazy Hugging Face backend for text-only causal-language-model inference."""

    def __init__(self, model_config: dict[str, Any], seed: int) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                StoppingCriteria,
                StoppingCriteriaList,
                TextIteratorStreamer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face inference dependencies are missing. Install with "
                "pip install -e '.[eval]' or pip install -e '.[runtime]' in a "
                "CUDA PyTorch environment."
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
        self.StoppingCriteria = StoppingCriteria
        self.StoppingCriteriaList = StoppingCriteriaList
        self.TextIteratorStreamer = TextIteratorStreamer
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
        return self.generate_detailed(messages, generation_config).text

    def generate_detailed(
        self,
        messages: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> GenerationOutput:
        inputs, generate_kwargs, prompt_length = self._prepare_generation(
            messages,
            generation_config,
        )

        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)

        completion_ids = generated[:, prompt_length:]
        text = self.tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return GenerationOutput(
            text=text,
            input_tokens=int(prompt_length),
            output_tokens=int(completion_ids.shape[1]),
        )

    def _prepare_generation(
        self,
        messages: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], int]:
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

        prompt_length = inputs["input_ids"].shape[1]
        return inputs, generate_kwargs, int(prompt_length)

    def generate_detailed_stream(
        self,
        messages: list[dict[str, Any]],
        generation_config: dict[str, Any],
        *,
        cancel_event: Event,
    ) -> Iterator[str | GenerationOutput]:
        """Yield decoded text chunks followed by one detailed terminal result."""

        if cancel_event.is_set():
            return

        inputs, generate_kwargs, prompt_length = self._prepare_generation(
            messages,
            generation_config,
        )
        streamer = self.TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        torch = self.torch
        signal = cancel_event
        stopping_criteria_base = self.StoppingCriteria

        class CancellationCriteria(stopping_criteria_base):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                del scores, kwargs
                return torch.full(
                    (input_ids.shape[0],),
                    signal.is_set(),
                    dtype=torch.bool,
                    device=input_ids.device,
                )

        worker_kwargs = {
            **inputs,
            **generate_kwargs,
            "streamer": streamer,
            "stopping_criteria": self.StoppingCriteriaList([CancellationCriteria()]),
        }
        generated_outputs: list[Any] = []
        generation_errors: list[Exception] = []

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    generated_outputs.append(self.model.generate(**worker_kwargs))
            # Relay every ordinary worker failure and always unblock the streamer.
            except Exception as exc:  # noqa: BLE001
                generation_errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(
            target=run_generation,
            name="amitai-transformers-generate",
            daemon=True,
        )
        worker.start()
        chunks: list[str] = []
        completed = False
        try:
            for chunk in streamer:
                if chunk:
                    chunks.append(chunk)
                    yield chunk

            worker.join()
            if generation_errors:
                raise generation_errors[0]
            if len(generated_outputs) != 1:
                raise RuntimeError("Transformers generation ended without token output")

            completion_ids = generated_outputs[0][:, prompt_length:]
            completed = True
            yield GenerationOutput(
                text="".join(chunks),
                input_tokens=prompt_length,
                output_tokens=int(completion_ids.shape[1]),
            )
        finally:
            if not completed:
                cancel_event.set()
            if worker.is_alive():
                worker.join()
