"""Real Hugging Face chat adapter with bounded mechanical repair."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from backend.chat_service import ChatGenerationResult, GenerationMessage
from evaluation.constraints import (
    MAX_MECHANICAL_RETRIES,
    validate_with_bounded_retries,
)
from evaluation.hf_backend import GenerationOutput, TransformersGenerator

from .config import RuntimeConfig


class DetailedGenerationEngine(Protocol):
    def generate_detailed(
        self,
        messages: list[dict[str, str]],
        generation_config: dict[str, object],
    ) -> GenerationOutput: ...


EngineFactory = Callable[[dict[str, object], int], DetailedGenerationEngine]


class TransformersChatGenerator:
    """Adapt the tested text generator to the persistent chat backend contract."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        engine_factory: EngineFactory = TransformersGenerator,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not config.mechanical_constraints_enabled:
            raise ValueError("Runtime mechanical constraint validation must be enabled")
        self.config = config
        self._engine_factory = engine_factory
        self._clock = clock
        self._engine: DetailedGenerationEngine | None = None
        self._initialization_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def _get_engine(self) -> DetailedGenerationEngine:
        engine = self._engine
        if engine is not None:
            return engine
        with self._initialization_lock:
            if self._engine is None:
                candidate = self._engine_factory(
                    dict(self.config.model),
                    int(self.config.generation["seed"]),
                )
                self._engine = candidate
            return self._engine

    def _generate_once(self, messages: list[dict[str, str]]) -> GenerationOutput:
        engine = self._get_engine()
        with self._generation_lock:
            output = engine.generate_detailed(messages, dict(self.config.generation))
        if not isinstance(output, GenerationOutput):
            raise TypeError("Runtime engine must return GenerationOutput")
        if not isinstance(output.text, str) or not output.text.strip():
            raise ValueError("Runtime engine returned an empty response")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(output, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"Runtime engine {field_name} must be nonnegative")
        return GenerationOutput(
            text=output.text.strip(),
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
        )

    def _model_messages(
        self,
        messages: Sequence[GenerationMessage],
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.config.runtime_system_prompt},
            *[{"role": item.role, "content": item.content} for item in messages],
        ]

    def generate_response(
        self,
        messages: Sequence[GenerationMessage],
    ) -> ChatGenerationResult:
        if not messages or messages[-1].role != "user":
            raise ValueError("Runtime chat messages must end with the current user turn")

        start = self._clock()
        current_prompt = messages[-1].content
        prior_history = tuple(messages[:-1])
        input_tokens = 0
        output_tokens = 0

        def generate(model_messages: list[dict[str, str]]) -> str:
            nonlocal input_tokens, output_tokens
            output = self._generate_once(model_messages)
            input_tokens += output.input_tokens
            output_tokens += output.output_tokens
            return output.text

        original_response = generate(self._model_messages(messages))

        def retry(corrective_prompt: str) -> str:
            retry_messages = (
                *prior_history,
                GenerationMessage(role="user", content=corrective_prompt),
            )
            return generate(self._model_messages(retry_messages))

        validation = validate_with_bounded_retries(
            current_prompt,
            original_response,
            retry,
            max_retries=MAX_MECHANICAL_RETRIES,
        )
        retry_count = validation["retry_count"]
        final_validation = validation["final_validation"]
        validator_metadata = {
            "retry_attempted": retry_count > 0,
            "retry_passed": None if retry_count == 0 else final_validation["passed"],
            "retry_count": retry_count,
            "parsed_constraints": validation["parsed_constraints"],
            "final_validation": final_validation,
        }
        if retry_count:
            validator_metadata["first_retry_passed"] = validation["retry_passed"]

        latency_ms = max(0, int((self._clock() - start) * 1000))
        return ChatGenerationResult(
            response=validation["final_response"],
            model=str(self.config.model["name"]),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            validator=validator_metadata,
            tools=[],
            memory=[],
        )
