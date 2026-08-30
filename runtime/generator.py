"""Real Hugging Face chat adapter with bounded mechanical repair."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from threading import Event
from typing import Any, Protocol

from backend.chat_service import (
    ChatGenerationDelta,
    ChatGenerationError,
    ChatGenerationResult,
    GenerationMessage,
)
from evaluation.constraints import (
    MAX_MECHANICAL_RETRIES,
    parse_constraints,
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
        return self._validate_output(output, strip_text=True)

    @staticmethod
    def _validate_output(
        output: object,
        *,
        strip_text: bool,
    ) -> GenerationOutput:
        if not isinstance(output, GenerationOutput):
            raise TypeError("Runtime engine must return GenerationOutput")
        if not isinstance(output.text, str) or not output.text.strip():
            raise ValueError("Runtime engine returned an empty response")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(output, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"Runtime engine {field_name} must be nonnegative")
        return GenerationOutput(
            text=output.text.strip() if strip_text else output.text,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
        )

    def _stream_once(
        self,
        messages: list[dict[str, str]],
        *,
        cancel_event: Event,
    ) -> Iterator[str | GenerationOutput]:
        engine = self._get_engine()
        stream_method = getattr(engine, "generate_detailed_stream", None)
        if not callable(stream_method):
            output = self._generate_once(messages)
            if cancel_event.is_set():
                return
            yield output.text
            yield output
            return

        with self._generation_lock:
            stream = iter(
                stream_method(
                    messages,
                    dict(self.config.generation),
                    cancel_event=cancel_event,
                )
            )
            chunks: list[str] = []
            output: GenerationOutput | None = None
            try:
                for item in stream:
                    if cancel_event.is_set():
                        return
                    if isinstance(item, str):
                        if output is not None:
                            raise TypeError("Runtime engine streamed text after final output")
                        if not item:
                            continue
                        chunks.append(item)
                        yield item
                        continue
                    if isinstance(item, GenerationOutput):
                        if output is not None:
                            raise TypeError("Runtime engine returned final output more than once")
                        output = self._validate_output(item, strip_text=False)
                        continue
                    raise TypeError(
                        "Runtime stream must yield text chunks or GenerationOutput"
                    )
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

        if cancel_event.is_set():
            return
        if output is None:
            raise TypeError("Runtime engine stream ended without final output")
        if "".join(chunks) != output.text:
            raise ValueError("Runtime engine chunks do not reconstruct its final output")
        yield output

    def _model_messages(
        self,
        messages: Sequence[GenerationMessage],
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.config.runtime_system_prompt},
            *[{"role": item.role, "content": item.content} for item in messages],
        ]

    @staticmethod
    def _validator_metadata(validation: dict[str, Any]) -> dict[str, Any]:
        retry_count = validation["retry_count"]
        final_validation = validation["final_validation"]
        validator_metadata: dict[str, Any] = {
            "retry_attempted": retry_count > 0,
            "retry_passed": None if retry_count == 0 else final_validation["passed"],
            "retry_count": retry_count,
            "parsed_constraints": validation["parsed_constraints"],
            "final_validation": final_validation,
        }
        if retry_count:
            validator_metadata["first_retry_passed"] = validation["retry_passed"]
        return validator_metadata

    def generate_response(
        self,
        messages: Sequence[GenerationMessage],
    ) -> ChatGenerationResult:
        return self._generate_validated_response(messages, self._generate_once)

    def _generate_validated_response(
        self,
        messages: Sequence[GenerationMessage],
        generate_once: Callable[[list[dict[str, str]]], GenerationOutput],
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
            output = generate_once(model_messages)
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
        if validation["final_validation"]["passed"] is not True:
            raise ChatGenerationError("Assistant generation failed")

        latency_ms = max(0, int((self._clock() - start) * 1000))
        return ChatGenerationResult(
            response=validation["final_response"],
            model=str(self.config.model["name"]),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            validator=self._validator_metadata(validation),
            tools=[],
            memory=[],
        )

    def stream_response(
        self,
        messages: Sequence[GenerationMessage],
        *,
        cancel_event: Event,
    ) -> Iterator[ChatGenerationDelta | ChatGenerationResult]:
        if not messages or messages[-1].role != "user":
            raise ValueError("Runtime chat messages must end with the current user turn")

        current_prompt = messages[-1].content
        if parse_constraints(current_prompt):
            def generate_buffered(model_messages: list[dict[str, str]]) -> GenerationOutput:
                if cancel_event.is_set():
                    raise RuntimeError("Runtime streaming generation was cancelled")
                output: GenerationOutput | None = None
                engine_stream = self._stream_once(
                    model_messages,
                    cancel_event=cancel_event,
                )
                try:
                    for item in engine_stream:
                        if cancel_event.is_set():
                            raise RuntimeError("Runtime streaming generation was cancelled")
                        if isinstance(item, GenerationOutput):
                            output = item
                finally:
                    close = getattr(engine_stream, "close", None)
                    if callable(close):
                        close()
                if cancel_event.is_set():
                    raise RuntimeError("Runtime streaming generation was cancelled")
                if output is None:
                    raise TypeError("Runtime engine stream ended without final output")
                return output

            result = self._generate_validated_response(messages, generate_buffered)
            if cancel_event.is_set():
                return
            yield ChatGenerationDelta(delta=result.response)
            yield result
            return

        start = self._clock()
        pending = ""
        emitted: list[str] = []
        output: GenerationOutput | None = None
        engine_stream = self._stream_once(
            self._model_messages(messages),
            cancel_event=cancel_event,
        )
        try:
            for item in engine_stream:
                if cancel_event.is_set():
                    return
                if isinstance(item, GenerationOutput):
                    output = item
                    continue

                pending += item
                if not emitted:
                    pending = pending.lstrip()
                if not pending:
                    continue
                last_non_whitespace = next(
                    (
                        index
                        for index in range(len(pending) - 1, -1, -1)
                        if not pending[index].isspace()
                    ),
                    -1,
                )
                if last_non_whitespace < 0:
                    continue
                delta = pending[: last_non_whitespace + 1]
                pending = pending[last_non_whitespace + 1 :]
                emitted.append(delta)
                yield ChatGenerationDelta(delta=delta)
        finally:
            close = getattr(engine_stream, "close", None)
            if callable(close):
                close()

        if cancel_event.is_set():
            return
        if output is None:
            raise TypeError("Runtime engine stream ended without final output")

        response = "".join(emitted)
        if response != output.text.strip():
            raise ValueError("Normalized runtime deltas do not match the final output")

        def unexpected_retry(_: str) -> str:
            raise RuntimeError("Unconstrained streaming must not retry")

        validation = validate_with_bounded_retries(
            current_prompt,
            response,
            unexpected_retry,
            max_retries=MAX_MECHANICAL_RETRIES,
        )
        latency_ms = max(0, int((self._clock() - start) * 1000))
        yield ChatGenerationResult(
            response=response,
            model=str(self.config.model["name"]),
            latency_ms=latency_ms,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            validator=self._validator_metadata(validation),
            tools=[],
            memory=[],
        )
