"""Provider-neutral chat orchestration with bounded mechanical repair."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from threading import Event
from typing import Any

from backend.chat_service import (
    ChatGenerationDelta,
    ChatGenerationError,
    ChatGenerationResult,
    ChatPrivacyError,
    GenerationMessage,
)
from evaluation.constraints import (
    MAX_MECHANICAL_RETRIES,
    parse_constraints,
    validate_with_bounded_retries,
)
from evaluation.hf_backend import GenerationOutput

from .calculator import CalculatorTool
from .config import RuntimeConfig
from .context import compile_model_messages
from .privacy import RemoteDisclosureBlockedError, require_execution_scope
from .providers import (
    EngineFactory,
    InferenceProvider,
    LocalTransformersInferenceProvider,
)
from .tooling import (
    MAX_TOOL_ITERATIONS,
    LateToolProtocolFilter,
    ToolAttempt,
    ToolFailure,
    ToolRegistry,
    classify_tool_protocol_prefix,
    failed_tool_attempt,
    format_tool_call,
    format_tool_result,
    is_reserved_tool_candidate,
    parse_tool_call,
    sanitize_late_tool_protocol,
)


@dataclass(frozen=True)
class _ToolLoopOutput:
    output: GenerationOutput
    tools: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _StreamCandidateOutput:
    output: GenerationOutput
    reserved: bool


@dataclass
class _ToolBudget:
    attempted_turns: int = 0

    def consume(self) -> int:
        self.attempted_turns += 1
        if self.attempted_turns > MAX_TOOL_ITERATIONS:
            raise ChatGenerationError("Assistant generation failed")
        return self.attempted_turns


class ProviderChatGenerator:
    """Keep private chat orchestration local while delegating stateless inference."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        provider: InferenceProvider,
        clock: Callable[[], float] = time.perf_counter,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        if not config.mechanical_constraints_enabled:
            raise ValueError("Runtime mechanical constraint validation must be enabled")
        self.config = config
        self._provider = provider
        self._execution_scope = require_execution_scope(provider)
        self._clock = clock
        self._tool_registry = tool_registry or ToolRegistry([CalculatorTool()])

    def _generate_once(self, messages: list[dict[str, str]]) -> GenerationOutput:
        try:
            output = self._provider.generate(messages, self.config.generation)
        except RemoteDisclosureBlockedError:
            raise ChatPrivacyError() from None
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
        stream: Iterator[str | GenerationOutput] | None = None
        chunks: list[str] = []
        output: GenerationOutput | None = None
        try:
            stream = iter(self._provider.stream(
                messages, self.config.generation, cancel_event=cancel_event,
            ))
            for item in stream:
                if cancel_event.is_set():
                    return
                if isinstance(item, str):
                    if output is not None:
                        raise TypeError("Runtime provider streamed text after final output")
                    if not item:
                        continue
                    chunks.append(item)
                    yield item
                    continue
                if isinstance(item, GenerationOutput):
                    if output is not None:
                        raise TypeError("Runtime provider returned final output more than once")
                    output = self._validate_output(item, strip_text=False)
                    continue
                raise TypeError(
                    "Runtime provider stream must yield text chunks or GenerationOutput"
                )
        except RemoteDisclosureBlockedError:
            raise ChatPrivacyError() from None
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        if cancel_event.is_set():
            return
        if output is None:
            raise TypeError("Runtime provider stream ended without final output")
        if "".join(chunks) != output.text:
            raise ValueError("Runtime provider chunks do not reconstruct its final output")
        yield output

    @staticmethod
    def _safe_assistant_tool_message(text: str) -> str:
        try:
            return format_tool_call(parse_tool_call(text))
        except ToolFailure:
            return '<tool_call>{"arguments":{},"name":"invalid_tool_call"}</tool_call>'

    def _execute_tool_candidate(
        self,
        text: str,
        *,
        attempt_number: int,
    ) -> ToolAttempt:
        try:
            call = parse_tool_call(text)
        except ToolFailure as exc:
            attempt = failed_tool_attempt(attempt=attempt_number, failure=exc)
        else:
            attempt = self._tool_registry.execute(call, attempt=attempt_number)
        return attempt

    def _run_tool_loop(
        self,
        model_messages: list[dict[str, str]],
        generate_once: Callable[[list[dict[str, str]]], GenerationOutput],
        *,
        budget: _ToolBudget,
    ) -> _ToolLoopOutput:
        working_messages = list(model_messages)
        tool_records: list[dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0

        while True:
            output = generate_once(working_messages)
            input_tokens += output.input_tokens
            output_tokens += output.output_tokens
            response = output.text.strip()
            if not is_reserved_tool_candidate(response):
                response = sanitize_late_tool_protocol(response).strip()
                if not response:
                    raise ChatGenerationError("Assistant generation failed")
                return _ToolLoopOutput(
                    output=GenerationOutput(response, input_tokens, output_tokens),
                    tools=tuple(tool_records),
                )

            attempt_number = budget.consume()
            attempt = self._execute_tool_candidate(
                response,
                attempt_number=attempt_number,
            )
            record = attempt.as_record()
            tool_records.append(record)
            working_messages.extend(
                (
                    {
                        "role": "assistant",
                        "content": self._safe_assistant_tool_message(response),
                    },
                    {
                        "role": "system",
                        "content": format_tool_result(attempt),
                    },
                )
            )

    def _stream_candidate(
        self,
        model_messages: list[dict[str, str]],
        *,
        cancel_event: Event,
    ) -> Iterator[str | _StreamCandidateOutput]:
        prefix_buffer = ""
        pending = ""
        emitted: list[str] = []
        mode = "prefix"
        protocol_filter: LateToolProtocolFilter | None = None
        output: GenerationOutput | None = None

        def normalize(chunk: str) -> str | None:
            nonlocal pending
            pending += chunk
            if not emitted:
                pending = pending.lstrip()
            if not pending:
                return None
            last_non_whitespace = next(
                (
                    index
                    for index in range(len(pending) - 1, -1, -1)
                    if not pending[index].isspace()
                ),
                -1,
            )
            if last_non_whitespace < 0:
                return None
            delta = pending[: last_non_whitespace + 1]
            pending = pending[last_non_whitespace + 1 :]
            emitted.append(delta)
            return delta

        engine_stream = self._stream_once(model_messages, cancel_event=cancel_event)
        try:
            for item in engine_stream:
                if cancel_event.is_set():
                    return
                if isinstance(item, GenerationOutput):
                    output = item
                    continue
                if mode == "tool":
                    continue
                if mode == "prefix":
                    prefix_buffer += item
                    prefix_state = classify_tool_protocol_prefix(prefix_buffer)
                    if prefix_state == "ambiguous":
                        continue
                    if prefix_state == "reserved":
                        mode = "tool"
                        continue
                    mode = "normal"
                    protocol_filter = LateToolProtocolFilter()
                    item = prefix_buffer
                    prefix_buffer = ""
                if protocol_filter is None:
                    raise RuntimeError("Normal stream is missing its protocol filter")
                delta = normalize(protocol_filter.feed(item))
                if delta is not None:
                    yield delta
        finally:
            close = getattr(engine_stream, "close", None)
            if callable(close):
                close()

        if cancel_event.is_set():
            return
        if output is None:
            raise TypeError("Runtime engine stream ended without final output")
        if mode == "tool":
            yield _StreamCandidateOutput(
                output=GenerationOutput(
                    output.text.strip(),
                    output.input_tokens,
                    output.output_tokens,
                ),
                reserved=True,
            )
            return

        if mode == "prefix":
            protocol_filter = LateToolProtocolFilter()
            delta = normalize(protocol_filter.feed(prefix_buffer))
            if delta is not None:
                yield delta
        if protocol_filter is None:
            raise RuntimeError("Normal stream is missing its protocol filter")
        delta = normalize(protocol_filter.finish())
        if delta is not None:
            yield delta

        response = "".join(emitted)
        if not response:
            raise ChatGenerationError("Assistant generation failed")
        yield _StreamCandidateOutput(
            output=GenerationOutput(
                response,
                output.input_tokens,
                output.output_tokens,
            ),
            reserved=False,
        )

    def _stream_tool_loop(
        self,
        model_messages: list[dict[str, str]],
        *,
        cancel_event: Event,
    ) -> Iterator[str | _ToolLoopOutput]:
        working_messages = list(model_messages)
        tool_records: list[dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        budget = _ToolBudget()

        while not cancel_event.is_set():
            candidate: _StreamCandidateOutput | None = None
            candidate_stream = self._stream_candidate(
                working_messages,
                cancel_event=cancel_event,
            )
            try:
                for item in candidate_stream:
                    if cancel_event.is_set():
                        return
                    if isinstance(item, _StreamCandidateOutput):
                        candidate = item
                    else:
                        yield item
            finally:
                close = getattr(candidate_stream, "close", None)
                if callable(close):
                    close()

            if cancel_event.is_set():
                return
            if candidate is None:
                raise TypeError("Runtime stream ended without a candidate result")
            input_tokens += candidate.output.input_tokens
            output_tokens += candidate.output.output_tokens
            if not candidate.reserved:
                yield _ToolLoopOutput(
                    output=GenerationOutput(
                        candidate.output.text,
                        input_tokens,
                        output_tokens,
                    ),
                    tools=tuple(tool_records),
                )
                return

            attempt_number = budget.consume()
            attempt = self._execute_tool_candidate(
                candidate.output.text,
                attempt_number=attempt_number,
            )
            record = attempt.as_record()
            tool_records.append(record)
            working_messages.extend(
                (
                    {
                        "role": "assistant",
                        "content": self._safe_assistant_tool_message(
                            candidate.output.text
                        ),
                    },
                    {
                        "role": "system",
                        "content": format_tool_result(attempt),
                    },
                )
            )

    def _model_messages(
        self,
        messages: Sequence[GenerationMessage],
    ) -> list[dict[str, str]]:
        return compile_model_messages(
            messages,
            runtime_system_prompt=self.config.runtime_system_prompt,
            tool_instructions=self._tool_registry.instructions(),
            execution_scope=self._execution_scope,
        )

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
        tool_records: list[dict[str, Any]] = []
        tool_budget = _ToolBudget()

        def generate(model_messages: list[dict[str, str]]) -> str:
            nonlocal input_tokens, output_tokens
            loop_output = self._run_tool_loop(
                model_messages,
                generate_once,
                budget=tool_budget,
            )
            input_tokens += loop_output.output.input_tokens
            output_tokens += loop_output.output.output_tokens
            tool_records.extend(loop_output.tools)
            return loop_output.output.text

        model_messages = self._model_messages(messages)
        original_response = generate(model_messages)

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
            retry_original_prompt=model_messages[-1]["content"],
        )
        if validation["final_validation"]["passed"] is not True:
            raise ChatGenerationError("Assistant generation failed")

        latency_ms = max(0, int((self._clock() - start) * 1000))
        return ChatGenerationResult(
            response=validation["final_response"],
            model=self._provider.model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            validator=self._validator_metadata(validation),
            tools=tool_records,
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
        loop_output: _ToolLoopOutput | None = None
        tool_stream = self._stream_tool_loop(
            self._model_messages(messages),
            cancel_event=cancel_event,
        )
        try:
            for item in tool_stream:
                if cancel_event.is_set():
                    return
                if isinstance(item, _ToolLoopOutput):
                    loop_output = item
                else:
                    yield ChatGenerationDelta(delta=item)
        finally:
            close = getattr(tool_stream, "close", None)
            if callable(close):
                close()

        if cancel_event.is_set():
            return
        if loop_output is None:
            raise TypeError("Runtime tool stream ended without final output")

        def unexpected_retry(_: str) -> str:
            raise RuntimeError("Unconstrained streaming must not retry")

        validation = validate_with_bounded_retries(
            current_prompt,
            loop_output.output.text,
            unexpected_retry,
            max_retries=MAX_MECHANICAL_RETRIES,
        )
        latency_ms = max(0, int((self._clock() - start) * 1000))
        yield ChatGenerationResult(
            response=loop_output.output.text,
            model=self._provider.model_name,
            latency_ms=latency_ms,
            input_tokens=loop_output.output.input_tokens,
            output_tokens=loop_output.output.output_tokens,
            validator=self._validator_metadata(validation),
            tools=list(loop_output.tools),
            memory=[],
        )


class TransformersChatGenerator(ProviderChatGenerator):
    """Backward-compatible local Transformers composition of the provider runtime."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        engine_factory: EngineFactory | None = None,
        clock: Callable[[], float] = time.perf_counter,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        if engine_factory is None:
            provider = LocalTransformersInferenceProvider(
                config.model,
                int(config.generation["seed"]),
            )
        else:
            provider = LocalTransformersInferenceProvider(
                config.model,
                int(config.generation["seed"]),
                engine_factory=engine_factory,
            )
        super().__init__(
            config,
            provider=provider,
            clock=clock,
            tool_registry=tool_registry,
        )
