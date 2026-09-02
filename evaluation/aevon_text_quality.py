"""Production-path text measurement: deterministic checks plus pending human review.

Fake responses are harness fixtures, never quality evidence or provider input.
No database, memory mutations, judge model, or inference defaults are introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from threading import Event
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from backend.chat_service import ChatGenerationDelta, ChatGenerationResult, GenerationMessage
from backend.memory import MAX_MEMORY_CONTEXT_CHARS, MAX_RETRIEVED_MEMORIES, format_memory_context
from evaluation.baseline import append_jsonl, load_jsonl, stable_fingerprint, write_json
from evaluation.constraints import parse_constraints, validate_response
from evaluation.hf_backend import GenerationOutput
from evaluation.run_baseline import git_revision
from runtime.app import select_response_generator
from runtime.config import RuntimeConfig, load_production_runtime_config
from runtime.context import MAX_HISTORY_CONTEXT_CHARS, MAX_HISTORY_MESSAGES
from runtime.generator import ProviderChatGenerator
from runtime.privacy import InferenceExecutionScope
from runtime.providers import (
    InferenceProvider,
    LocalTransformersInferenceProvider,
    RemoteInferenceProvider,
)

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "eval/aevon_text_quality_v1.jsonl"
Category = Literal[
    "identity", "conversation", "judgment", "technical", "reasoning", "continuity",
    "memory", "tone", "format", "tools", "uncertainty", "long_context",
]
Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
MALFORMED_TOOL_FIXTURE = '<tool_call>{"name":</tool_call>'


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CaseMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: Text


class SyntheticMemory(StrictModel):
    category: Literal["preference", "profile", "project", "workflow", "instruction"]
    key: Text
    value: Text


class Expectations(StrictModel):
    contains: list[Text] = Field(default_factory=list)
    not_contains: list[Text] = Field(default_factory=list)
    identity: Literal["assistant", "model"] | None = None
    tools: Literal["required", "forbidden"] | None = None
    mechanical: bool = False
    memory_contains: list[Text] = Field(default_factory=list)
    memory_not_contains: list[Text] = Field(default_factory=list)
    context_contains: list[Text] = Field(default_factory=list)
    context_not_contains: list[Text] = Field(default_factory=list)


class QualityCase(StrictModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
    category: Category
    messages: list[CaseMessage] = Field(min_length=1)
    expectations: Expectations
    human_review: list[Text] = Field(min_length=1)
    fake_responses: list[Text] = Field(min_length=1)
    memory: list[SyntheticMemory] = Field(default_factory=list, max_length=MAX_RETRIEVED_MEMORIES)
    history_fixture: Literal["count_window", "char_window", "orphan", "oversized"] | None = None
    scenario: Literal["natural", "forced_malformed_tool_call"] = "natural"

    @model_validator(mode="after")
    def check_context(self) -> QualityCase:
        if self.messages[-1].role != "user":
            raise ValueError("Case must end with the current user message")
        if self.expectations.mechanical and not parse_constraints(self.messages[-1].content):
            raise ValueError("Mechanical case must use a supported production constraint")
        if self.memory and len(format_memory_context(
            [item.model_dump() for item in self.memory]
        )) > MAX_MEMORY_CONTEXT_CHARS:
            raise ValueError("Synthetic memory exceeds the production context limit")
        if self.scenario != "natural" and self.category != "tools":
            raise ValueError("Fault injection must be explicitly categorized as tools")
        return self


def load_cases(path: str | Path = DEFAULT_CASES) -> list[QualityCase]:
    cases = [QualityCase.model_validate(row) for row in load_jsonl(path)]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Benchmark must be nonempty with unique case IDs")
    return cases


def generation_messages(case: QualityCase) -> list[GenerationMessage]:
    """Expand reproducible synthetic context, never retrieve or mutate real memory."""
    messages = []
    if case.memory:
        messages.append(GenerationMessage("system", format_memory_context(
            [item.model_dump() for item in case.memory]
        )))
    fixture = case.history_fixture
    if fixture in {"count_window", "char_window"}:
        # Fixed V1 fixtures, not derived from whatever limits a future runtime uses.
        width = 32 if fixture == "count_window" else 1200
        messages.extend(
            GenerationMessage(
                "user" if index % 2 == 0 else "assistant",
                (f"OLD_CONTEXT_CANARY_{index:02d} " + "archived irrelevant detail " * width)[:width],
            )
            for index in range(24)
        )
    elif fixture == "orphan":
        messages.append(GenerationMessage("assistant", "ORPHAN_CONTEXT_CANARY"))
    elif fixture == "oversized":
        messages.extend((
            GenerationMessage("user", "OLD_CONTEXT_CANARY_00"),
            GenerationMessage("assistant", "OVERSIZED_CONTEXT_CANARY " + "x" * 20_001),
        ))
    messages.extend(GenerationMessage(item.role, item.content) for item in case.messages)
    return messages


class ScriptedProvider:
    """Offline harness fixture. Counts and latency are synthetic, not model metrics."""

    provider_name = "fake-scripted"
    model_name = "fake-scripted"
    execution_scope = InferenceExecutionScope.LOCAL

    def __init__(self) -> None:
        self.responses: Iterator[str] = iter(())

    def generate(self, messages, generation_config) -> GenerationOutput:
        text = next(self.responses)
        return GenerationOutput(text, 0, 0)

    def stream(self, messages, generation_config, *, cancel_event):
        output = self.generate(messages, generation_config)
        for offset in range(0, len(output.text), 7):
            if cancel_event.is_set():
                return
            yield output.text[offset:offset + 7]
        yield output


class ObservedProvider:
    """Evaluation-only provider decorator; does not alter ordinary model calls.

    The single labeled recovery scenario substitutes one synthetic malformed
    output BEFORE real inference. Subsequent calls run the unmodified tool loop.
    Captured prompts remain request-local; artifacts store input cases, not raw
    intermediate model/tool candidates or provider credentials.
    """

    def __init__(self, provider: InferenceProvider) -> None:
        self.provider = provider
        self.provider_name = provider.provider_name
        self.model_name = provider.model_name
        self.execution_scope = provider.execution_scope
        self.calls: list[list[dict[str, str]]] = []
        self.inject_fault = False
        self.injected_outputs = 0

    def prepare(self, case: QualityCase) -> None:
        self.calls = []
        self.inject_fault = case.scenario == "forced_malformed_tool_call"
        self.injected_outputs = 0
        if isinstance(self.provider, ScriptedProvider):
            self.provider.responses = iter(case.fake_responses)

    def _observe(self, messages: Sequence[Mapping[str, str]]) -> GenerationOutput | None:
        self.calls.append([dict(message) for message in messages])
        if self.inject_fault:
            self.inject_fault = False
            self.injected_outputs += 1
            return GenerationOutput(MALFORMED_TOOL_FIXTURE, 0, 0)
        return None

    def generate(self, messages, generation_config):
        injected = self._observe(messages)
        return injected if injected is not None else self.provider.generate(messages, generation_config)

    def stream(self, messages, generation_config, *, cancel_event):
        injected = self._observe(messages)
        if injected is not None:
            yield injected.text
            yield injected
        else:
            yield from self.provider.stream(messages, generation_config, cancel_event=cancel_event)

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


def build_runtime(mode: str) -> tuple[ProviderChatGenerator, ObservedProvider]:
    if mode == "fake":
        observed = ObservedProvider(ScriptedProvider())
        return ProviderChatGenerator(
            load_production_runtime_config(), provider=observed, clock=lambda: 0.0,
        ), observed

    providers: list[ObservedProvider] = []

    def local_factory(config: RuntimeConfig) -> ProviderChatGenerator:
        observed = ObservedProvider(LocalTransformersInferenceProvider(
            config.model, int(config.generation["seed"]),
        ))
        providers.append(observed)
        return ProviderChatGenerator(config, provider=observed)

    def remote_factory(**kwargs) -> ObservedProvider:
        observed = ObservedProvider(RemoteInferenceProvider(**kwargs))
        providers.append(observed)
        return observed

    if mode not in {"transformers", "remote"}:
        raise ValueError("Explicit benchmark mode must be fake, transformers, or remote")
    # Existing startup owns settings, environment selection, transport and privacy.
    generator = select_response_generator(
        mode=mode, generator_factory=local_factory, remote_provider_factory=remote_factory,
    )
    if not isinstance(generator, ProviderChatGenerator):
        raise TypeError("Benchmark requires the production provider-backed generator")
    return generator, providers[0]


def grade(
    case: QualityCase,
    result: ChatGenerationResult | None,
    calls: list[list[dict[str, str]]],
    *,
    config: RuntimeConfig,
) -> list[dict[str, Any]]:
    """Small literal checks are evidence, never a substitute for semantic review."""
    checks: list[dict[str, Any]] = []
    text = result.response if result is not None else ""
    folded = text.casefold()
    expected = case.expectations

    def add(name: str, group: str, passed: bool, detail: object = None) -> None:
        checks.append({"name": name, "group": group, "passed": passed, "detail": detail})

    add("generation_completed", "generation", result is not None)
    add("no_protocol_leakage", "tools", result is not None and not any(
        marker in folded for marker in ("<tool_", "</tool_", "<memory_context", "memory_context_v1")
    ))
    for field, group, positive in (
        ("contains", case.category, True), ("not_contains", case.category, False),
        ("memory_contains", "memory", True), ("memory_not_contains", "memory", False),
    ):
        for index, phrase in enumerate(getattr(expected, field)):
            add(f"{field}_{index}", group, result is not None and (
                (phrase.casefold() in folded) == positive
            ), phrase)
    if expected.identity == "assistant":
        # Deliberately narrow ONLY for short name probes, not conversation grading.
        add("assistant_identity", "identity", re.fullmatch(
            r"(?:my name is |i am |i'm )?aevon[.!]?", text.strip(), re.IGNORECASE,
        ) is not None)
    elif expected.identity == "model":
        add("configured_model_identity", "identity", str(config.model["name"]).casefold() in folded)
    tools = result.tools if result is not None else []
    if expected.tools:
        passed = (
            any(tool.get("name") == "calculator" and tool.get("success") is True for tool in tools)
            if expected.tools == "required" else not tools
        )
        add("tool_usage", "tools", result is not None and passed)
    if case.scenario == "forced_malformed_tool_call":
        add("malformed_tool_recovery", "tools", result is not None and bool(tools)
            and tools[0].get("success") is False
            and tools[0].get("error", {}).get("code") == "malformed_tool_call")
    constraints = parse_constraints(case.messages[-1].content)
    if constraints:
        validation = validate_response(text, constraints) if result is not None else None
        add("mechanical_constraints", "mechanical", validation is not None and validation["passed"],
            validation)

    initial = calls[0] if calls else []
    add("production_identity_context", "context", bool(initial) and initial[0]["content"].startswith(
        config.runtime_system_prompt + "\n\n"
    ))
    add("current_user_retained", "context", bool(initial) and initial[-1] == {
        "role": "user", "content": case.messages[-1].content,
    })
    trusted_count = 1 if case.memory else 0
    history = initial[1 + trusted_count:-1]
    add("history_limits", "context", bool(initial) and len(history) <= MAX_HISTORY_MESSAGES
        and sum(len(item["content"]) for item in history) <= MAX_HISTORY_CONTEXT_CHARS)
    add("no_orphan_assistant", "context", bool(initial) and (not history or history[0]["role"] != "assistant"))
    if case.memory:
        add("trusted_memory_order", "context", len(initial) >= 3 and initial[1] == {
            "role": "system", "content": format_memory_context([item.model_dump() for item in case.memory]),
        })
    for field, positive in (("context_contains", True), ("context_not_contains", False)):
        for index, phrase in enumerate(getattr(expected, field)):
            # All requests: repairs/tool followups must not restore dropped history.
            add(f"{field}_{index}", "context", bool(calls) and all(
                (phrase in "\n".join(item["content"] for item in call)) == positive for call in calls
            ), phrase)
    return checks


def evaluate_case(
    case: QualityCase, generator: ProviderChatGenerator, observed: ObservedProvider,
    *, streaming: bool = False,
) -> dict[str, Any]:
    observed.prepare(case)
    messages = generation_messages(case)
    started = time.perf_counter()
    result = None
    error = None
    deltas: list[str] = []
    cancel_event = Event()
    try:
        if streaming:
            for event in generator.stream_response(messages, cancel_event=cancel_event):
                if isinstance(event, ChatGenerationDelta):
                    deltas.append(event.delta)
                elif isinstance(event, ChatGenerationResult):
                    result = event
        else:
            result = generator.generate_response(messages)
    except Exception:  # noqa: BLE001 - artifacts/logs must not copy sensitive exception bodies
        result = None
        error = "Assistant generation failed; final diagnostics unavailable"
    finally:
        cancel_event.set()
    if result is None:
        error = "Assistant generation failed; final diagnostics unavailable"
    fake = isinstance(observed.provider, ScriptedProvider)
    checks = grade(case, result, observed.calls, config=generator.config)
    if streaming:
        checks.append({
            "name": "stream_reconstruction", "group": "streaming",
            "passed": result is not None and "".join(deltas) == result.response, "detail": None,
        })
    return {
        "id": case.id, "category": case.category, "scenario": case.scenario,
        "messages": [{"role": item.role, "content": item.content} for item in messages],
        "expectations": case.expectations.model_dump(),
        "response": result.response if result is not None else None,
        "model": result.model if result is not None else observed.model_name,
        "latency_ms": (0 if fake else round((time.perf_counter() - started) * 1000))
            if result is None else result.latency_ms,
        "input_tokens": result.input_tokens if result is not None else None,
        "output_tokens": result.output_tokens if result is not None else None,
        "metrics_kind": "synthetic" if fake else "runtime",
        "validator": result.validator if result is not None else None,
        "tools": result.tools if result is not None else None,
        "provider_calls": len(observed.calls) - observed.injected_outputs,
        "injected_outputs": observed.injected_outputs,
        "error": error, "checks": checks,
        "deterministic_pass": all(check["passed"] for check in checks),
        "human_review": {
            "required": True, "status": "pending", "rubric": case.human_review,
            "flags": ["subjective_quality_not_automatically_scored"]
                + (["fake_output_not_quality_evidence"] if fake else [])
                + (["controlled_fault_injection"] if case.scenario != "natural" else [])
                + (["final_diagnostics_unavailable"] if result is None else [])
                + (["code_only_content_not_mechanically_verified"] if result is not None and any(
                    check.get("actual") == "unfenced_unverified"
                    for check in result.validator.get("final_validation", {}).get("checks", [])
                ) else []),
            "overall_pass": None, "notes": "",
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rates(selected):
        passed = sum(row["deterministic_pass"] for row in selected)
        return {"total": len(selected), "passed": passed, "pass_rate": passed / len(selected)}

    def failed_groups(groups):
        return sum(any(not check["passed"] and check["group"] in groups for check in row["checks"])
                   for row in rows)

    return {
        "total_cases": len(rows), "deterministic": rates(rows),
        "by_category": {
            category: rates(selected) for category in get_args(Category)
            if (selected := [row for row in rows if row["category"] == category])
        },
        "mechanical_constraint_failures": failed_groups({"mechanical"}),
        "tool_failures": failed_groups({"tools"}),
        "failed_tool_attempts": sum(
            tool.get("success") is False for row in rows for tool in (row["tools"] or [])
        ),
        "identity_failures": failed_groups({"identity"}),
        "memory_context_failures": failed_groups({"memory", "context"}),
        "generation_failures": sum(row["error"] is not None for row in rows),
        "cases_requiring_human_review": [row["id"] for row in rows],
        "interpretation": (
            "Pass rates measure only declared deterministic checks, not overall text quality. "
            "Missing final responses fail applicable checks; failure cause/attempt metadata may "
            "be unavailable. Failed tool attempts include recovered injected faults."
        ),
    }


def run(
    *, mode: str = "fake", output_dir: str | Path, cases_path: str | Path = DEFAULT_CASES,
    ids: Sequence[str] | None = None, streaming: bool = False,
) -> Path:
    cases = load_cases(cases_path)
    if ids is not None:
        if not ids or set(ids) - {case.id for case in cases}:
            raise ValueError("Unknown or empty case selection")
        cases = [case for case in cases if case.id in ids]
    destination = Path(output_dir)
    # Never overwrite historical benchmark or human-review artifacts implicitly.
    destination.mkdir(parents=True, exist_ok=False)
    generator, observed = build_runtime(mode)
    manifest = {
        "suite": "aevon_text_quality_v1", "schema_version": 1, "mode": mode,
        "streaming": streaming, "status": "running", "source_revision": git_revision(),
        "case_sha256": stable_fingerprint([case.model_dump() for case in cases]),
        "model_settings": generator.config.model, "generation_settings": generator.config.generation,
        "production_prompt_sha256": hashlib.sha256(
            generator.config.runtime_system_prompt.encode("utf-8")
        ).hexdigest(),
        "case_ids": [case.id for case in cases],
        "notice": "Synthetic inputs; fake outputs are harness checks, not model-quality evidence.",
    }
    rows = []
    try:
        write_json(destination / "run.json", manifest)
        for case in cases:
            row = evaluate_case(case, generator, observed, streaming=streaming)
            append_jsonl(destination / "results.jsonl", row)
            rows.append(row)
        write_json(destination / "summary.json", summarize(rows))
        manifest["status"] = "complete"
        write_json(destination / "run.json", manifest)
    finally:
        observed.close()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fake", "transformers", "remote"), default="fake")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    try:
        destination = run(
            mode=args.mode, output_dir=args.output_dir, cases_path=args.cases,
            ids=args.ids, streaming=args.stream,
        )
    except Exception:  # noqa: BLE001 - no prompt/config/token details in CLI errors
        parser.exit(2, "Benchmark could not complete; check configuration and use a new output directory.\n")
    print(f"Benchmark artifacts written to {destination}")


if __name__ == "__main__":
    main()
