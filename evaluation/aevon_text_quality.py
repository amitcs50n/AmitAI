"""Production-path text measurement: deterministic checks plus pending human review.

Fake responses are harness fixtures, never quality evidence or provider input.
No database, memory mutations, judge model, or inference defaults are introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from threading import Event
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from backend.chat_service import ChatGenerationDelta, ChatGenerationResult, GenerationMessage
from backend.memory import MAX_MEMORY_CONTEXT_CHARS, MAX_RETRIEVED_MEMORIES, format_memory_context
from evaluation.baseline import load_jsonl, stable_fingerprint
from evaluation.constraints import parse_constraints, validate_response
from evaluation.context_layouts import LAYOUTS, Layout, LayoutProvider, layout_messages
from evaluation.hf_backend import GenerationOutput
from evaluation.run_baseline import git_revision
from evaluation.text_quality_storage import (
    RunArtifactError,
    atomic_json,
    durable_append,
    exclusive_run,
    flush_durable,
    read_manifest,
    read_results,
    truncate_torn_tail,
)
from runtime.app import select_response_generator
from runtime.calculator import CalculatorTool
from runtime.config import (
    DEFAULT_RUNTIME_CONFIG_PATH,
    RuntimeConfig,
    load_production_runtime_config,
)
from runtime.context import MAX_HISTORY_CONTEXT_CHARS, MAX_HISTORY_MESSAGES, compile_model_messages
from runtime.generator import ProviderChatGenerator
from runtime.privacy import InferenceExecutionScope
from runtime.providers import (
    InferenceProvider,
    LocalTransformersInferenceProvider,
    RemoteInferenceProvider,
)
from runtime.tooling import ToolRegistry

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "eval/aevon_text_quality_v1.jsonl"
Category = Literal[
    "identity", "conversation", "judgment", "technical", "reasoning", "continuity",
    "memory", "tone", "format", "tools", "uncertainty", "long_context",
    "ambiguous_reference", "insufficient_evidence", "false_premise", "contradiction",
    "continuity_no_invention", "trusted_memory_fidelity", "missing_history",
    "unsupported_nonexistence", "hypothesis_vs_fact", "technical_no_schema_invention",
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

    def __init__(self, provider: InferenceProvider, *, context_layout: Layout | None = None) -> None:
        self.provider = provider
        self.provider_name = provider.provider_name
        self.model_name = provider.model_name
        self.execution_scope = provider.execution_scope
        self.calls: list[list[dict[str, str]]] = []
        self.inject_fault = False
        self.injected_outputs = 0
        self.context_layout = context_layout

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


def build_runtime(
    mode: str, *, config: RuntimeConfig | None = None,
    context_layout: Layout | None = None,
) -> tuple[ProviderChatGenerator, ObservedProvider]:
    if context_layout is not None and context_layout not in LAYOUTS:
        raise ValueError("Unknown experimental context layout")

    def adapt(observed):
        return observed if context_layout is None else LayoutProvider(observed, context_layout)

    if mode == "fake":
        observed = ObservedProvider(ScriptedProvider(), context_layout=context_layout)
        return ProviderChatGenerator(
            config or load_production_runtime_config(), provider=adapt(observed), clock=lambda: 0.0,
        ), observed

    providers: list[ObservedProvider] = []

    def local_factory(config: RuntimeConfig) -> ProviderChatGenerator:
        observed = ObservedProvider(LocalTransformersInferenceProvider(
            config.model, int(config.generation["seed"]),
        ), context_layout=context_layout)
        providers.append(observed)
        return ProviderChatGenerator(config, provider=adapt(observed))

    def remote_factory(**kwargs):
        observed = ObservedProvider(RemoteInferenceProvider(**kwargs), context_layout=context_layout)
        providers.append(observed)
        return adapt(observed)

    if mode not in {"transformers", "remote"}:
        raise ValueError("Explicit benchmark mode must be fake, transformers, or remote")
    # Existing startup owns settings, environment selection, transport and privacy.
    generator = select_response_generator(
        mode=mode, generator_factory=local_factory, remote_provider_factory=remote_factory,
        config_path=config.source_path if config is not None else None,
    )
    if not isinstance(generator, ProviderChatGenerator):
        raise TypeError("Benchmark requires the production provider-backed generator")
    if config is not None and generator.config != config:
        providers[0].close()
        raise RunArtifactError("Benchmark configuration changed during startup")
    return generator, providers[0]


def grade(
    case: QualityCase,
    result: ChatGenerationResult | None,
    calls: list[list[dict[str, str]]],
    *,
    config: RuntimeConfig,
    context_layout: Layout | None = None,
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
    if context_layout is not None:
        canonical = compile_model_messages(
            generation_messages(case), runtime_system_prompt=config.runtime_system_prompt,
            tool_instructions=ToolRegistry([CalculatorTool()]).instructions(),
            execution_scope=InferenceExecutionScope.LOCAL,
        )
        matches = initial == layout_messages(canonical, context_layout)
        add("experimental_layout_matches", "context", matches)
        # Existing order/limit checks describe canonical compilation. Normalize
        # only after exact equality proves the requested layout reached the provider.
        # The raw `calls` below still grade content across every tool/retry request.
        initial = canonical if matches else []
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
    checks = grade(case, result, observed.calls, config=generator.config,
                   context_layout=observed.context_layout)
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
        return {"total": len(selected), "passed": passed,
                "pass_rate": passed / len(selected) if selected else None}

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


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("backend", "evaluation", "runtime"):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
            digest.update(b"\0")
    return digest.hexdigest()


def _requested_manifest(
    cases, mode: str, streaming: bool, config: RuntimeConfig, *, suite: str,
    context_layout: Layout | None = None,
) -> dict[str, Any]:
    revision = git_revision()
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RunArtifactError("Benchmark requires an identifiable source revision")
    return {
        "suite": suite, "schema_version": 2, "mode": mode,
        **({"experimental_context_layout": context_layout} if context_layout is not None else {}),
        "streaming": streaming, "status": "running", "source_revision": revision,
        "source_code_sha256": _code_fingerprint(),
        "case_sha256": stable_fingerprint([case.model_dump() for case in cases]),
        "model_settings": config.model, "generation_settings": config.generation,
        "production_prompt_sha256": hashlib.sha256(
            config.runtime_system_prompt.encode("utf-8")
        ).hexdigest(),
        "case_ids": [case.id for case in cases], "expected_total_cases": len(cases),
        "notice": "Synthetic inputs; fake outputs are harness checks, not model-quality evidence.",
    }


def _validate_manifest(existing: dict[str, Any], requested: dict[str, Any]) -> None:
    if existing.get("suite") != requested["suite"] or type(existing.get("schema_version")) is not int:
        raise RunArtifactError("Unsupported benchmark run manifest")
    if existing.get("status") == "complete":
        raise RunArtifactError("Benchmark run is already complete")
    if existing["schema_version"] != 2:
        raise RunArtifactError("Unsupported benchmark run schema; start a new run")
    if existing.get("status") not in ("running", "incomplete"):
        raise RunArtifactError("Benchmark run is not incomplete")
    normalized = {**existing, "status": "running"}
    if stable_fingerprint(normalized) != stable_fingerprint(requested):
        raise RunArtifactError("Benchmark resume invocation or source does not match the original run")


Nonnegative = Annotated[int, Field(ge=0)]


class _SavedCheck(StrictModel):
    name: Text
    group: Text
    passed: bool
    detail: Any


class _SavedReview(StrictModel):
    required: Literal[True]
    status: Text
    rubric: list[Text]
    flags: list[Text]
    overall_pass: bool | None
    notes: str


class _SavedTool(StrictModel):
    attempt: Annotated[int, Field(ge=1)]
    name: str | None
    success: bool
    arguments: dict[str, Any] | None = None
    result: str | None = None
    error: dict[str, str] | None = None


class _SavedResult(StrictModel):
    id: Text
    category: Category
    scenario: Literal["natural", "forced_malformed_tool_call"]
    messages: list[dict[str, str]]
    expectations: Expectations
    response: Text | None
    model: Text
    latency_ms: Nonnegative | None
    input_tokens: Nonnegative | None
    output_tokens: Nonnegative | None
    metrics_kind: Literal["synthetic", "runtime"]
    validator: dict[str, Any] | None
    tools: list[_SavedTool] | None
    provider_calls: Nonnegative
    injected_outputs: Nonnegative
    error: str | None
    checks: list[_SavedCheck] = Field(min_length=1)
    deterministic_pass: bool
    human_review: _SavedReview


def _validate_saved_rows(
    rows, cases, *, config: RuntimeConfig, mode: str, streaming: bool,
    context_layout: Layout | None = None,
) -> None:
    for index, row in enumerate(rows):
        try:
            _SavedResult.model_validate(row)
        except ValueError:
            raise RunArtifactError("Invalid benchmark result schema") from None
        case = cases[index]
        immutable = {
            "id": case.id, "category": case.category, "scenario": case.scenario,
            "messages": [{"role": item.role, "content": item.content}
                         for item in generation_messages(case)],
            "expectations": case.expectations.model_dump(),
            "model": "fake-scripted" if mode == "fake" else str(config.model["name"]),
            "metrics_kind": "synthetic" if mode == "fake" else "runtime",
        }
        if stable_fingerprint({key: row[key] for key in immutable}) != stable_fingerprint(immutable):
            raise RunArtifactError("Benchmark results must match the exact case prefix and definitions")
        if row["human_review"]["rubric"] != case.human_review:
            raise RunArtifactError("Benchmark review rubric does not match the case definition")
        expected_checks = [(check["name"], check["group"]) for check in grade(
            case, None, [], config=config, context_layout=context_layout,
        )]
        if streaming:
            expected_checks.append(("stream_reconstruction", "streaming"))
        if [(check["name"], check["group"]) for check in row["checks"]] != expected_checks:
            raise RunArtifactError("Benchmark result checks do not match the case definition")
        if row["deterministic_pass"] != all(check["passed"] for check in row["checks"]):
            raise RunArtifactError("Benchmark result has inconsistent deterministic statistics")
        if (row["response"] is None) != (row["error"] is not None):
            raise RunArtifactError("Benchmark result has inconsistent completion status")


def _progress_summary(rows, expected_total: int) -> dict[str, Any]:
    return {
        **summarize(rows),
        "expected_total_cases": expected_total, "completed_cases": len(rows),
        "remaining_cases": expected_total - len(rows),
        "status": "complete" if len(rows) == expected_total else "incomplete",
        "statistics_scope": "completed_cases_only",
    }


def run(
    *, mode: str = "fake", output_dir: str | Path, cases_path: str | Path = DEFAULT_CASES,
    ids: Sequence[str] | None = None, streaming: bool = False, resume: bool = False,
    context_layout: Layout | None = None,
) -> Path:
    if context_layout is not None and context_layout not in LAYOUTS:
        raise ValueError("Unknown experimental context layout")
    cases = load_cases(cases_path)
    if ids is not None:
        if not ids or len(set(ids)) != len(ids) or set(ids) - {case.id for case in cases}:
            raise ValueError("Unknown, duplicate or empty case selection")
        by_id = {case.id: case for case in cases}
        cases = [by_id[case_id] for case_id in ids]
    destination = Path(output_dir)
    if destination.is_symlink():
        raise RunArtifactError("Benchmark output must be a local directory")
    if resume:
        if not destination.is_dir():
            raise RunArtifactError("Resume requires an existing benchmark run directory")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    if mode not in {"fake", "transformers", "remote"}:
        raise RunArtifactError("Unsupported benchmark mode")
    # Read settings only: no provider construction, transport, engine or database.
    config = load_production_runtime_config() if mode == "fake" else load_production_runtime_config(
        os.getenv("AMITAI_RUNTIME_CONFIG", str(DEFAULT_RUNTIME_CONFIG_PATH)),
    )
    requested = _requested_manifest(cases, mode, streaming, config, suite=Path(cases_path).stem,
                                    context_layout=context_layout)
    if resume:
        # Give content-free compatibility/completed errors even for pre-lock legacy runs.
        # Recheck under the exclusive lease before trusting any recoverable progress.
        _validate_manifest(read_manifest(destination / "run.json"), requested)
    with exclusive_run(destination, resume=resume):
        if resume:
            _validate_manifest(read_manifest(destination / "run.json"), requested)
            rows, torn_offset = read_results(destination / "results.jsonl", expected_count=len(cases))
            _validate_saved_rows(rows, cases, config=config, mode=mode, streaming=streaming,
                                 context_layout=context_layout)
            if torn_offset is not None:
                truncate_torn_tail(destination / "results.jsonl", torn_offset)
            print(f"Resuming {len(rows)}/{len(cases)} completed cases.")
        else:
            rows = []
            with (destination / "results.jsonl").open("xb") as handle:
                flush_durable(handle)
            atomic_json(destination / "run.json", requested)
        # Results are authoritative if a crash left summary/manifest behind.
        atomic_json(destination / "summary.json", _progress_summary(rows, len(cases)))
        if len(rows) < len(cases):
            _run_pending(cases, rows, destination, mode=mode, streaming=streaming, config=config,
                         context_layout=context_layout)
        atomic_json(destination / "run.json", {**requested, "status": "complete"})
    return destination


def _run_pending(cases, rows, destination, *, mode, streaming, config, context_layout=None) -> None:
    generator, observed = build_runtime(mode, config=config, context_layout=context_layout)
    try:
        for case in cases[len(rows):]:
            row = evaluate_case(case, generator, observed, streaming=streaming)
            durable_append(destination / "results.jsonl", row)
            rows.append(row)
            atomic_json(destination / "summary.json", _progress_summary(rows, len(cases)))
            print(f"[{len(rows)}/{len(cases)}] {case.id} complete")
    finally:
        observed.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fake", "transformers", "remote"), default="fake")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--context-layout", choices=LAYOUTS,
                        help="Opt-in evaluation experiment only; production stays unchanged")
    args = parser.parse_args()
    try:
        destination = run(
            mode=args.mode, output_dir=args.output_dir, cases_path=args.cases,
            ids=args.ids, streaming=args.stream, resume=args.resume,
            context_layout=args.context_layout,
        )
    except RunArtifactError as exc:
        parser.exit(2, f"{exc}\n")
    except Exception:  # noqa: BLE001 - no prompt/config/token details in CLI errors
        parser.exit(2, "Benchmark could not complete; check configuration and use a new output directory.\n")
    print(f"Benchmark artifacts written to {destination}")


if __name__ == "__main__":
    main()
