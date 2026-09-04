"""CPU repair contracts and killer fixtures; fake answers are not quality evidence."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from backend.chat_service import (
    ChatGenerationDelta,
    ChatGenerationError,
    ChatGenerationResult,
    RemoteProjection,
)
from backend.chat_service import GenerationMessage as Message
from evaluation import aevon_text_quality as quality
from evaluation.constraints import (
    build_retry_prompt,
    parse_constraints,
    validate_repair_literals,
    validate_response,
    validate_with_bounded_retries,
)
from evaluation.hf_backend import GenerationOutput
from runtime.config import load_production_runtime_config
from runtime.generator import ProviderChatGenerator
from tests.test_remote_privacy import RemoteHarness

CASES = quality.DEFAULT_CASES.with_name("aevon_epistemic_killer_v5_2.jsonl")
CAPITAL = "What is the capital of France? Answer in exactly three words."
TOOL = '<tool_call>{"name":"calculator","arguments":{"expression":"17*83"}}</tool_call>'


def run_response(generator, messages, streaming):
    if not streaming:
        return generator.generate_response(messages)
    events = list(generator.stream_response(messages, cancel_event=Event()))
    assert isinstance(events[-1], ChatGenerationResult)
    assert "".join(e.delta for e in events if isinstance(e, ChatGenerationDelta)) == events[-1].response
    return events[-1]


class Provider(quality.ScriptedProvider):
    def __init__(self, outputs):
        self.responses = iter(outputs)
        self.calls = []

    def generate(self, messages, generation_config):
        self.calls.append(([dict(message) for message in messages], dict(generation_config)))
        return GenerationOutput(next(self.responses), 10, 5)


def test_killer_cases_unique_small_and_v1_citation_derived():
    cases = quality.load_cases(CASES)
    old = quality.load_cases()
    assert len(cases) == len({c.id for c in cases}) == 12
    assert len(old) == 54 and not {c.id for c in old} & {c.id for c in cases}
    assert all(c.scenario == "natural" and c.expectations.epistemic_guardrail is None for c in cases)
    assert cases[4].messages == next(c for c in old if c.id == "uncertainty_citation").messages
    assert cases[5].fake_responses == ["The capital is Paris.", "Paris is capital."]


@pytest.mark.parametrize("streaming", [False, True])
def test_killer_fake_suite_and_exact_call_counts(tmp_path, monkeypatch, streaming):
    def forbidden(*args, **kwargs):
        pytest.fail("Killer fake suite must not construct real inference or a database")

    monkeypatch.setattr(quality, "select_response_generator", forbidden)
    monkeypatch.setattr(quality, "LocalTransformersInferenceProvider", forbidden)
    monkeypatch.setattr(quality, "RemoteInferenceProvider", forbidden)
    output = quality.run(cases_path=CASES, output_dir=tmp_path / "killer", streaming=streaming)
    rows = [json.loads(line) for line in (output / "results.jsonl").read_bytes().splitlines()]
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["total_cases"] == summary["deterministic"]["passed"] == 12
    assert summary["generation_failures"] == summary["tool_failures"] == 0
    assert sum(row["provider_calls"] for row in rows) == 13
    for row in rows:
        repair = row["id"] == "v52_capital_repair"
        assert row["provider_calls"] == 1 + int(repair)
        assert row["validator"]["retry_attempted"] is repair
        assert row["validator"]["retry_count"] == int(repair)
        assert ("first_validation" in row["validator"]) is repair
        assert row["tools"] == [] and row["injected_outputs"] == 0
        assert row["human_review"]["overall_pass"] is None
        assert "fake_output_not_quality_evidence" in row["human_review"]["flags"]


def test_repair_prompt_carries_exact_request_candidate_and_each_failed_constraint():
    prompt = "Use exactly 4 words in exactly 2 bullets."
    original = "- one"
    validation = validate_response(original, parse_constraints(prompt))
    repair = build_retry_prompt(prompt, original, validation)
    assert f"Original user request:\n{prompt}" in repair
    assert f"Previous answer:\n{original}" in repair
    assert all(failure in repair for failure in validation["failures"])
    for term in ("FORMAT REPAIR", "do not solve", "entities", "numbers", "dates", "polarity",
                 "negation", "comparison direction", "uncertainty", "qualifications",
                 "quoted literal identifiers", "tool-derived values", "subject/object",
                 "Do not add new facts", "strengthen/weaken uncertainty", "Do not call tools"):
        assert term in repair
    assert "as much as possible" not in repair


@pytest.mark.parametrize("streaming", [False, True])
def test_capital_repair_provider_uses_original_proposition_and_preservation_instruction(streaming):
    class InstructionSensitiveProvider(Provider):
        def generate(self, messages, generation_config):
            if self.calls:
                prompt = messages[-1]["content"]
                assert f"Original user request:\n{CAPITAL}" in prompt
                assert "Previous answer:\nThe capital is Paris." in prompt
                assert "Expected exactly 3 words, but the answer contained 4 words." in prompt
                assert "Preserve factual subject/object relationships" in prompt
                assert "reverse relationships" in prompt
            return super().generate(messages, generation_config)

    provider = InstructionSensitiveProvider(["The capital is Paris.", "Paris is capital."])
    generator = ProviderChatGenerator(load_production_runtime_config(), provider=provider)
    result = run_response(generator, [Message("user", CAPITAL)], streaming)
    assert result.response == "Paris is capital."
    assert len(provider.calls) == 2
    assert result.input_tokens == 20 and result.output_tokens == 10
    assert result.validator["first_validation"]["passed"] is False
    assert result.validator["first_validation"]["checks"][0]["actual"] == 4
    assert result.validator["retry_attempted"] is True
    assert result.validator["retry_count"] == 1
    assert result.validator["retry_passed"] is result.validator["first_retry_passed"] is True
    assert result.validator["final_validation"]["checks"][0]["actual"] == 3
    assert result.validator["repair_safety"] == {"passed": True, "failures": []}
    assert "The capital is Paris." not in json.dumps(result.validator)
    assert provider.calls[1][0][:-1] == provider.calls[0][0][:-1]
    assert provider.calls[0][1] == provider.calls[1][1] == generator.config.generation


@pytest.mark.parametrize("bad", ["Capital of Paris.", "France is capital.", "Paris is France."])
def test_killer_expectations_reject_known_relationship_corruption(bad):
    # Deliberately a SUITE check, not a Paris-specific runtime heuristic.
    case = quality.load_cases(CASES)[5]
    generator, observed = quality.build_runtime("fake")
    case.fake_responses[-1] = bad
    row = quality.evaluate_case(case, generator, observed)
    assert row["provider_calls"] == 2
    assert not row["deterministic_pass"]
    assert any(not check["passed"] and check["name"].startswith("not_contains")
               for check in row["checks"])
    assert row["human_review"]["overall_pass"] is None


@pytest.mark.parametrize("streaming", [False, True])
def test_passing_candidate_has_identical_context_config_and_no_repair_metadata(streaming, monkeypatch):
    config = load_production_runtime_config()
    provider = Provider(["Paris is capital."])
    generator = ProviderChatGenerator(config, provider=provider)
    messages = [Message("user", CAPITAL)]
    expected = generator._compile_context(messages)
    monkeypatch.setattr("evaluation.constraints.build_retry_prompt",
                        lambda *args: pytest.fail("Passing candidate built a repair prompt"))
    monkeypatch.setattr("evaluation.constraints.validate_repair_literals",
                        lambda *args: pytest.fail("Passing candidate ran repair safety"))
    result = run_response(generator, messages, streaming)
    assert provider.calls == [(expected.messages, config.generation)]
    assert result.validator == {
        "retry_attempted": False, "retry_passed": None, "retry_count": 0,
        "parsed_constraints": parse_constraints(CAPITAL),
        "final_validation": validate_response(result.response, parse_constraints(CAPITAL)),
    }


@pytest.mark.parametrize("original,repaired,code", [
    ("The result is 1411.", "Result is 1412.", "repair_changed_numeric_literals"),
    ("The result is 1411.", "Unknown result here.", "repair_changed_numeric_literals"),
    ("Due on 2026-09-04.", "Due 2026-09-05.", "repair_changed_numeric_literals"),
    ("Use `POSTGRES_URL` here.", "Use `MYSQL_URL`.", "repair_changed_quoted_identifiers"),
    ("No, it cannot.", "Yes, it can.", "repair_reversed_leading_polarity"),
    ("Yes, it can.", "No, it cannot.", "repair_reversed_leading_polarity"),
])
def test_obvious_literal_mutations_fail_without_copying_evidence(original, repaired, code):
    safety = validate_repair_literals(original, repaired)
    assert safety == {"passed": False, "failures": [code]}


@pytest.mark.parametrize("original,repaired", [
    ("The result is 1411.", "Result is 1411."),
    ("Due on 2026-09-04.", "Due 2026-09-04."),
    ("Use `POSTGRES_URL` here.", "Use `POSTGRES_URL`."),
    ("No, it cannot.", "No, impossible."),
    ("- Red\n- Blue", "1. Red\n2. Blue"),
])
def test_unchanged_literals_and_list_numbering_pass(original, repaired):
    assert validate_repair_literals(original, repaired) == {"passed": True, "failures": []}


def test_failed_literal_repair_is_recorded_as_failure_despite_passing_word_count():
    result = validate_with_bounded_retries(
        "Answer in exactly 3 words.", "The result is 1411.", lambda _: "Result is 1412.",
    )
    assert result["retry_count"] == 1
    assert result["final_validation"]["checks"][0]["passed"] is True
    assert result["final_validation"]["passed"] is result["retry_passed"] is False
    assert result["repair_safety"]["failures"] == ["repair_changed_numeric_literals"]
    assert result["retry_attempts"][0]["repair_safety"] == result["repair_safety"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("repaired", ["Result is 1411.", "Result is 1412.", TOOL])
def test_tool_result_not_reexecuted_or_silently_changed(streaming, repaired, monkeypatch):
    provider = Provider([TOOL, "The result is 1411.", repaired])
    generator = ProviderChatGenerator(load_production_runtime_config(), provider=provider)
    execute = generator._tool_registry.execute
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return execute(*args, **kwargs)

    monkeypatch.setattr(generator._tool_registry, "execute", counted)
    messages = [Message("user", "What is 17 * 83? Use the calculator. Answer in exactly 3 words.")]
    if repaired == "Result is 1411.":
        result = run_response(generator, messages, streaming)
        assert result.response == repaired and result.tools[0]["result"] == "1411"
        assert result.input_tokens == 30 and result.output_tokens == 15
    else:
        with pytest.raises(ChatGenerationError, match="Assistant generation failed"):
            run_response(generator, messages, streaming)
    assert len(provider.calls) == 3 and len(calls) == 1
    prompt = provider.calls[2][0][-1]["content"]
    assert '"result": "1411"' in prompt
    assert "do not recompute" in prompt and "Previous answer:\nThe result is 1411." in prompt
    assert provider.calls[2][0][:-1] == provider.calls[0][0][:-1]


@pytest.mark.parametrize("streaming", [False, True])
def test_remote_repair_uses_only_already_projected_context(streaming):
    harness = RemoteHarness(["The result is 1411.", "Result is 1411."])
    try:
        messages = [
            Message("user", "PRIVATE_HISTORY_CANARY", RemoteProjection(None)),
            Message("user", "PRIVATE_REQUEST_CANARY", RemoteProjection("Answer in exactly 3 words.")),
        ]
        # Mechanical constraints are parsed from the raw request; retain them
        # there while projecting away the private content of the user turn.
        messages[-1] = replace(messages[-1], content="PRIVATE_REQUEST_CANARY. Answer in exactly 3 words.")
        result = run_response(harness.generator, messages, streaming)
        assert result.response == "Result is 1411."
        assert len(harness.calls) == 2
        assert "PRIVATE_" not in json.dumps(harness.calls)
        assert harness.calls[1]["messages"][:-1] == harness.calls[0]["messages"][:-1]
        assert "Original user request:\nAnswer in exactly 3 words." in harness.calls[1]["messages"][-1]["content"]
    finally:
        harness.provider.close()


def test_current_v1_schema_and_all_nonfake_case_content_stay_frozen():
    rows = [json.loads(line) for line in quality.DEFAULT_CASES.read_bytes().splitlines()]
    assert len(rows) == 54
    for row in rows:
        row.pop("fake_responses")
    content = json.dumps(rows, sort_keys=True).encode()
    assert hashlib.sha256(content).hexdigest() == "2e541ab813198681e2ed1e27dfbe9f998497c1769421431ff322a122ba47304e"


@pytest.mark.parametrize("path,digest", [
    ("configs/production_runtime.yaml", "7b57ba7eba26185da1bb04e08481f7c1896afcc4a0ceb647140b0ab82b0fa845"),
    ("configs/baseline_eval_v2_constrained.yaml", "f2f437e3adafb31b3974e0a78c4f5f7fd1d298e45a18b1d8770233d42663a2bd"),
    ("runtime/epistemic.py", "060d36257fddd43fe47d4c51f411f71d1486b78a5322cbd4f6ea5486fcba90ca"),
    ("runtime/context.py", "082a8191fc7143a2659154fc2780ec0e24581c17e744467ac445c0d8489ff49e"),
])
def test_frozen_prompt_config_guards_and_compiler(path, digest):
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest() == digest
