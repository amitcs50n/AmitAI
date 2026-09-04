"""V2 calibration fixtures and harness contracts; scripted outputs are not quality evidence."""

import json
from collections import Counter
from typing import get_args

import httpx
import pytest
from pydantic import ValidationError

from backend.chat_service import ChatGenerationResult
from backend.memory import format_memory_context
from evaluation import aevon_text_quality as quality

V2_CASES = quality.DEFAULT_CASES.with_name("aevon_epistemic_regression_v2.jsonl")
CATEGORY_COUNTS = {
    "ambiguous_reference": 3,
    "insufficient_evidence": 2,
    "false_premise": 3,
    "contradiction": 2,
    "continuity_no_invention": 2,
    "trusted_memory_fidelity": 2,
    "missing_history": 3,
    "unsupported_nonexistence": 2,
    "hypothesis_vs_fact": 2,
    "technical_no_schema_invention": 3,
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("V2 harness checks must not construct model, network, or database")

    monkeypatch.setattr(quality, "select_response_generator", forbidden)
    monkeypatch.setattr(quality, "LocalTransformersInferenceProvider", forbidden)
    monkeypatch.setattr(quality, "RemoteInferenceProvider", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr("sqlalchemy.create_engine", forbidden)


def case_by_id(case_id):
    return next(case for case in quality.load_cases(V2_CASES) if case.id == case_id)


def test_v2_schema_unique_ids_categories_and_calibration_controls():
    cases = quality.load_cases(V2_CASES)
    v1 = quality.load_cases()
    assert len(cases) == len({case.id for case in cases}) == 24
    assert not {case.id for case in cases} & {case.id for case in v1}
    assert Counter(case.category for case in cases) == CATEGORY_COUNTS
    assert {case.category for case in v1 + cases} == set(get_args(quality.Category))
    # Each category has a positive control; most also have missing/false-evidence probes.
    controls = [case for case in cases if case.id.endswith("_control")]
    assert len(controls) == 10
    assert {case.category for case in controls} == set(CATEGORY_COUNTS)
    assert all(len(case.human_review) >= 2 and case.scenario == "natural" for case in cases)
    assert all(case.messages[-1].role == "user" for case in cases)
    assert all(case.expectations.tools in {"required", "forbidden"} for case in cases)
    assert {case.history_fixture for case in cases} == {
        None, "count_window", "char_window", "oversized",
    }


@pytest.mark.parametrize("field,value", [
    ("category", "epistemic_guess"),
    ("expectations", {"root_cause_regex": ".*"}),
    ("expectations", {"mechanical": "true"}),
    ("human_review", []),
    ("fake_responses", []),
    ("messages", [{"role": "assistant", "content": "No user turn"}]),
    ("history_fixture", "invented_summary"),
    ("memory", [{"category": "project", "key": "database", "value": 123}]),
])
def test_v2_malformed_cases_rejected_by_shared_loader(tmp_path, field, value):
    row = case_by_id("ambiguous_move_target").model_dump()
    row[field] = value
    path = tmp_path / "bad-v2.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        quality.load_cases(path)


def test_v2_duplicate_ids_rejected(tmp_path):
    row = case_by_id("ambiguous_move_target").model_dump_json()
    path = tmp_path / "duplicates.jsonl"
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        quality.load_cases(path)


@pytest.mark.parametrize("streaming", [False, True])
def test_v2_fake_run_artifacts_and_review_status(tmp_path, streaming):
    directory = quality.run(cases_path=V2_CASES, output_dir=tmp_path / "v2", streaming=streaming)
    manifest = json.loads((directory / "run.json").read_bytes())
    summary = json.loads((directory / "summary.json").read_bytes())
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_bytes().splitlines()]
    assert manifest["suite"] == "aevon_epistemic_regression_v2"
    assert manifest["mode"] == "fake" and manifest["status"] == "complete"
    assert manifest["case_ids"] == [case.id for case in quality.load_cases(V2_CASES)]
    assert summary["total_cases"] == summary["deterministic"]["passed"] == 24
    assert summary["generation_failures"] == summary["failed_tool_attempts"] == 0
    assert {key: value["total"] for key, value in summary["by_category"].items()} == CATEGORY_COUNTS
    assert len(summary["cases_requiring_human_review"]) == 24
    for row in rows:
        assert row["metrics_kind"] == "synthetic" and row["latency_ms"] == 0
        assert row["human_review"]["status"] == "pending"
        assert row["human_review"]["overall_pass"] is None
        assert "fake_output_not_quality_evidence" in row["human_review"]["flags"]


@pytest.mark.parametrize("case_id", [
    "ambiguous_move_target", "continuity_courier_unknown", "memory_database_and_config",
    "memory_user_correction_control", "history_first_message_count", "history_removed_exchange",
    "history_recent_fact_control",
])
def test_v2_context_contains_only_supplied_trusted_context_and_retained_history(case_id):
    case = case_by_id(case_id)
    before = case.model_dump_json()
    source = quality.generation_messages(case)
    generator, observed = quality.build_runtime("fake")
    row = quality.evaluate_case(case, generator, observed)
    assert row["deterministic_pass"]
    compiled = observed.calls[0] if observed.calls else generator._model_messages(source)
    if not observed.calls:
        assert row["validator"]["epistemic_guardrail"]["provider_bypassed"] is True
    assert compiled[0]["content"].startswith(generator.config.runtime_system_prompt + "\n\n")
    assert compiled[-1] == {"role": "user", "content": case.messages[-1].content}
    supplied = [{"role": message.role, "content": message.content} for message in source]
    # No invented summaries, references, or recovered turns from case metadata/fake answers.
    assert all(message in supplied for message in compiled[1:])
    assert case.model_dump_json() == before
    if case.memory:
        assert compiled[1] == {
            "role": "system",
            "content": format_memory_context([item.model_dump() for item in case.memory]),
        }
        assert '"value":"PostgreSQL"' in compiled[1]["content"]
    if case.history_fixture:
        for removed in case.expectations.context_not_contains:
            assert any(removed in message.content for message in source)
            assert all(removed not in message["content"] for message in compiled)
    if case_id in {"ambiguous_move_target", "history_removed_exchange"}:
        assert len(compiled) == 2  # System and current user; no invented prior exchange.
    if not case.memory:
        assert all(message["role"] != "system" for message in compiled[1:])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("case_id", [
    "memory_database_and_config", "history_first_message_count",
    "history_removed_exchange", "history_recent_fact_control",
])
def test_v2_repairs_preserve_prompt_memory_and_do_not_restore_history(case_id, streaming):
    case = case_by_id(case_id)
    final = case.fake_responses[0]
    case.messages[-1].content += f" Answer in exactly {len(final.split())} words."
    case.expectations.mechanical = True
    case.fake_responses = ["Too short.", final]
    generator, observed = quality.build_runtime("fake")
    row = quality.evaluate_case(case, generator, observed, streaming=streaming)
    assert row["deterministic_pass"]
    if case_id == "history_first_message_count":
        # V5 refuses missing-history reconstruction before formatting repair.
        assert row["validator"]["epistemic_guardrail"]["kind"] == "missing_history"
        assert row["validator"]["retry_count"] == 0 and not observed.calls
        assert row["provider_calls"] == 0 and row["tools"] == []
        return
    assert row["validator"]["retry_count"] == 1
    assert len(observed.calls) == 2
    assert observed.calls[0][:-1] == observed.calls[1][:-1]
    assert case.messages[-1].content in observed.calls[1][-1]["content"]


@pytest.mark.parametrize("answer", ["Eighteen kilograms total.", "18 kilograms total."])
def test_v2_numeric_control_accepts_words_and_digits(answer):
    case = case_by_id("evidence_liquid_mass_control")
    case.fake_responses[-1] = answer
    generator, observed = quality.build_runtime("fake")
    row = quality.evaluate_case(case, generator, observed)
    assert row["deterministic_pass"]
    assert row["human_review"]["overall_pass"] is None


def test_semantic_correctness_is_not_claimed_by_literal_graders():
    case = case_by_id("memory_database_and_config")
    generator, observed = quality.build_runtime("fake")
    quality.evaluate_case(case, generator, observed)
    # Paraphrases/negations are not adjudicated with brittle required/forbidden words.
    # Even the wrong answer below needs human grading; a deterministic pass is not a verdict.
    for answer in (
        "Postgres; the configuration key hasn't been supplied.",
        "It uses PostgreSQL, not SQLite. I don't have its config key.",
        "It uses MySQL via the mysql config key.",
    ):
        checks = quality.grade(case, ChatGenerationResult(answer), observed.calls,
                               config=generator.config)
        assert all(check["passed"] for check in checks)
    assert case.human_review


def test_v2_resume_retains_suite_identity_and_completed_results(tmp_path, monkeypatch):
    directory = tmp_path / "resumed"
    evaluate = quality.evaluate_case
    completed = []

    def interrupt(case, *args, **kwargs):
        if completed:
            raise KeyboardInterrupt()
        completed.append(case.id)
        return evaluate(case, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(quality, "evaluate_case", interrupt)
        with pytest.raises(KeyboardInterrupt):
            quality.run(cases_path=V2_CASES, output_dir=directory)
    prefix = (directory / "results.jsonl").read_bytes()
    quality.run(cases_path=V2_CASES, output_dir=directory, resume=True)
    assert (directory / "results.jsonl").read_bytes().startswith(prefix)
    clean = quality.run(cases_path=V2_CASES, output_dir=tmp_path / "clean")
    for name in ("run.json", "results.jsonl", "summary.json"):
        assert (directory / name).read_bytes() == (clean / name).read_bytes()
