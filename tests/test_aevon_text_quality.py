"""Benchmark harness tests: no GPU, database, external inference, or judge model."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import get_args

import httpx
import pytest
from pydantic import ValidationError

from backend.chat_service import ChatGenerationDelta, ChatGenerationResult
from evaluation import aevon_text_quality as quality
from evaluation.constraints import parse_constraints, validate_response
from evaluation.hf_backend import GenerationOutput
from runtime.config import EXPECTED_MODEL_NAME, load_production_runtime_config
from runtime.context import MAX_HISTORY_CONTEXT_CHARS, MAX_HISTORY_MESSAGES


def case_by_id(case_id):
    return next(case for case in quality.load_cases() if case.id == case_id)


def checks_by_name(checks):
    return {check["name"]: check for check in checks}


def test_dataset_schema_unique_ids_categories_and_coverage():
    cases = quality.load_cases()
    assert len(cases) == 54
    assert len({case.id for case in cases}) == 54
    assert {case.category for case in cases} == set(get_args(quality.Category))
    assert all(case.human_review for case in cases)
    assert all(case.messages[-1].role == "user" for case in cases)
    assert {constraint["type"] for case in cases
            for constraint in parse_constraints(case.messages[-1].content)} == {
        "exact_words", "exact_bullets", "at_most_bullets", "code_only",
    }


@pytest.mark.parametrize("field,value", [
    ("id", "bad id"), ("id", ""), ("category", "unknown"), ("messages", []),
    ("messages", [{"role": "system", "content": "untrusted"}]),
    ("messages", [{"role": "assistant", "content": "no current user"}]),
    ("messages", [{"role": "user", "content": "   "}]),
    ("messages", [{"role": "user", "content": 10}]),
    ("expectations", {"mechanical": "yes"}),
    ("expectations", {"mechanical": True}),
    ("expectations", {"tools": "browser"}),
    ("expectations", {"unknown_rule": True}),
    ("expectations", {"contains": [""]}),
    ("human_review", []), ("fake_responses", []),
    ("scenario", "forced_malformed_tool_call"),
    ("history_fixture", "unbounded"), ("unknown_field", "invalid"),
    ("memory", [{"category": "secrets", "key": "a", "value": "b"}]),
    ("memory", [{"category": "profile", "key": "a", "value": "b", "id": "internal"}]),
    ("memory", [{"category": "profile", "key": "a", "value": "b"}] * 9),
    ("memory", [{"category": "profile", "key": "a", "value": "x" * 4001}]),
])
def test_malformed_cases_fail(field, value):
    payload = case_by_id("identity_name").model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        quality.QualityCase.model_validate(payload)


@pytest.mark.parametrize("content", ["", "{bad json}\n", "[]\n", "{}\n"])
def test_bad_dataset_files_fail(tmp_path, content):
    path = tmp_path / "cases.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        quality.load_cases(path)


def test_duplicate_case_ids_fail(tmp_path):
    row = case_by_id("identity_name").model_dump_json()
    path = tmp_path / "cases.jsonl"
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        quality.load_cases(path)


@pytest.mark.parametrize("text,passed", [
    ("Aevon", True), ("I'm Aevon.", True), ("My name is Aevon!", True),
    ("AmitAI", False), ("Qwen", False), ("Not Aevon", False),
    ("Aevon is not my name", False),
])
def test_assistant_identity_check(text, passed):
    checks = quality.grade(case_by_id("identity_name"), ChatGenerationResult(text), [],
                           config=load_production_runtime_config())
    assert checks_by_name(checks)["assistant_identity"]["passed"] is passed


def test_model_identity_uses_configured_identifier():
    config = load_production_runtime_config()
    configured = replace(config, model={**config.model, "name": "Test/ConfiguredModel"})
    for text, passed in [(EXPECTED_MODEL_NAME, False), ("Test/ConfiguredModel", True)]:
        checks = quality.grade(case_by_id("identity_configured_model"), ChatGenerationResult(text), [],
                               config=configured)
        assert checks_by_name(checks)["configured_model_identity"]["passed"] is passed


@pytest.mark.parametrize("text,contains,forbidden", [
    ("Use IS NULL", True, True), ("use is null", True, True),
    ("Use equality", False, True), ("IS NULL. AS AN AI...", True, False),
])
def test_contains_and_forbidden_text_are_case_insensitive(text, contains, forbidden):
    case = case_by_id("technical_sql_null")
    case.expectations.not_contains = ["as an AI"]
    checks = checks_by_name(quality.grade(case, ChatGenerationResult(text), [],
                                        config=load_production_runtime_config()))
    assert checks["contains_0"]["passed"] is contains
    assert checks["not_contains_0"]["passed"] is forbidden


@pytest.mark.parametrize("tools,required,forbidden", [
    ([], False, True),
    ([{"name": "calculator", "success": True}], True, False),
    ([{"name": "calculator", "success": False}], False, False),
    ([{"name": None, "success": False}], False, False),
])
def test_tool_used_means_success_and_not_used_means_no_attempt(tools, required, forbidden):
    for case_id, expected in [("tools_calculate", required), ("tools_no_need", forbidden)]:
        checks = quality.grade(case_by_id(case_id), ChatGenerationResult("1411", tools=tools), [],
                               config=load_production_runtime_config())
        assert checks_by_name(checks)["tool_usage"]["passed"] is expected


@pytest.mark.parametrize("text", [
    "Prose <tool_call>{}</tool_call>", "<tool_result>1411</tool_result>",
    "prose </tool_call>", "MEMORY_CONTEXT_V1", "<tool_", "<memory_context>{}",
])
def test_protocol_leakage_fails(text):
    checks = quality.grade(case_by_id("tools_no_need"), ChatGenerationResult(text), [],
                           config=load_production_runtime_config())
    assert not checks_by_name(checks)["no_protocol_leakage"]["passed"]


@pytest.mark.parametrize("case_id,text", [
    ("format_words", "Paris is the capital"),
    ("format_exact_bullets", "- One\n- Two"),
    ("format_at_most_bullets", "- One\n- Two\n- Three"),
    ("format_code_only", "Here is the code:\n```python\nprint(1)\n```"),
    ("format_one_item", "- Apple\n- Pear"),
])
def test_grader_delegates_to_existing_mechanical_validator(case_id, text):
    case = case_by_id(case_id)
    expected = validate_response(text, parse_constraints(case.messages[-1].content))
    checks = quality.grade(case, ChatGenerationResult(text), [],
                           config=load_production_runtime_config())
    actual = checks_by_name(checks)["mechanical_constraints"]
    assert actual["detail"] == expected
    assert not actual["passed"]


@pytest.mark.parametrize("streaming", [False, True])
def test_all_fake_cases_use_production_orchestration_and_pass(streaming):
    generator, observed = quality.build_runtime("fake")
    rows = [quality.evaluate_case(case, generator, observed, streaming=streaming)
            for case in quality.load_cases()]
    assert all(row["deterministic_pass"] for row in rows)
    assert all(row["metrics_kind"] == "synthetic" and row["latency_ms"] == 0 for row in rows)
    assert all(row["human_review"]["overall_pass"] is None for row in rows)
    assert all(row["human_review"]["status"] == "pending" for row in rows)
    summary = quality.summarize(rows)
    assert summary["deterministic"]["pass_rate"] == 1.0
    assert summary["failed_tool_attempts"] == 1  # recovered controlled fault
    assert summary["tool_failures"] == 0
    assert len(summary["cases_requiring_human_review"]) == 54
    code_row = next(row for row in rows if row["id"] == "format_code_only")
    assert "code_only_content_not_mechanically_verified" in code_row["human_review"]["flags"]


@pytest.mark.parametrize("streaming", [False, True])
def test_controlled_malformed_tool_recovery_is_labeled_and_sanitized(streaming):
    generator, observed = quality.build_runtime("fake")
    row = quality.evaluate_case(case_by_id("tools_recovery"), generator, observed, streaming=streaming)
    assert row["scenario"] == "forced_malformed_tool_call"
    assert row["injected_outputs"] == 1 and row["provider_calls"] == 2
    assert [attempt["success"] for attempt in row["tools"]] == [False, True]
    assert "controlled_fault_injection" in row["human_review"]["flags"]
    assert quality.MALFORMED_TOOL_FIXTURE not in json.dumps(row)
    assert "invalid_tool_call" in observed.calls[1][-2]["content"]
    assert '"success":false' in observed.calls[1][-1]["content"]
    assert '"result":"1411"' in observed.calls[2][-1]["content"]


@pytest.mark.parametrize("streaming", [False, True])
def test_retry_and_tool_followup_keep_minimized_context(streaming):
    payload = case_by_id("tools_constrained_final").model_dump()
    payload.update(history_fixture="char_window", messages=[
        {"role": "user", "content": "Recent relevant RECENT_CONTEXT_CANARY"},
        {"role": "assistant", "content": "Acknowledged."},
        payload["messages"][-1],
    ])
    payload["expectations"].update(
        context_contains=["RECENT_CONTEXT_CANARY"], context_not_contains=["OLD_CONTEXT_CANARY_00"],
    )
    case = quality.QualityCase.model_validate(payload)
    generator, observed = quality.build_runtime("fake")
    row = quality.evaluate_case(case, generator, observed, streaming=streaming)
    assert row["deterministic_pass"]
    assert row["validator"]["retry_count"] == 1
    assert row["response"] == "It is 1411."
    assert len(observed.calls) == 4
    for call in observed.calls:
        assert call[0]["content"].startswith(generator.config.runtime_system_prompt)
        assert "RECENT_CONTEXT_CANARY" in repr(call)
        assert "OLD_CONTEXT_CANARY_00" not in repr(call)


@pytest.mark.parametrize("streaming", [False, True])
def test_missing_final_response_is_not_success_and_does_not_stop_run(streaming):
    case = case_by_id("format_words")
    case.fake_responses = ["Paris is the capital"] * 3
    generator, observed = quality.build_runtime("fake")
    failed = quality.evaluate_case(case, generator, observed, streaming=streaming)
    assert failed["response"] is None and not failed["deterministic_pass"]
    assert failed["validator"] is None and failed["tools"] is None
    assert "Paris is the capital" not in json.dumps(failed)
    assert len(observed.calls) == 3
    good = quality.evaluate_case(case_by_id("identity_name"), generator, observed, streaming=streaming)
    summary = quality.summarize([failed, good])
    assert summary["generation_failures"] == 1
    assert summary["mechanical_constraint_failures"] == 1
    assert summary["deterministic"]["pass_rate"] == 0.5
    assert summary["by_category"]["format"]["pass_rate"] == 0.0


def test_provider_exceptions_do_not_copy_bodies_to_logs_or_artifacts(monkeypatch, capsys, caplog):
    generator, observed = quality.build_runtime("fake")

    def fail(*args, **kwargs):
        raise RuntimeError("UNSAFE_EXCEPTION_BODY_CANARY")

    monkeypatch.setattr(observed.provider, "generate", fail)
    row = quality.evaluate_case(case_by_id("identity_name"), generator, observed)
    assert row["error"] and row["response"] is None
    assert "UNSAFE_EXCEPTION_BODY_CANARY" not in json.dumps(row) + caplog.text + capsys.readouterr().out


@pytest.mark.parametrize("raises", [False, True])
def test_incomplete_stream_is_a_generation_failure_without_a_partial_artifact(monkeypatch, raises):
    generator, observed = quality.build_runtime("fake")

    def incomplete(messages, *, cancel_event):
        yield ChatGenerationDelta("PARTIAL_OUTPUT_CANARY")
        if raises:
            raise RuntimeError("PRIVATE_ERROR_CANARY")

    monkeypatch.setattr(generator, "stream_response", incomplete)
    row = quality.evaluate_case(case_by_id("identity_name"), generator, observed, streaming=True)
    assert row["response"] is None and row["error"] is not None
    assert not row["deterministic_pass"]
    assert "PARTIAL_OUTPUT_CANARY" not in json.dumps(row)
    assert "PRIVATE_ERROR_CANARY" not in json.dumps(row)
    assert quality.summarize([row])["generation_failures"] == 1


def test_fake_artifacts_byte_deterministic_and_no_inference_or_database(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Fake benchmark must not create model, network, or database")

    monkeypatch.setattr(quality, "select_response_generator", forbidden)
    monkeypatch.setattr(quality, "LocalTransformersInferenceProvider", forbidden)
    monkeypatch.setattr(quality, "RemoteInferenceProvider", forbidden)
    monkeypatch.setattr("sqlalchemy.create_engine", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setenv("AMITAI_INFERENCE_PROVIDER", "remote")
    monkeypatch.setenv("AMITAI_DB_KEY", "DB_SECRET_CANARY")
    monkeypatch.setenv("AMITAI_LOCAL_API_TOKEN", "LOCAL_SECRET_CANARY")
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", "REMOTE_SECRET_CANARY")
    first = quality.run(output_dir=tmp_path / "first")
    second = quality.run(output_dir=tmp_path / "second")
    for name in ("run.json", "results.jsonl", "summary.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert "SECRET_CANARY" not in (first / name).read_text(encoding="utf-8")
    manifest = json.loads((first / "run.json").read_text())
    assert manifest["mode"] == "fake" and manifest["status"] == "complete"
    assert manifest["model_settings"] == load_production_runtime_config().model
    with pytest.raises(FileExistsError):
        quality.run(output_dir=first)
    with pytest.raises(ValueError, match="selection"):
        quality.run(output_dir=tmp_path / "unknown", ids=["unknown_id"])


@pytest.mark.parametrize("mode", ["transformers", "remote"])
@pytest.mark.parametrize("streaming", [False, True])
def test_real_mode_wiring_uses_production_profile_and_only_generation_inputs(
    mode, streaming, monkeypatch,
):
    captured = []
    config = load_production_runtime_config()
    remote_class = quality.RemoteInferenceProvider
    local_class = quality.LocalTransformersInferenceProvider

    class Engine:
        def generate_detailed(self, messages, generation_config):
            captured.append({"messages": messages, "generation_config": generation_config})
            return GenerationOutput("Juniper, using PostgreSQL.", 15, 4)

        def stream_detailed(self, messages, generation_config, *, cancel_event):
            output = self.generate_detailed(messages, generation_config)
            yield "Juniper, "
            yield "using PostgreSQL."
            yield output

    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert request.headers["Authorization"] == "Bearer benchmark_transport_token_0123456789"
        if streaming:
            assert request.url.path == "/v1/generate/stream"
            events = [
                ("delta", {"delta": "Juniper, "}),
                ("delta", {"delta": "using PostgreSQL."}),
                ("final", {"request_id": payload["request_id"], "model": EXPECTED_MODEL_NAME,
                           "text": "Juniper, using PostgreSQL.",
                           "input_tokens": 15, "output_tokens": 4}),
                ("done", {}),
            ]
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text="".join(
                f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in events
            ))
        assert request.url.path == "/v1/generate"
        return httpx.Response(200, json={
            "request_id": payload["request_id"], "model": EXPECTED_MODEL_NAME,
            "text": "Juniper, using PostgreSQL.", "input_tokens": 15, "output_tokens": 4,
        })

    monkeypatch.setattr(quality, "LocalTransformersInferenceProvider", lambda model, seed: local_class(
        model, seed, engine_factory=lambda *args: Engine(),
    ))
    monkeypatch.setattr(quality, "RemoteInferenceProvider", lambda **kwargs: remote_class(
        **kwargs, transport=httpx.MockTransport(handler), resolver=lambda *args: ["8.8.8.8"],
    ))
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_URL", "https://gpu.example")
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS", "https://gpu.example")
    monkeypatch.setenv("AMITAI_REMOTE_INFERENCE_TOKEN", "benchmark_transport_token_0123456789")
    generator, observed = quality.build_runtime(mode)
    try:
        row = quality.evaluate_case(case_by_id("context_order_orphan"), generator, observed,
                                    streaming=streaming)
        assert row["deterministic_pass"], row
        assert row["model"] == EXPECTED_MODEL_NAME
        assert row["metrics_kind"] == "runtime"
        assert captured[0]["generation_config"] == config.generation
        messages = captured[0]["messages"]
        assert messages[0]["content"].startswith(config.runtime_system_prompt)
        assert messages[1]["content"].startswith("MEMORY_CONTEXT_V1\n")
        assert messages[2]["role"] == "user" and "RECENT_CONTEXT_CANARY" in messages[2]["content"]
        assert messages[-1]["content"] == "Which release and SQL dialect are we discussing?"
        body = json.dumps(captured)
        for forbidden in ("ORPHAN_CONTEXT_CANARY", "fake_responses", "human_review", "expectations",
                          "benchmark_transport_token_0123456789", "owner_id", "database_url"):
            assert forbidden not in body
        assert set(captured[0]) <= {"messages", "generation_config", "request_id"}
    finally:
        observed.close()


def test_long_context_fixtures_cross_production_limits_and_preserve_input():
    assert (MAX_HISTORY_MESSAGES, MAX_HISTORY_CONTEXT_CHARS) == (20, 20_000)
    generator, observed = quality.build_runtime("fake")
    for case_id in ("context_recent_count", "context_recent_chars", "context_order_orphan", "context_oversized"):
        case = case_by_id(case_id)
        before = case.model_dump_json()
        messages = quality.generation_messages(case)
        if case_id == "context_recent_chars":
            assert len(messages[:-1]) > MAX_HISTORY_MESSAGES
            assert sum(len(message.content) for message in messages[:-1]) > MAX_HISTORY_CONTEXT_CHARS
        row = quality.evaluate_case(case, generator, observed)
        assert row["deterministic_pass"]
        assert case.model_dump_json() == before
        assert observed.calls[0][-1]["content"] == case.messages[-1].content
        if case_id == "context_oversized":
            assert len(observed.calls[0]) == 2


def test_memory_checks_are_only_literal_evidence_and_fail_on_leak():
    case = case_by_id("memory_irrelevant")
    checks = checks_by_name(quality.grade(case, ChatGenerationResult("Use reverse. Orchid"), [],
                                        config=load_production_runtime_config()))
    assert not checks["memory_not_contains_0"]["passed"]
    assert case.human_review  # Literal pass alone cannot establish natural use or non-leakage.


@pytest.mark.parametrize("path,blob_id", [
    ("configs/baseline_eval.yaml", "e4a9a1ad33173587b77543edb80be55047dda553"),
    ("configs/baseline_eval_v2.yaml", "9faae6a220867969c4200242e9d883b57defc313"),
    ("configs/baseline_eval_v2_constrained.yaml", "3d4910c7f4437e6aa75021aeb7f50b6af3668bd0"),
    ("eval/behavior_v1.jsonl", "75371426e340c4bd9fccfa32bdea7893520bbf9e"),
    ("evaluation/baseline.py", "602d371376fc1828894a8fb319352f7251afcd71"),
    ("evaluation/constraints.py", "61a041e3f21de9466f195191429067903f88f28a"),
    ("evaluation/run_baseline.py", "c92cf92be64890ffea8961611670beec4cd95387"),
    ("evaluation/summarize.py", "f94260effd77929f0b6691d24a99a838b905f0b7"),
])
def test_frozen_baseline_assets_remain_unchanged(path, blob_id):
    # Git blobs from baseline 8586073; allow only checkout CRLF normalization.
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(content)).encode() + b"\0" + content
    assert hashlib.sha1(blob, usedforsecurity=False).hexdigest() == blob_id
