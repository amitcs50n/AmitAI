"""V5 fake runs prove only harness and control flow, never model quality."""

import json
from dataclasses import replace
from threading import Event

import httpx
import pytest

from backend.chat_service import ChatGenerationDelta, ChatGenerationResult
from evaluation import aevon_text_quality as quality
from evaluation.text_quality_storage import RunArtifactError
from runtime.context import HISTORY_OMISSION_NOTICE
from tests.test_epistemic_guardrails import ENV_PROMPT, UNRELATED_ENV_HISTORY, RemoteHarness

V5_CASES = quality.DEFAULT_CASES.with_name("aevon_epistemic_guardrails_v5.jsonl")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("V5 harness must not construct real inference, network, or database")

    monkeypatch.setattr(quality, "select_response_generator", forbidden)
    monkeypatch.setattr(quality, "LocalTransformersInferenceProvider", forbidden)
    monkeypatch.setattr(quality, "RemoteInferenceProvider", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr("sqlalchemy.create_engine", forbidden)


def test_v5_small_explicit_diagnostic_suite():
    cases = quality.load_cases(V5_CASES)
    assert len(cases) == len({case.id for case in cases}) == 22
    assert all("epistemic_guardrail" in case.expectations.model_fields_set for case in cases)
    assert sum(case.expectations.epistemic_guardrail is not None for case in cases) == 11
    assert {case.expectations.epistemic_guardrail for case in cases} == {
        None, "missing_history", "ambiguous_reference", "unknown_internal_env_var",
    }
    assert all(case.scenario == "natural" for case in cases)
    old = quality.load_cases()
    for version in (2, 3, 4):
        old += quality.load_cases(V5_CASES.with_name(f"aevon_epistemic_regression_v{version}.jsonl"))
    assert not {case.id for case in old} & {case.id for case in cases}
    # Frozen suites retain the original expectation dictionaries on disk/resume.
    assert all("epistemic_guardrail" not in case.model_dump()["expectations"] for case in old)


@pytest.mark.parametrize("streaming", [False, True])
def test_v5_fake_suite_exact_bypass_and_control_calls(tmp_path, streaming):
    output = quality.run(cases_path=V5_CASES, output_dir=tmp_path / "v5", streaming=streaming)
    rows = [json.loads(line) for line in (output / "results.jsonl").read_bytes().splitlines()]
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["total_cases"] == summary["deterministic"]["passed"] == 22
    assert summary["generation_failures"] == summary["tool_failures"] == 0
    for row in rows:
        expected = row["expectations"]["epistemic_guardrail"]
        assert row["response"] and row["error"] is None
        assert row["provider_calls"] == (0 if expected else 1)
        assert row["injected_outputs"] == 0 and row["tools"] == []
        assert row["input_tokens"] == row["output_tokens"] == 0
        assert row["human_review"]["overall_pass"] is None
        assert "fake_output_not_quality_evidence" in row["human_review"]["flags"]
        context_check = next(c for c in row["checks"] if c["name"] == "context_source")
        if expected:
            assert row["validator"]["epistemic_guardrail"]["kind"] == expected
            assert context_check["detail"] == "compiled_locally_provider_bypassed"
        else:
            assert "epistemic_guardrail" not in row["validator"]
            assert context_check["detail"] == "provider_visible"


@pytest.mark.parametrize("mutation", [
    "kind", "provider", "input", "output", "tools", "empty", "metadata", "blank_response",
])
def test_grader_rejects_broken_guard_contract(mutation):
    case = quality.load_cases(V5_CASES)[0]
    generator, _ = quality.build_runtime("fake")
    result = generator.generate_response(quality.generation_messages(case))
    calls = []
    if mutation == "kind":
        result.validator["epistemic_guardrail"]["kind"] = "ambiguous_reference"
    elif mutation == "provider":
        calls = [generator._model_messages(quality.generation_messages(case))]
    elif mutation == "input":
        result = replace(result, input_tokens=1)
    elif mutation == "output":
        result = replace(result, output_tokens=1)
    elif mutation == "tools":
        result = replace(result, tools=[{"name": "calculator", "success": True}])
    elif mutation == "empty":
        result = None
    elif mutation == "metadata":
        result = replace(result, validator={})
    elif mutation == "blank_response":
        result = replace(result, response=" ")
    checks = quality.grade(case, result, calls, config=generator.config)
    assert not all(check["passed"] for check in checks)


def test_explicit_null_rejects_unexpected_guard_and_zero_calls():
    case = quality.load_cases(V5_CASES)[9]
    generator, _ = quality.build_runtime("fake")
    guarded = generator.generate_response(quality.generation_messages(quality.load_cases(V5_CASES)[6]))
    for result in (guarded, ChatGenerationResult("A provider-like answer")):
        checks = quality.grade(case, result, [], config=generator.config)
        assert not all(check["passed"] for check in checks)


def test_guarded_context_checks_still_fail_for_missing_notice_or_restored_canary():
    case = quality.load_cases(V5_CASES)[0]
    generator, observed = quality.build_runtime("fake")
    case.expectations.context_contains.append("This is not present")
    row = quality.evaluate_case(case, generator, observed)
    assert not row["deterministic_pass"] and row["provider_calls"] == 0
    case.expectations.context_contains = [HISTORY_OMISSION_NOTICE]
    case.expectations.context_not_contains.append(case.messages[-1].content)
    assert not quality.evaluate_case(case, generator, observed)["deterministic_pass"]


@pytest.mark.parametrize("streaming", [False, True])
def test_v5_resume_preserves_guard_rows_and_rejects_changed_code(tmp_path, monkeypatch, streaming):
    output = tmp_path / "partial"
    evaluate = quality.evaluate_case
    completed = []

    def interrupt(case, *args, **kwargs):
        if len(completed) == 3:
            raise KeyboardInterrupt
        completed.append(case.id)
        return evaluate(case, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(quality, "evaluate_case", interrupt)
        with pytest.raises(KeyboardInterrupt):
            quality.run(cases_path=V5_CASES, output_dir=output, streaming=streaming)
    prefix = (output / "results.jsonl").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(quality, "_code_fingerprint", lambda: "0" * 64)
        patch.setattr(quality, "build_runtime", lambda *a, **k: pytest.fail("Constructed provider"))
        with pytest.raises(RunArtifactError, match="match"):
            quality.run(cases_path=V5_CASES, output_dir=output, streaming=streaming, resume=True)
    assert (output / "results.jsonl").read_bytes() == prefix
    quality.run(cases_path=V5_CASES, output_dir=output, streaming=streaming, resume=True)
    clean = quality.run(cases_path=V5_CASES, output_dir=tmp_path / "clean", streaming=streaming)
    assert (output / "results.jsonl").read_bytes().startswith(prefix)
    for name in ("run.json", "results.jsonl", "summary.json"):
        assert (output / name).read_bytes() == (clean / name).read_bytes()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("history", [[], *UNRELATED_ENV_HISTORY[:2]])
def test_real_remote_provider_object_guarded_before_dns_or_http(streaming, history):
    # The actual provider is composed with mock transport; the offline fixture
    # independently forbids even HTTP client entry. No environment credentials.
    def forbidden(*args):
        pytest.fail("DNS invoked for guarded request")

    harness = RemoteHarness(resolver=forbidden)
    try:
        from backend.chat_service import GenerationMessage

        messages = [*history, GenerationMessage("user", ENV_PROMPT)]
        if streaming:
            events = list(harness.generator.stream_response(messages, cancel_event=Event()))
            assert isinstance(events[0], ChatGenerationDelta)
            result = events[-1]
        else:
            result = harness.generator.generate_response(messages)
        assert result.validator["epistemic_guardrail"]["provider_bypassed"]
        assert harness.calls == harness.bodies == harness.paths == []
    finally:
        harness.provider.close()
