"""V3 uses the existing harness; semantic success still requires human review."""

import json
from collections import Counter

import httpx
import pytest

from backend.memory import format_memory_context
from evaluation import aevon_text_quality as quality
from runtime.context import HISTORY_OMISSION_NOTICE

V3_CASES = quality.DEFAULT_CASES.with_name("aevon_epistemic_regression_v3.jsonl")
COUNTS = {
    "ambiguous_reference": 3,
    "trusted_memory_fidelity": 2,
    "missing_history": 3,
    "false_premise": 3,
    "unsupported_nonexistence": 2,
    "technical_no_schema_invention": 3,
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("V3 harness tests must not create inference, network, or database")

    monkeypatch.setattr(quality, "select_response_generator", forbidden)
    monkeypatch.setattr(quality, "LocalTransformersInferenceProvider", forbidden)
    monkeypatch.setattr(quality, "RemoteInferenceProvider", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr("sqlalchemy.create_engine", forbidden)


def test_v3_schema_ids_categories_and_conservative_grading():
    cases = quality.load_cases(V3_CASES)
    assert len(cases) == len({case.id for case in cases}) == 16
    assert Counter(case.category for case in cases) == COUNTS
    old = quality.load_cases() + quality.load_cases(V3_CASES.with_name("aevon_epistemic_regression_v2.jsonl"))
    assert not {case.id for case in cases} & {case.id for case in old}
    assert sum(case.id.endswith("_control") for case in cases) == 5
    for case in cases:
        assert len(case.human_review) >= 2 and case.messages[-1].role == "user"
        assert case.scenario == "natural"
        assert case.expectations.tools == "forbidden"
        # Do not grade agreement, negation, SQL meaning, or memory fidelity by keywords.
        assert not any(getattr(case.expectations, field) for field in (
            "contains", "not_contains", "memory_contains", "memory_not_contains",
        ))


@pytest.mark.parametrize("field,value", [
    ("category", "unknown_category"),
    ("expectations", {"semantic_success": True}),
    ("messages", [{"role": "system", "content": "untrusted system"}]),
    ("human_review", []),
    ("fake_responses", []),
])
def test_v3_malformed_definitions_use_shared_validation(tmp_path, field, value):
    row = quality.load_cases(V3_CASES)[0].model_dump()
    row[field] = value
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        quality.load_cases(path)


def test_v3_duplicate_ids_rejected(tmp_path):
    row = quality.load_cases(V3_CASES)[0].model_dump_json()
    path = tmp_path / "duplicates.jsonl"
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        quality.load_cases(path)


def test_v3_truncated_and_retained_fixtures_have_the_correct_signal_and_exact_memory():
    generator, observed = quality.build_runtime("fake")
    for case in quality.load_cases(V3_CASES):
        before = case.model_dump_json()
        row = quality.evaluate_case(case, generator, observed)
        assert row["deterministic_pass"], case.id
        compiled = observed.calls[0]
        assert compiled[0]["content"].count(HISTORY_OMISSION_NOTICE) == int(bool(case.history_fixture))
        if case.history_fixture:
            raw = quality.generation_messages(case)
            assert any("OLD_CONTEXT_CANARY_00" in item.content for item in raw)
            assert "OLD_CONTEXT_CANARY_00" not in repr(compiled)
        if case.memory:
            assert compiled[1] == {
                "role": "system",
                "content": format_memory_context([item.model_dump() for item in case.memory]),
            }
        assert compiled[-1] == {"role": "user", "content": case.messages[-1].content}
        assert case.model_dump_json() == before


def test_v3_open_closed_world_and_null_cases_have_distinct_evidence():
    cases = {case.id: case for case in quality.load_cases(V3_CASES)}
    open_case = cases["v3_open_world_sources"]
    closed = cases["v3_closed_world_inventory"]
    assert "nobody has ever published" in open_case.messages[-1].content
    assert "complete set of included adapters is {Larch, Willow}" in closed.messages[-1].content
    assert all(not case.memory for case in (open_case, closed))
    null_case = cases["v3_null_event_unknown"]
    assert "delivered_at IS NULL" in null_case.messages[-1].content
    assert "no rule that unrecorded deliveries are impossible" in null_case.messages[-1].content
    assert "real-world delivery event" in " ".join(null_case.human_review)


@pytest.mark.parametrize("streaming", [False, True])
def test_v3_fake_artifacts_complete_with_human_review_pending(tmp_path, streaming):
    output = quality.run(cases_path=V3_CASES, output_dir=tmp_path / "v3", streaming=streaming)
    manifest = json.loads((output / "run.json").read_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_bytes().splitlines()]
    assert manifest["suite"] == "aevon_epistemic_regression_v3"
    assert manifest["status"] == "complete" and manifest["mode"] == "fake"
    assert summary["total_cases"] == summary["deterministic"]["passed"] == 16
    assert summary["generation_failures"] == summary["tool_failures"] == 0
    assert {key: value["total"] for key, value in summary["by_category"].items()} == COUNTS
    for row in rows:
        assert row["metrics_kind"] == "synthetic"
        assert row["human_review"]["status"] == "pending"
        assert row["human_review"]["overall_pass"] is None
