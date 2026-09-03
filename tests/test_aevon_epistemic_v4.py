"""V4 reuses the text runner, with explicit layout identity and human-only semantics."""

import json
from collections import Counter

import httpx
import pytest

from evaluation import aevon_text_quality as quality
from evaluation.context_layout_inspection import V4_CASES
from evaluation.context_layouts import LAYOUTS, LayoutProvider
from evaluation.text_quality_storage import RunArtifactError

COUNTS = {
    "trusted_memory_fidelity": 2, "missing_history": 2, "ambiguous_reference": 2,
    "technical_no_schema_invention": 2, "false_premise": 1, "unsupported_nonexistence": 1,
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("V4 tests must not construct real inference, network, or database")

    monkeypatch.setattr(quality, "select_response_generator", forbidden)
    monkeypatch.setattr(quality, "LocalTransformersInferenceProvider", forbidden)
    monkeypatch.setattr(quality, "RemoteInferenceProvider", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr("sqlalchemy.create_engine", forbidden)


def test_suite_schema_unique_ids_categories_and_human_grading():
    cases = quality.load_cases(V4_CASES)
    assert len(cases) == len({case.id for case in cases}) == 10
    assert Counter(case.category for case in cases) == COUNTS
    old = [*quality.load_cases()]
    for version in (2, 3):
        old.extend(quality.load_cases(V4_CASES.with_name(f"aevon_epistemic_regression_v{version}.jsonl")))
    assert not {case.id for case in cases} & {case.id for case in old}
    assert sum(case.id.endswith("_control") for case in cases) == 3
    for case in cases:
        assert len(case.human_review) >= 2
        assert case.scenario == "natural" and case.expectations.tools == "forbidden"
        assert not any(getattr(case.expectations, field) for field in (
            "contains", "not_contains", "memory_contains", "memory_not_contains", "mechanical",
        ))
    assert {case.history_fixture for case in cases if case.history_fixture} == {"count_window", "oversized"}


@pytest.mark.parametrize("change", ["duplicate", "category", "grader", "review"])
def test_malformed_suite_uses_existing_validation(tmp_path, change):
    row = quality.load_cases(V4_CASES)[0].model_dump()
    if change == "category":
        row["category"] = "layout_winner"
    elif change == "grader":
        row["expectations"] = {"layout_b_wins": True}
    elif change == "review":
        row["human_review"] = []
    path = tmp_path / "bad.jsonl"
    path.write_text((json.dumps(row) + "\n") * (2 if change == "duplicate" else 1), encoding="utf-8")
    with pytest.raises(ValueError):
        quality.load_cases(path)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("streaming", [False, True])
def test_fake_artifacts_all_layouts_same_cases_semantics_pending(tmp_path, layout, streaming):
    output = quality.run(cases_path=V4_CASES, output_dir=tmp_path / layout,
                         context_layout=layout, streaming=streaming)
    manifest = json.loads((output / "run.json").read_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_bytes().splitlines()]
    assert manifest["experimental_context_layout"] == layout
    assert manifest["status"] == "complete" and manifest["mode"] == "fake"
    assert summary["total_cases"] == summary["deterministic"]["passed"] == 10
    assert summary["generation_failures"] == summary["tool_failures"] == 0
    for row in rows:
        assert row["metrics_kind"] == "synthetic"
        assert row["human_review"]["status"] == "pending"
        assert row["human_review"]["overall_pass"] is None


def test_default_runner_remains_production_and_layout_does_not_change_configuration():
    default, observed = quality.build_runtime("fake")
    assert not isinstance(default._provider, LayoutProvider)
    assert observed.context_layout is None
    case = quality.load_cases(V4_CASES)[0]
    ordinary = quality.evaluate_case(case, default, observed)
    production_calls = observed.calls
    for layout in LAYOUTS:
        generator, selected = quality.build_runtime("fake", context_layout=layout)
        row = quality.evaluate_case(case, generator, selected)
        assert generator.config == default.config
        assert row["response"] == ordinary["response"]  # scripted fixtures only, not evidence
        assert row["deterministic_pass"]
        if layout == "A":
            assert selected.calls == production_calls


def test_context_grader_rejects_wrong_layout_or_missing_content():
    case = quality.load_cases(V4_CASES)[0]
    generator, observed = quality.build_runtime("fake", context_layout="B")
    quality.evaluate_case(case, generator, observed)
    for calls in ([observed.calls[0][1:]], [[*observed.calls[0], {"role": "system", "content": "extra"}]]):
        checks = quality.grade(case, None, calls, config=generator.config, context_layout="B")
        assert not next(check for check in checks if check["name"] == "experimental_layout_matches")["passed"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_resume_keeps_layout_and_rejects_mixing_before_provider(tmp_path, monkeypatch, layout):
    output = tmp_path / "partial"
    evaluate = quality.evaluate_case
    completed = []

    def interrupt(case, *args, **kwargs):
        if len(completed) == 1:
            raise KeyboardInterrupt
        completed.append(case.id)
        return evaluate(case, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(quality, "evaluate_case", interrupt)
        with pytest.raises(KeyboardInterrupt):
            quality.run(cases_path=V4_CASES, output_dir=output, context_layout=layout)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    for other in (None, *[candidate for candidate in LAYOUTS if candidate != layout]):
        with monkeypatch.context() as patch:
            patch.setattr(quality, "build_runtime", lambda *a, **k: pytest.fail("Provider constructed"))
            with pytest.raises(RunArtifactError, match="match"):
                quality.run(cases_path=V4_CASES, output_dir=output, context_layout=other, resume=True)
        assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    quality.run(cases_path=V4_CASES, output_dir=output, context_layout=layout, resume=True)
    assert json.loads((output / "summary.json").read_bytes())["deterministic"]["passed"] == 10


def test_invalid_layout_rejected_before_artifacts_or_provider(tmp_path):
    with pytest.raises(ValueError, match="layout"):
        quality.run(cases_path=V4_CASES, output_dir=tmp_path / "bad", context_layout="E")
    assert not (tmp_path / "bad").exists()
