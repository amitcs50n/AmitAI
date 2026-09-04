"""Crash/recovery tests use scripted providers only; no GPU, HTTP or database."""

import errno
import hashlib
import json

import httpx
import pytest

from evaluation import aevon_text_quality as quality
from evaluation import text_quality_storage as storage

IDS = ("identity_name", "tools_recovery", "format_words", "tools_constrained_final")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Resume tests must not construct real inference, HTTP or database")

    monkeypatch.setattr(quality, "select_response_generator", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr("sqlalchemy.create_engine", forbidden)
    monkeypatch.setattr(quality, "git_revision", lambda: "a" * 40)
    monkeypatch.setattr(quality, "_code_fingerprint", lambda: "b" * 64)


def read_json(path):
    return json.loads(path.read_bytes())


def rewrite_json(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def partial_run(directory, monkeypatch, *, completed=2, streaming=False, ids=IDS):
    evaluate = quality.evaluate_case
    calls = []

    def interrupt(case, *args, **kwargs):
        if len(calls) == completed:
            raise KeyboardInterrupt()
        calls.append(case.id)
        return evaluate(case, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(quality, "evaluate_case", interrupt)
        with pytest.raises(KeyboardInterrupt):
            quality.run(output_dir=directory, ids=ids, streaming=streaming)
    assert calls == list(ids[:completed])
    assert read_json(directory / "run.json")["status"] == "running"
    return directory


def forbid_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Provider must not be constructed before resume validation or for completed cases")

    monkeypatch.setattr(quality, "build_runtime", forbidden)


@pytest.mark.parametrize("streaming", [False, True])
def test_interruption_resumes_only_remaining_cases_and_matches_clean_run(tmp_path, monkeypatch, streaming, capsys):
    directory = partial_run(tmp_path / "resumed", monkeypatch, streaming=streaming)
    before = (directory / "results.jsonl").read_bytes()
    partial = read_json(directory / "summary.json")
    assert (partial["expected_total_cases"], partial["completed_cases"], partial["remaining_cases"]) == (4, 2, 2)
    assert partial["statistics_scope"] == "completed_cases_only"
    assert partial["status"] == "incomplete"
    assert partial["deterministic"]["total"] == 2
    assert partial["failed_tool_attempts"] == 1
    evaluate = quality.evaluate_case
    build = quality.build_runtime
    remaining = []
    providers = []

    def capture(case, *args, **kwargs):
        remaining.append(case.id)
        return evaluate(case, *args, **kwargs)

    def construct(*args, **kwargs):
        runtime = build(*args, **kwargs)
        providers.append(runtime[1])
        return runtime

    capsys.readouterr()
    with monkeypatch.context() as patch:
        patch.setattr(quality, "evaluate_case", capture)
        patch.setattr(quality, "build_runtime", construct)
        quality.run(output_dir=directory, ids=IDS, streaming=streaming, resume=True)
    assert remaining == list(IDS[2:])
    assert len(providers) == 1  # One runtime/provider per invocation, no per-case construction.
    assert (directory / "results.jsonl").read_bytes().startswith(before)
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_bytes().splitlines()]
    assert [row["id"] for row in rows] == list(IDS)
    assert sum(row["injected_outputs"] for row in rows) == 1
    assert read_json(directory / "run.json")["status"] == "complete"
    summary = read_json(directory / "summary.json")
    assert summary["completed_cases"] == 4 and summary["remaining_cases"] == 0
    assert summary["deterministic"]["passed"] == 4 and summary["failed_tool_attempts"] == 1
    assert capsys.readouterr().out.splitlines() == [
        "Resuming 2/4 completed cases.", "[3/4] format_words complete", "[4/4] tools_constrained_final complete",
    ]
    clean = quality.run(output_dir=tmp_path / "clean", ids=IDS, streaming=streaming)
    assert snapshot(directory) == snapshot(clean)


@pytest.mark.parametrize("mode,streaming,ids", [
    ("remote", False, IDS), ("transformers", False, IDS), ("fake", True, IDS),
    ("fake", False, IDS[:-1]), ("fake", False, tuple(reversed(IDS))),
])
def test_invocation_mismatch_fails_before_provider_or_writes(tmp_path, monkeypatch, mode, streaming, ids):
    directory = partial_run(tmp_path / "run", monkeypatch)
    before = snapshot(directory)
    forbid_provider(monkeypatch)
    with pytest.raises(storage.RunArtifactError, match="match"):
        quality.run(output_dir=directory, resume=True, mode=mode, streaming=streaming, ids=ids)
    assert snapshot(directory) == before


@pytest.mark.parametrize("field,value", [
    ("suite", "wrong_suite"), ("schema_version", 1), ("schema_version", True),
    ("status", "failed"), ("case_ids", list(reversed(IDS))),
    ("case_sha256", "0" * 64), ("production_prompt_sha256", "0" * 64),
    ("model_settings", {"name": "changed"}), ("generation_settings", {"max_new_tokens": 1}),
    ("source_revision", "c" * 40), ("source_code_sha256", "c" * 64),
    ("expected_total_cases", 3), ("streaming", 0), ("unexpected_field", True),
])
def test_manifest_mismatches_fail_closed(tmp_path, monkeypatch, field, value):
    directory = partial_run(tmp_path / "run", monkeypatch)
    manifest = read_json(directory / "run.json")
    manifest[field] = value
    rewrite_json(directory / "run.json", manifest)
    # Validation must precede even a permissible torn-tail repair.
    with (directory / "results.jsonl").open("ab") as handle:
        handle.write(b'{"id":"unfinished')
    before = snapshot(directory)
    forbid_provider(monkeypatch)
    with pytest.raises(storage.RunArtifactError):
        quality.run(output_dir=directory, resume=True, ids=IDS)
    assert snapshot(directory) == before


@pytest.mark.parametrize("kind", ["case_content", "case_order", "prompt", "model", "generation", "revision", "code"])
def test_current_source_or_configuration_changes_fail(tmp_path, monkeypatch, kind):
    directory = partial_run(tmp_path / "run", monkeypatch)
    before = snapshot(directory)
    kwargs = {}
    if kind.startswith("case"):
        cases = quality.load_cases()
        if kind == "case_content":
            cases[0].messages[0].content += " Changed."
        else:
            # Without --ids the file defines order; test that complete requested order differs.
            cases.reverse()
        path = tmp_path / "changed.jsonl"
        path.write_text("\n".join(case.model_dump_json() for case in cases), encoding="utf-8")
        kwargs["cases_path"] = path
        if kind == "case_order":
            kwargs["ids"] = tuple(reversed(IDS))
    elif kind in {"revision", "code"}:
        name = "git_revision" if kind == "revision" else "_code_fingerprint"
        monkeypatch.setattr(quality, name, lambda: "c" * (40 if kind == "revision" else 64))
    else:
        from dataclasses import replace

        config = quality.load_production_runtime_config()
        if kind == "prompt":
            config = replace(config, runtime_system_prompt="Changed identity")
        elif kind == "model":
            config = replace(config, model={**config.model, "revision": "changed"})
        else:
            config = replace(config, generation={**config.generation, "max_new_tokens": 10})
        monkeypatch.setattr(quality, "load_production_runtime_config", lambda *args: config)
    forbid_provider(monkeypatch)
    with pytest.raises(storage.RunArtifactError):
        quality.run(output_dir=directory, resume=True, **{"ids": IDS, **kwargs})
    assert snapshot(directory) == before


@pytest.mark.parametrize("kind", [
    "duplicate", "unknown", "nonprefix", "wrong_category", "wrong_messages", "wrong_expectations",
    "wrong_scenario", "wrong_review", "missing_field", "wrong_checks", "wrong_pass", "nonobject",
    "malformed_middle", "duplicate_json_key", "nonfinite", "blank_line",
])
def test_invalid_saved_results_are_not_modified_or_regenerated(tmp_path, monkeypatch, kind):
    directory = partial_run(tmp_path / "run", monkeypatch)
    path = directory / "results.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if kind == "duplicate":
        rows[1] = rows[0]
    elif kind == "unknown":
        rows[1]["id"] = "unknown_case"
    elif kind == "nonprefix":
        rows = rows[1:]
    elif kind == "wrong_category":
        rows[0]["category"] = "tone"
    elif kind == "wrong_messages":
        rows[0]["messages"][0]["content"] = "changed"
    elif kind == "wrong_expectations":
        rows[0]["expectations"]["contains"] = ["changed"]
    elif kind == "wrong_scenario":
        rows[0]["scenario"] = "forced_malformed_tool_call"
    elif kind == "wrong_review":
        rows[0]["human_review"]["rubric"] = ["changed"]
    elif kind == "missing_field":
        del rows[0]["response"]
    elif kind == "wrong_checks":
        rows[0]["checks"] = []
    elif kind == "wrong_pass":
        rows[0]["deterministic_pass"] = False
    elif kind == "nonobject":
        rows[0] = []
    if kind in {"malformed_middle", "duplicate_json_key", "nonfinite", "blank_line"}:
        lines[0] = {
            "malformed_middle": b'{"id":\n', "duplicate_json_key": b'{"id":"a","id":"b"}\n',
            "nonfinite": b'{"id":NaN}\n', "blank_line": b'\n',
        }[kind]
        path.write_bytes(b"".join(lines))
    else:
        path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    with path.open("ab") as handle:
        handle.write(b'{"id":"unfinished')
    before = snapshot(directory)
    forbid_provider(monkeypatch)
    with pytest.raises(storage.RunArtifactError):
        quality.run(output_dir=directory, resume=True, ids=IDS)
    assert snapshot(directory) == before


@pytest.mark.parametrize("tail", [
    b'{', b'{"id":', b'{"id":"format_words","response":"torn',
    b'{"id":"format_words","response":"caf\xc3',
    b'{"a":[true,nu', b'{"a":"\\u00', b'{"a":1e+',
])
def test_torn_final_line_is_discarded_without_touching_valid_prefix(tmp_path, monkeypatch, tail):
    directory = partial_run(tmp_path / "run", monkeypatch)
    path = directory / "results.jsonl"
    prefix = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(tail)
    repaired = []
    truncate = quality.truncate_torn_tail

    def repair(file, offset):
        assert offset == len(prefix)
        truncate(file, offset)
        assert file.read_bytes() == prefix
        repaired.append(offset)

    monkeypatch.setattr(quality, "truncate_torn_tail", repair)
    quality.run(output_dir=directory, resume=True, ids=IDS)
    assert repaired == [len(prefix)]
    assert path.read_bytes().startswith(prefix)
    assert [json.loads(line)["id"] for line in path.read_bytes().splitlines()] == list(IDS)


@pytest.mark.parametrize("tail", [
    b'{"a":oops', b'{"a":1,}', b'{"a":01', b'{"a":"\\q', b'{"a":"\\u0q',
    b'{"a":1,"a":', b'{"a":NaN', b'{"a":1e999,', b'{"a":\xff',
    b'{"a":"\\u00\xc3', b'{"a":\xc3', b'[]', b'[', b'{}garbage',
    b'{"a":\n', b'{"a":"torn\n\n', b'\n', b'   ',
])
def test_ambiguous_or_malformed_tail_fails_without_repair(tmp_path, monkeypatch, tail):
    directory = partial_run(tmp_path / "run", monkeypatch)
    with (directory / "results.jsonl").open("ab") as handle:
        handle.write(tail)
    before = snapshot(directory)
    forbid_provider(monkeypatch)
    with pytest.raises(storage.RunArtifactError):
        quality.run(output_dir=directory, resume=True, ids=IDS)
    assert snapshot(directory) == before


@pytest.mark.parametrize("ascii_only", [False, True])
def test_every_byte_prefix_of_nested_valid_json_is_recognized_as_torn(ascii_only):
    raw = json.dumps({
        "id": "case", "text": 'café 😀 \\ \"\t',
        "nested": [True, False, None, -12.5e-12, {}, [0]],
    }, ensure_ascii=ascii_only, separators=(",", ":")).encode()
    assert not storage.is_torn_object(raw)
    for position in range(1, len(raw)):
        assert storage.is_torn_object(raw[:position]), (position, raw[:position])


def test_valid_final_row_without_newline_is_preserved(tmp_path, monkeypatch):
    directory = partial_run(tmp_path / "run", monkeypatch)
    path = directory / "results.jsonl"
    prefix = path.read_bytes().removesuffix(b"\n")
    path.write_bytes(prefix)
    quality.run(output_dir=directory, resume=True, ids=IDS)
    assert path.read_bytes().startswith(prefix + b"\n")
    assert len(path.read_bytes().splitlines()) == len(IDS)


def test_final_summary_crash_recovers_without_any_inference(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    write = quality.atomic_json

    def crash(path, document):
        if path.name == "run.json" and document["status"] == "complete":
            raise KeyboardInterrupt()
        write(path, document)

    with monkeypatch.context() as patch:
        patch.setattr(quality, "atomic_json", crash)
        with pytest.raises(KeyboardInterrupt):
            quality.run(output_dir=directory, ids=IDS)
    prefix = (directory / "results.jsonl").read_bytes()
    assert read_json(directory / "run.json")["status"] == "running"
    # A stale/broken summary is derived data, not evidence of missing completed cases.
    (directory / "summary.json").write_bytes(b'{torn summary')
    forbid_provider(monkeypatch)
    quality.run(output_dir=directory, resume=True, ids=IDS)
    assert (directory / "results.jsonl").read_bytes() == prefix
    assert read_json(directory / "run.json")["status"] == "complete"
    assert read_json(directory / "summary.json")["completed_cases"] == len(IDS)


def test_append_is_authoritative_when_partial_summary_write_crashes(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    write = quality.atomic_json

    def crash(path, document):
        if path.name == "summary.json" and document["completed_cases"] == 1:
            raise KeyboardInterrupt()
        write(path, document)

    with monkeypatch.context() as patch:
        patch.setattr(quality, "atomic_json", crash)
        with pytest.raises(KeyboardInterrupt):
            quality.run(output_dir=directory, ids=IDS)
    assert len((directory / "results.jsonl").read_bytes().splitlines()) == 1
    assert read_json(directory / "summary.json")["completed_cases"] == 0
    quality.run(output_dir=directory, resume=True, ids=IDS)
    assert read_json(directory / "summary.json")["completed_cases"] == 4


def test_zero_progress_resume_and_midstream_interrupt_do_not_save_partial_text(tmp_path, monkeypatch):
    directory = tmp_path / "run"

    def interrupted_stream(*args, **kwargs):
        yield "PARTIAL_STREAM_CANARY"
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(quality.ScriptedProvider, "stream", interrupted_stream)
        with pytest.raises(KeyboardInterrupt):
            quality.run(output_dir=directory, ids=IDS[:1], streaming=True)
    assert (directory / "results.jsonl").read_bytes() == b""
    summary = read_json(directory / "summary.json")
    assert summary["completed_cases"] == 0 and summary["deterministic"]["pass_rate"] is None
    assert b"PARTIAL_STREAM_CANARY" not in repr(snapshot(directory)).encode()
    quality.run(output_dir=directory, resume=True, ids=IDS[:1], streaming=True)
    assert read_json(directory / "summary.json")["completed_cases"] == 1


def test_recorded_failure_and_human_notes_are_preserved(tmp_path, monkeypatch):
    cases = [case for case in quality.load_cases() if case.id in {"format_words", "tools_no_need"}]
    cases[0].fake_responses = ["Paris is the capital"] * 3
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("\n".join(case.model_dump_json() for case in cases), encoding="utf-8")
    evaluate = quality.evaluate_case
    directory = tmp_path / "run"

    def interrupt(case, *args, **kwargs):
        if case.id == "tools_no_need":
            raise KeyboardInterrupt()
        return evaluate(case, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(quality, "evaluate_case", interrupt)
        with pytest.raises(KeyboardInterrupt):
            quality.run(output_dir=directory, cases_path=cases_path)
    path = directory / "results.jsonl"
    row = json.loads(path.read_bytes())
    assert row["error"] is not None
    row["human_review"].update(status="reviewed", notes="Insufficient final word compliance.", overall_pass=False)
    path.write_bytes(json.dumps(row).encode() + b"\n")
    before = path.read_bytes()
    called = []

    def capture(case, *args, **kwargs):
        called.append(case.id)
        return evaluate(case, *args, **kwargs)

    monkeypatch.setattr(quality, "evaluate_case", capture)
    quality.run(output_dir=directory, resume=True, cases_path=cases_path)
    assert called == ["tools_no_need"]
    assert path.read_bytes().startswith(before)
    summary = read_json(directory / "summary.json")
    assert summary["generation_failures"] == 1 and summary["deterministic"]["pass_rate"] == 0.5


def test_completed_and_missing_runs_do_not_construct_providers(tmp_path, monkeypatch):
    directory = quality.run(output_dir=tmp_path / "complete", ids=IDS[:1])
    before = snapshot(directory)
    forbid_provider(monkeypatch)
    with pytest.raises(storage.RunArtifactError, match="already complete"):
        quality.run(output_dir=directory, resume=True, ids=IDS[:1])
    assert snapshot(directory) == before
    with pytest.raises(FileExistsError):
        quality.run(output_dir=directory)
    with pytest.raises(storage.RunArtifactError, match="existing"):
        quality.run(output_dir=tmp_path / "missing", resume=True)
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("raw", [b'[]', b'{}', b'{"status":', b'{"suite":NaN}', b'{"suite":1,"suite":2}'])
def test_invalid_manifest_json_fails_without_provider(tmp_path, monkeypatch, raw):
    directory = partial_run(tmp_path / "run", monkeypatch)
    (directory / "run.json").write_bytes(raw)
    before = snapshot(directory)
    forbid_provider(monkeypatch)
    with pytest.raises(storage.RunArtifactError):
        quality.run(output_dir=directory, resume=True, ids=IDS)
    assert snapshot(directory) == before


@pytest.mark.parametrize("artifact", ["run.json", "results.jsonl", ".run.lock"])
def test_missing_required_artifact_is_not_recreated(tmp_path, monkeypatch, artifact):
    directory = partial_run(tmp_path / "run", monkeypatch)
    (directory / artifact).unlink()
    before = snapshot(directory)
    forbid_provider(monkeypatch)
    with pytest.raises(storage.RunArtifactError):
        quality.run(output_dir=directory, resume=True, ids=IDS)
    assert snapshot(directory) == before


def test_flush_precedes_fsync_and_append_is_compact_utf8(tmp_path, monkeypatch):
    order = []

    class Handle:
        def flush(self):
            order.append("flush")

        def fileno(self):
            return 42

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "fsync", lambda descriptor: order.append(("fsync", descriptor)))
        storage.flush_durable(Handle())
    assert order == ["flush", ("fsync", 42)]
    calls = []
    flush = storage.flush_durable

    def capture(handle):
        flush(handle)
        calls.append(handle.name)

    monkeypatch.setattr(storage, "flush_durable", capture)
    path = tmp_path / "results.jsonl"
    storage.durable_append(path, {"id": "one", "response": "café"})
    assert path.read_bytes() == '{"id":"one","response":"café"}\n'.encode()
    assert len(calls) == 1
    with path.open("ab") as handle:
        handle.write(b'{"torn":')
    storage.truncate_torn_tail(path, len('{"id":"one","response":"café"}\n'.encode()))
    assert len(calls) == 2


def test_real_fsync_failure_stops_before_next_case_or_progress(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "run"
    flush = storage.flush_durable

    def fail_nonempty_result(handle):
        if str(handle.name).endswith("results.jsonl") and handle.tell() > 0:
            handle.flush()
            raise OSError(errno.EIO, "disk failure")
        flush(handle)

    monkeypatch.setattr(storage, "flush_durable", fail_nonempty_result)
    with pytest.raises(OSError):
        quality.run(output_dir=directory, ids=IDS)
    assert read_json(directory / "run.json")["status"] == "running"
    assert read_json(directory / "summary.json")["completed_cases"] == 0
    assert "[1/4]" not in capsys.readouterr().out


def test_atomic_replace_failure_keeps_old_document_and_removes_own_temp(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    storage.atomic_json(path, {"completed_cases": 1})
    before = path.read_bytes()

    def fail(*args):
        raise OSError(errno.EIO, "replace failure")

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError):
        storage.atomic_json(path, {"completed_cases": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_resume_is_rejected_and_lock_releases_on_exception(tmp_path, monkeypatch):
    directory = partial_run(tmp_path / "run", monkeypatch)
    before = snapshot(directory)
    with storage.exclusive_run(directory, resume=True), pytest.raises(
        storage.RunArtifactError, match="already in use",
    ):
        quality.run(output_dir=directory, resume=True, ids=IDS)
    assert snapshot(directory) == before
    quality.run(output_dir=directory, resume=True, ids=IDS)


def test_bounds_fail_closed(tmp_path, monkeypatch):
    directory = partial_run(tmp_path / "run", monkeypatch)
    before = snapshot(directory)
    forbid_provider(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(storage, "MAX_RESULT_LINE_BYTES", 32)
        with pytest.raises(storage.RunArtifactError, match="size limit"):
            quality.run(output_dir=directory, resume=True, ids=IDS)
    with monkeypatch.context() as patch:
        patch.setattr(storage, "MAX_RESULTS_BYTES", 32)
        with pytest.raises(storage.RunArtifactError, match="size limit"):
            quality.run(output_dir=directory, resume=True, ids=IDS)
    with monkeypatch.context() as patch:
        patch.setattr(storage, "MAX_MANIFEST_BYTES", 32)
        with pytest.raises(storage.RunArtifactError, match="size limit"):
            quality.run(output_dir=directory, resume=True, ids=IDS)
    assert snapshot(directory) == before


def test_cli_resume_and_completed_error_are_content_free(tmp_path, monkeypatch, capsys):
    directory = partial_run(tmp_path / "run", monkeypatch)
    arguments = ["aevon_text_quality", "--mode", "fake", "--output-dir", str(directory), "--resume", "--ids", *IDS]
    monkeypatch.setattr("sys.argv", arguments)
    capsys.readouterr()
    quality.main()
    output = capsys.readouterr()
    assert "Resuming 2/4 completed cases." in output.out
    assert "<tool_call>" not in output.out and "What is" not in output.out
    with pytest.raises(SystemExit) as exc:
        quality.main()
    assert exc.value.code == 2
    assert "already complete" in capsys.readouterr().err


def test_case_file_is_byte_logically_unchanged():
    content = quality.DEFAULT_CASES.read_bytes().replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(content)).encode() + b"\0" + content
    # V5.2.1 uses article-only repair in the calculator formatting fake sequence.
    # All 54 prompts/rubrics/expectations remain independently pinned.
    assert hashlib.sha1(blob, usedforsecurity=False).hexdigest() == "e90b1f1758934d16b537338e1b0b8f34441cc06d"
