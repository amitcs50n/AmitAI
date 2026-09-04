"""CPU-only V5.2.1 repair safety; no semantic judge or benchmark special cases."""

import json
from threading import Event

import pytest
from fastapi.testclient import TestClient

from backend.chat_service import ChatGenerationError, GenerationMessage
from evaluation.constraints import (
    parse_constraints,
    validate_exact_word_repair,
    validate_response,
    validate_with_bounded_retries,
)
from runtime.config import load_production_runtime_config
from runtime.generator import ProviderChatGenerator
from tests.app_factory import create_test_app
from tests.test_aevon_epistemic_v5_2 import CAPITAL, TOOL, Provider, run_response
from tests.test_backend_streaming import _parse_sse


@pytest.mark.parametrize("original,candidate,expected", [
    ("The capital is Paris.", "capital is Paris.", True),
    ("The capital is Paris.", "Capital of Paris", False),
    ("The capital is Paris.", "Paris is capital.", False),
    ("The capital is Paris.", "capital was Paris.", False),
    ("This is not safe.", "This is safe.", False),
    ("There is no evidence.", "There is evidence.", False),
    ("This never was safe.", "This was safe.", False),
    ("This may be unsafe.", "This is unsafe.", False),
    ("Cats chase small mice.", "Mice chase cats.", False),
    ("Cats chase mice.", "Cats chase mice. Again.", False),
    ("The result is 1411.", "result is 1412.", False),
    ("The result is 1411.", "result is 1411.", True),
    ("Use the `POSTGRES_URL` value.", "Use `POSTGRES_URL` value.", True),
    ("Use the `POSTGRES_URL` value.", "Use `MYSQL_URL` value.", False),
    ("No, the answer is negative.", "Yes, answer is negative.", False),
    ("No, the answer is negative.", "No, answer is negative.", True),
    ('Use "the value" now.', 'Use "value" now.', False),
    ("Use `the value` now.", "Use `value` now.", False),
    ("Visit ‘a place with the name’ now.", "Visit ‘a place with name’ now.", False),
    ("The, answer is uncertain.", "answer is uncertain.", False),
    ("The result is 12.", "result is 12?", False),
    ("A is a label.", "is a label.", False),
])
def test_exact_word_safety_is_ordered_article_deletion_only(original, candidate, expected):
    safety = validate_exact_word_repair(original, candidate, parse_constraints(CAPITAL))
    assert safety["passed"] is expected
    assert safety["failures"] == ([] if expected else ["repair_changed_exact_word_tokens"])


@pytest.mark.parametrize("original,candidate,count,expected,deterministic", [
    ("The capital is Paris.", "Capital of Paris", 3, "capital is Paris.", True),
    ("The capital is Paris.", "capital is Paris.", 3, "capital is Paris.", False),
    ("This is not safe.", "This is safe.", 3, "This is not safe.", False),
    ("There is no evidence.", "There is evidence.", 3, "There is no evidence.", False),
    ("We never accepted this.", "We accepted this.", 3, "We never accepted this.", False),
    ("Cats chase small mice.", "Mice chase cats.", 3, "Cats chase small mice.", False),
    ("Keep calm.", "Please keep calm.", 3, "Keep calm.", False),
    ("The cat saw a dog.", "Dog saw the cat.", 4, "cat saw a dog.", True),
    ("The cat saw an animal.", "Animal saw cat.", 3, "cat saw animal.", True),
    ("The cat saw no dog.", "Cats saw dogs.", 3, "The cat saw no dog.", False),
    ("The answer is 1411.", "answer is 1412.", 3, "answer is 1411.", True),
    ("The identifier is `a`.", "identifier is `b`.", 3, "identifier is `a`.", True),
    ("A is a label.", "is a label.", 3, "A is label.", True),
    ('Read "a note from the editor".', 'Read "note from editor".', 4,
     'Read "a note from the editor".', False),
    ("The, answer is uncertain.", "answer is uncertain.", 3, "The, answer is uncertain.", False),
])
def test_safe_recovery_or_original_after_one_model_call(
    original, candidate, count, expected, deterministic,
):
    calls = []

    def retry(prompt):
        calls.append(prompt)
        return candidate

    prompt = f"Answer in exactly {count} words."
    result = validate_with_bounded_retries(prompt, original, retry)
    assert len(calls) == result["retry_count"] == 1
    assert result["final_response"] == expected
    assert result["final_validation"] == validate_response(expected, parse_constraints(prompt))
    assert result.get("deterministic_repair_used", False) is deterministic
    fallback = not result["final_validation"]["passed"]
    assert result.get("semantic_fallback_used", False) is fallback
    if deterministic or fallback:
        assert result["repair_safety"]["passed"] is False
        assert result["retry_passed"] is False
        assert result["second_validation"]["passed"] is False
    else:
        assert result["repair_safety"]["passed"] is True


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("original,candidate,expected,fallback", [
    ("The capital is Paris.", "Capital of Paris", "capital is Paris.", False),
    ("The capital is Paris.", "capital is Paris.", "capital is Paris.", False),
    ("This is not safe.", "This is safe.", "This is not safe.", True),
    ("Cats chase small mice.", "Mice chase cats.", "Cats chase small mice.", True),
    ("Keep calm.", "Please keep calm.", "Keep calm.", True),
])
def test_sync_and_streaming_never_return_the_unsafe_repair(
    streaming, original, candidate, expected, fallback,
):
    provider = Provider([original, candidate, "UNUSED_THIRD_CALL"])
    generator = ProviderChatGenerator(load_production_runtime_config(), provider=provider)
    result = run_response(generator, [GenerationMessage("user", CAPITAL)], streaming)
    assert result.response == expected and result.response != "Capital of Paris"
    assert result.tools == [] and len(provider.calls) == 2
    assert next(provider.responses) == "UNUSED_THIRD_CALL"
    assert result.input_tokens == 20 and result.output_tokens == 10
    assert result.validator.get("semantic_fallback_used", False) is fallback
    assert result.validator["final_validation"]["passed"] is not fallback
    assert result.validator["retry_count"] == 1
    serialized = json.dumps(result.validator)
    assert "Previous answer" not in serialized and candidate not in serialized
    assert provider.calls[1][0][:-1] == provider.calls[0][0][:-1]
    assert provider.calls[1][1] == provider.calls[0][1] == generator.config.generation


@pytest.mark.parametrize("streaming", [False, True])
def test_fallback_persists_and_streams_only_original_with_truthful_metadata(tmp_path, streaming):
    provider = Provider(["This is not safe.", "This is safe."])
    generator = ProviderChatGenerator(load_production_runtime_config(), provider=provider)
    app = create_test_app(f"sqlite:///{tmp_path / 'chat.db'}", generator=generator,
                          asset_directory=tmp_path / "assets")
    with TestClient(app) as client:
        response = client.post("/api/chat/stream" if streaming else "/api/chat", json={
            "message": "Assess this action. Answer in exactly 3 words.",
        })
        assert response.status_code == 200
        if streaming:
            events = _parse_sse(response.text.splitlines())
            assert [item["event"] for item in events] == ["start", "text", "final", "done"]
            body = events[-2]["data"]
            assert events[1]["data"] == {"delta": "This is not safe."}
        else:
            body = response.json()
        assert body["response"] == "This is not safe."
        assert body["metadata"]["validator"]["semantic_fallback_used"] is True
        assert body["metadata"]["validator"]["final_validation"]["passed"] is False
        history = client.get(f"/api/conversations/{body['conversation_id']}").json()
        assert len(history["messages"]) == 2
        assert history["messages"][-1]["content"] == "This is not safe."
        assert "This is safe." not in json.dumps(history)
    assert len(provider.calls) == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_unsafe_tool_value_repair_falls_back_without_reexecution(streaming, monkeypatch):
    provider = Provider([TOOL, "1411 is not approximate.", "1411 is approximate."])
    generator = ProviderChatGenerator(load_production_runtime_config(), provider=provider)
    calls = []
    execute = generator._tool_registry.execute

    def counted(*args, **kwargs):
        calls.append(args)
        return execute(*args, **kwargs)

    monkeypatch.setattr(generator._tool_registry, "execute", counted)
    result = run_response(generator, [GenerationMessage(
        "user", "What is 17 * 83? Use the calculator. Answer in exactly 3 words.",
    )], streaming)
    assert result.response == "1411 is not approximate."
    assert result.validator["semantic_fallback_used"] is True
    assert result.tools[0]["result"] == "1411"
    assert len(calls) == 1 and len(provider.calls) == 3


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("candidate", ["Only two", "", TOOL])
def test_safe_invalid_or_malformed_repairs_keep_existing_error_behavior(streaming, candidate):
    provider = Provider(["Only two", candidate])
    generator = ProviderChatGenerator(load_production_runtime_config(), provider=provider)
    with pytest.raises((ChatGenerationError, ValueError)):
        run_response(generator, [GenerationMessage("user", CAPITAL)], streaming)
    assert len(provider.calls) == 2


def test_no_retry_and_unconstrained_paths_do_not_use_fallback():
    def forbidden(_prompt):
        pytest.fail("Provider must not be retried")

    for prompt, original, retries in [(CAPITAL, "capital is Paris.", 1),
                                      ("Hello", "Only two", 1), (CAPITAL, "Only two", 0)]:
        result = validate_with_bounded_retries(prompt, original, forbidden, max_retries=retries)
        assert result["retry_count"] == 0
        assert "semantic_fallback_used" not in result
        assert "deterministic_repair_used" not in result


def test_code_and_conflicting_counts_do_not_receive_article_deletion():
    for prompt in ["Return code only. Answer in exactly 3 words.",
                   "Answer in exactly 3 words and exactly 2 words."]:
        result = validate_with_bounded_retries(prompt, "The answer is fixed.", lambda _: "Changed")
        assert result["final_response"] == "The answer is fixed."
        assert result["semantic_fallback_used"] is True
        assert "deterministic_repair_used" not in result


def test_article_recovery_requires_all_other_mechanical_checks_to_pass():
    result = validate_with_bounded_retries(
        "Answer in exactly 3 words and exactly 2 bullets.",
        "The answer is fixed.", lambda _: "Changed",
    )
    assert result["final_response"] == "The answer is fixed."
    assert result["semantic_fallback_used"] is True
    assert result["final_validation"]["passed"] is False


def test_cancelled_stream_does_not_emit_a_fallback():
    signal = Event()

    class CancellingProvider(Provider):
        def generate(self, messages, generation_config):
            output = super().generate(messages, generation_config)
            if len(self.calls) == 2:
                signal.set()
            return output

    provider = CancellingProvider(["This is not safe.", "This is safe."])
    generator = ProviderChatGenerator(load_production_runtime_config(), provider=provider)
    with pytest.raises(RuntimeError, match="cancelled"):
        assert list(generator.stream_response([GenerationMessage("user", CAPITAL)],
                                               cancel_event=signal)) == []
    assert len(provider.calls) == 2
