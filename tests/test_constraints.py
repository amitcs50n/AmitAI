import hashlib
import json
from pathlib import Path

import pytest
import yaml

from evaluation.baseline import (
    generate_constrained_case,
    load_jsonl,
    validate_reviews_against_responses,
)
from evaluation.constraints import (
    build_retry_prompt,
    count_bullets,
    count_words,
    normalize_code_only,
    parse_constraints,
    validate_response,
    validate_with_one_retry,
)
from evaluation.run_baseline import load_config, mechanical_constraints_enabled, run


BASELINE_V1_CONFIG = Path("configs/baseline_eval.yaml")
BASELINE_V2_CONFIG = Path("configs/baseline_eval_v2.yaml")
CONSTRAINED_CONFIG = Path("configs/baseline_eval_v2_constrained.yaml")
FROZEN_CONFIG_SHA256 = {
    BASELINE_V1_CONFIG: "11026d34398165b8810ec125fe1d107880d571165ab94e302965ce719b99bfee",
    BASELINE_V2_CONFIG: "dc1fd04d916d49f1f788d25255b0fbf25a7b66e064634df33173bef0e99451dc",
}


def _case(prompt: str) -> dict:
    return {
        "id": "eval_mechanical_001",
        "spec_version": "1.1.0",
        "category": "mechanical_constraints",
        "primary_rules": ["RESPONSE-007"],
        "prompt": prompt,
        "pass_criteria": ["Satisfies the explicit mechanical constraint"],
        "failure_signals": ["Misses the explicit mechanical constraint"],
    }


def test_parse_constraints_accepts_only_supported_explicit_shapes() -> None:
    assert parse_constraints(
        "Use exactly 90 words. Then exactly 5 bullets, at most 7 bullets. Return code only."
    ) == [
        {"type": "exact_words", "count": 90},
        {"type": "exact_bullets", "count": 5},
        {"type": "at_most_bullets", "count": 7},
        {"type": "code_only"},
    ]
    assert parse_constraints("Code only.") == [{"type": "code_only"}]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Use exactly five words.", [{"type": "exact_words", "count": 5}]),
        ("Use exactly five bullets.", [{"type": "exact_bullets", "count": 5}]),
        ("Use at most five bullets.", [{"type": "at_most_bullets", "count": 5}]),
        ("Write exactly ninety words.", [{"type": "exact_words", "count": 90}]),
        ("Use exactly twenty-five words.", [{"type": "exact_words", "count": 25}]),
        ("Use exactly twenty five words.", [{"type": "exact_words", "count": 25}]),
        ("Use exactly one hundred words.", [{"type": "exact_words", "count": 100}]),
    ],
)
def test_parse_constraints_supports_written_integer_counts(
    prompt: str,
    expected: list[dict],
) -> None:
    assert parse_constraints(prompt) == expected


def test_written_integer_vocabulary_covers_zero_through_one_hundred() -> None:
    written_counts = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "ninety-nine": 99,
        "one hundred": 100,
    }

    for written, count in written_counts.items():
        assert parse_constraints(f"Use exactly {written} words.") == [
            {"type": "exact_words", "count": count}
        ]


def test_written_and_digit_counts_deduplicate_after_normalization() -> None:
    assert parse_constraints("Exactly five words; exactly 5 words.") == [
        {"type": "exact_words", "count": 5}
    ]


@pytest.mark.parametrize(
    "prompt",
    [
        "I had three interviews.",
        "Give me five ideas.",
        "Give me a few ideas.",
        "Give me one dinner idea.",
        "Keep it short.",
        "Do not return code only.",
        "Do not use exactly 90 words.",
        "Do not use exactly five words.",
        "Do not answer in exactly ninety words.",
        "Use exactly 3 sentences.",
        "Write five sentences.",
        "Give me one bullet.",
        "Use a few bullets.",
        "Use exactly a hundred words.",
        "Use exactly fifth word.",
        "Use exactly one hundred and five words.",
        "Use exactly twenty and five words.",
        "Use exactly ninety-first words.",
        "Use exactly one-and-twenty words.",
        "Use exactly IV words.",
        "Use exactly 2.5 words.",
        'Discuss the phrase "exactly 90 words".',
        'Discuss the phrase "exactly five words".',
        'Explain "at most five bullets".',
        'Explain the phrase "return code only".',
        'Explain the instruction "Please answer in exactly 90 words about cats".',
        'Discuss "Return code only when using this API".',
        "Explain 'Please answer in exactly 90 words about cats'.",
        "Discuss `Return code only when using this API`.",
        'Explain the malformed instruction "exactly 90 words.',
        "Explain the malformed instruction 'exactly five words.",
    ],
)
def test_vague_or_negated_language_does_not_trigger_constraints(prompt: str) -> None:
    assert parse_constraints(prompt) == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Without headings, write exactly 90 words.",
        "Do not be verbose; answer in exactly 90 words.",
        "Don't use extra headings. Write exactly 90 words.",
        "It's the user's request. Write exactly 90 words.",
    ],
)
def test_unrelated_negative_language_does_not_hide_a_constraint(prompt: str) -> None:
    assert parse_constraints(prompt) == [{"type": "exact_words", "count": 90}]


@pytest.mark.parametrize(
    "prompt",
    [
        'Explain the malformed instruction "exactly 90 words.',
        "Explain the malformed instruction 'return code only.",
        "Explain the malformed instruction `exactly 3 bullets.",
        "'Crossed \"exactly 90 words' delimiters\".",
    ],
)
def test_malformed_or_ambiguous_quotes_suppress_constraints(prompt: str) -> None:
    assert parse_constraints(prompt) == []


def test_quoted_constraint_does_not_hide_a_later_unquoted_constraint() -> None:
    prompt = 'Discuss "exactly 90 words", then write exactly 20 words.'

    assert parse_constraints(prompt) == [{"type": "exact_words", "count": 20}]


def test_exact_word_count_passes_and_fails_using_whitespace_splitting() -> None:
    constraint = [{"type": "exact_words", "count": 3}]

    assert count_words("  one\n two\tthree  ") == 3
    assert validate_response("one two three", constraint)["passed"] is True
    failed = validate_response("one two", constraint)
    assert failed["passed"] is False
    assert failed["checks"][0]["actual"] == 2
    assert failed["failures"] == [
        "Expected exactly 3 words, but the answer contained 2 words."
    ]


def test_exact_bullet_count_supports_markdown_markers_and_fails_on_mismatch() -> None:
    response = "\n".join(
        [
            "  - dash",
            "\t* star",
            "+ plus",
            "1. ordered dot",
            "2) ordered paren",
            "normal prose",
            "-not a bullet",
        ]
    )

    assert count_bullets(response) == 5
    assert validate_response(
        response, [{"type": "exact_bullets", "count": 5}]
    )["passed"] is True
    assert validate_response(
        response, [{"type": "exact_bullets", "count": 4}]
    )["passed"] is False


def test_at_most_bullet_count_passes_and_fails() -> None:
    response = "- first\n- second\n- third"

    assert validate_response(
        response, [{"type": "at_most_bullets", "count": 3}]
    )["passed"] is True
    failed = validate_response(
        response, [{"type": "at_most_bullets", "count": 2}]
    )
    assert failed["passed"] is False
    assert failed["failures"] == [
        "Expected at most 2 bullets, but the answer contained 3 bullets."
    ]


def test_fenced_code_only_normalizes_for_validation() -> None:
    response = "```python\nprint('hello')\n```"

    assert normalize_code_only(response) == "print('hello')"
    validation = validate_response(response, [{"type": "code_only"}])
    assert validation["passed"] is True
    assert validation["checks"][0]["actual"] == "single_fenced_code_block"
    assert validation["normalized_response"] == "print('hello')"


def test_code_fence_normalization_allows_inline_fence_characters() -> None:
    response = '````python\nprint("```")\n````'

    assert normalize_code_only(response) == 'print("```")'
    assert normalize_code_only("~~~python\nprint('hello')\n~~~") == "print('hello')"
    assert normalize_code_only("```python\n\n```") is None


def test_prose_around_a_fenced_code_block_fails_code_only() -> None:
    response = "Here is the code:\n```python\nprint('hello')\n```"

    assert normalize_code_only(response) is None
    assert validate_response(response, [{"type": "code_only"}])["passed"] is False


def test_unfenced_code_only_is_accepted_when_it_cannot_be_disproved() -> None:
    validation = validate_response("print('hello')", [{"type": "code_only"}])

    assert validation["passed"] is True
    assert validation["checks"][0]["actual"] == "unfenced_unverified"
    assert validate_response('print("```")', [{"type": "code_only"}])["passed"] is True


def test_passing_fenced_code_is_preserved_in_the_final_response() -> None:
    calls: list[str] = []
    original = "```python\nprint('hello')\n```"

    result = validate_with_one_retry("Return code only.", original, calls.append)

    assert calls == []
    assert result["original_response"] == original
    assert result["retry_happened"] is False
    assert result["first_validation"]["normalized_response"] == "print('hello')"
    assert result["final_response"] == original


def test_successful_fenced_code_retry_is_preserved_in_the_final_response() -> None:
    retry_response = "```python\nprint('hello')\n```"

    result = validate_with_one_retry(
        "Return code only.",
        "Here is the code:\n```python\nprint('hello')\n```",
        lambda _prompt: retry_response,
    )

    assert result["retry_response"] == retry_response
    assert result["retry_passed"] is True
    assert result["second_validation"]["normalized_response"] == "print('hello')"
    assert result["final_response"] == retry_response


def test_generated_response_and_review_preserve_a_valid_fenced_response() -> None:
    fenced_response = "```python\nprint('hello')\n```"

    class FencedGenerator:
        def generate(self, _messages, _generation_config):
            return fenced_response

    result, review = generate_constrained_case(
        _case("Return code only."),
        FencedGenerator(),
        system_prompt="System instruction",
        generation_config={"max_new_tokens": 32, "do_sample": False},
    )

    assert result["first_validation"]["normalized_response"] == "print('hello')"
    assert result["response"] == result["final_response"] == fenced_response
    assert review["response"] == review["final_response"] == fenced_response


def test_generated_response_and_review_preserve_a_valid_fenced_retry() -> None:
    retry_response = "```python\nprint('hello')\n```"

    class FencedRetryGenerator:
        def __init__(self) -> None:
            self.responses = iter(
                ["Here is the code:\n```python\nprint('hello')\n```", retry_response]
            )

        def generate(self, _messages, _generation_config):
            return next(self.responses)

    result, review = generate_constrained_case(
        _case("Return code only."),
        FencedRetryGenerator(),
        system_prompt="System instruction",
        generation_config={"max_new_tokens": 32, "do_sample": False},
    )

    assert result["retry_passed"] is True
    assert result["second_validation"]["normalized_response"] == "print('hello')"
    assert result["response"] == result["final_response"] == retry_response
    assert review["response"] == review["final_response"] == retry_response


def test_empty_fenced_code_triggers_one_retry() -> None:
    empty_fence = "```python\n\n```"

    result = validate_with_one_retry(
        "Return code only.",
        empty_fence,
        lambda _prompt: "print('hello')",
    )

    assert result["first_validation"]["passed"] is False
    assert result["retry_happened"] is True
    assert result["retry_passed"] is True
    assert result["final_response"] == "print('hello')"


def test_empty_fenced_retry_is_kept_as_the_failed_final_response() -> None:
    empty_fence = "```python\n\n```"

    result = validate_with_one_retry(
        "Return code only.",
        "Here is code:\n```python\nprint('hello')\n```",
        lambda _prompt: empty_fence,
    )

    assert result["retry_passed"] is False
    assert result["retry_response"] == empty_fence
    assert result["final_response"] == empty_fence


def test_no_supported_constraint_returns_the_response_without_retry() -> None:
    calls: list[str] = []

    result = validate_with_one_retry(
        "Give me one dinner idea and keep it short.",
        "Vegetable curry.",
        calls.append,
    )

    assert calls == []
    assert result["parsed_constraints"] == []
    assert result["retry_happened"] is False
    assert result["final_response"] == "Vegetable curry."


def test_passing_first_response_causes_no_retry() -> None:
    calls: list[str] = []

    result = validate_with_one_retry(
        "Answer in exactly 3 words.",
        "One two three",
        calls.append,
    )

    assert calls == []
    assert result["first_validation"]["passed"] is True
    assert result["retry_happened"] is False
    assert result["retry_passed"] is None
    assert result["final_response"] == "One two three"


def test_failing_response_gets_one_successful_retry_with_complete_prompt() -> None:
    retry_prompts: list[str] = []

    def retry(prompt: str) -> str:
        retry_prompts.append(prompt)
        return "One two three"

    result = validate_with_one_retry(
        "Answer in exactly 3 words.",
        "Only two",
        retry,
    )

    assert len(retry_prompts) == 1
    retry_prompt = retry_prompts[0]
    assert "Original user request:\nAnswer in exactly 3 words." in retry_prompt
    assert "Previous answer:\nOnly two" in retry_prompt
    assert (
        "Validation failure:\nExpected exactly 3 words, but the answer contained 2 words."
        in retry_prompt
    )
    assert "contains 2 whitespace-separated words" in retry_prompt
    assert "required total is exactly 3 words" in retry_prompt
    assert "1 word short" in retry_prompt
    assert "add exactly 1 word" in retry_prompt
    assert "Count words exactly as whitespace-separated tokens" in retry_prompt
    assert "Edit the previous answer minimally" in retry_prompt
    assert "Do not rewrite it from scratch unless unavoidable" in retry_prompt
    assert "internally recount using whitespace-separated tokens" in retry_prompt
    assert "Preserve the original content, tone, and task requirements" in retry_prompt
    assert retry_prompt.endswith("Output only the corrected answer.")
    assert result["original_user_prompt"] == "Answer in exactly 3 words."
    assert result["original_response"] == "Only two"
    assert result["parsed_constraints"] == [{"type": "exact_words", "count": 3}]
    assert result["first_validation"]["checks"][0]["actual"] == 2
    assert result["retry_happened"] is True
    assert result["retry_reason"] == (
        "Expected exactly 3 words, but the answer contained 2 words."
    )
    assert result["retry_prompt"] == retry_prompt
    assert result["retry_response"] == "One two three"
    assert result["retry_passed"] is True
    assert result["second_validation"]["passed"] is True
    assert result["final_response"] == "One two three"


@pytest.mark.parametrize(
    ("actual", "direction", "edit"),
    [
        (72, "18 words short", "add exactly 18 words"),
        (104, "14 words too long", "remove exactly 14 words"),
    ],
)
def test_exact_word_retry_prompt_uses_measured_direction_and_delta(
    actual: int,
    direction: str,
    edit: str,
) -> None:
    response = " ".join(f"word{index}" for index in range(actual))
    validation = validate_response(
        response,
        [{"type": "exact_words", "count": 90}],
    )

    retry_prompt = build_retry_prompt(
        "Write exactly 90 words.",
        response,
        validation,
    )

    assert f"contains {actual} whitespace-separated words" in retry_prompt
    assert "required total is exactly 90 words" in retry_prompt
    assert direction in retry_prompt
    assert edit in retry_prompt
    assert "Count words exactly as whitespace-separated tokens" in retry_prompt
    assert "Edit the previous answer minimally" in retry_prompt
    assert "Do not rewrite it from scratch unless unavoidable" in retry_prompt
    assert "internally recount using whitespace-separated tokens" in retry_prompt
    assert retry_prompt.endswith("Output only the corrected answer.")


@pytest.mark.parametrize(
    ("actual", "direction", "edit", "preservation"),
    [
        (
            3,
            "2 bullets short",
            "add exactly 2 bullets",
            "Preserve the original task and content",
        ),
        (
            7,
            "2 excess bullets",
            "remove exactly 2 bullets",
            "Preserve the strongest relevant content",
        ),
    ],
)
def test_exact_bullet_retry_prompt_uses_measured_direction_and_delta(
    actual: int,
    direction: str,
    edit: str,
    preservation: str,
) -> None:
    response = "\n".join(f"- item {index}" for index in range(actual))
    validation = validate_response(
        response,
        [{"type": "exact_bullets", "count": 5}],
    )

    retry_prompt = build_retry_prompt(
        "Use exactly 5 bullets.",
        response,
        validation,
    )

    assert f"contains {actual} Markdown list-item bullets" in retry_prompt
    assert "required total is exactly 5 bullets" in retry_prompt
    assert direction in retry_prompt
    assert edit in retry_prompt
    assert preservation in retry_prompt
    assert "Do not invent unnecessary services or details" in retry_prompt
    assert "internally recount the Markdown list-item bullets" in retry_prompt


def test_at_most_bullet_retry_prompt_states_limit_and_excess() -> None:
    response = "\n".join(f"- item {index}" for index in range(7))
    validation = validate_response(
        response,
        [{"type": "at_most_bullets", "count": 5}],
    )

    retry_prompt = build_retry_prompt(
        "Use at most 5 bullets.",
        response,
        validation,
    )

    assert "contains 7 Markdown list-item bullets" in retry_prompt
    assert "maximum allowed total is 5 bullets" in retry_prompt
    assert "2 excess bullets" in retry_prompt
    assert "remove exactly 2 bullets" in retry_prompt
    assert "final count is no more than 5" in retry_prompt
    assert "Preserve the most important content" in retry_prompt
    assert "Do not invent unnecessary services or details" in retry_prompt


def test_failed_retry_stops_and_retry_response_is_still_final() -> None:
    retry_prompts: list[str] = []

    def retry(prompt: str) -> str:
        retry_prompts.append(prompt)
        return "Still two"

    result = validate_with_one_retry(
        "Answer in exactly 3 words.",
        "Only two",
        retry,
    )

    assert len(retry_prompts) == 1
    assert result["first_validation"]["passed"] is False
    assert result["second_validation"]["passed"] is False
    assert result["retry_response"] == "Still two"
    assert result["retry_passed"] is False
    assert result["final_response"] == "Still two"


def test_multiple_failures_are_combined_into_one_retry() -> None:
    retry_prompts: list[str] = []

    def retry(prompt: str) -> str:
        retry_prompts.append(prompt)
        return "- one\n- two"

    result = validate_with_one_retry(
        "Use exactly 4 words in exactly 2 bullets.",
        "- one",
        retry,
    )

    assert len(retry_prompts) == 1
    assert "Expected exactly 4 words" in retry_prompts[0]
    assert "Expected exactly 2 bullets" in retry_prompts[0]
    assert result["retry_passed"] is True


def test_build_retry_prompt_requires_a_measured_failure() -> None:
    with pytest.raises(ValueError, match="at least one validation failure"):
        build_retry_prompt("Prompt", "Answer", {"passed": True, "failures": []})


def test_generate_constrained_case_preserves_attempt_metadata() -> None:
    class SequenceGenerator:
        def __init__(self) -> None:
            self.responses = iter(["Only two", "One two three"])
            self.calls = []

        def generate(self, messages, generation_config):
            self.calls.append((messages, generation_config))
            return next(self.responses)

    generator = SequenceGenerator()
    case = _case("Answer in exactly 3 words.")
    result, review = generate_constrained_case(
        case,
        generator,
        system_prompt="System instruction",
        generation_config={"max_new_tokens": 32, "do_sample": False},
    )

    assert len(generator.calls) == 2
    assert generator.calls[0][0][-1]["content"] == case["prompt"]
    assert "Previous answer:\nOnly two" in generator.calls[1][0][-1]["content"]
    assert result["original_user_prompt"] == case["prompt"]
    assert result["original_response"] == "Only two"
    assert result["response"] == result["final_response"] == "One two three"
    assert review["response"] == "One two three"
    assert review["first_validation"] == result["first_validation"]
    assert review["second_validation"] == result["second_validation"]
    validate_reviews_against_responses([review], [result])
    review["retry_passed"] = False
    with pytest.raises(ValueError, match="differs from generated responses"):
        validate_reviews_against_responses([review], [result])


def test_generate_constrained_case_keeps_a_failed_retry_as_final() -> None:
    class SequenceGenerator:
        def __init__(self) -> None:
            self.responses = iter(["Only two", "Still two"])
            self.calls = []

        def generate(self, messages, generation_config):
            self.calls.append((messages, generation_config))
            return next(self.responses)

    generator = SequenceGenerator()
    result, review = generate_constrained_case(
        _case("Answer in exactly 3 words."),
        generator,
        system_prompt="System instruction",
        generation_config={"max_new_tokens": 32, "do_sample": False},
    )

    assert len(generator.calls) == 2
    assert result["retry_passed"] is False
    assert result["retry_response"] == "Still two"
    assert result["response"] == result["final_response"] == "Still two"
    assert review["response"] == "Still two"


def test_review_integrity_requires_null_constraint_metadata_fields() -> None:
    class PassingGenerator:
        def generate(self, _messages, _generation_config):
            return "One two three"

    result, review = generate_constrained_case(
        _case("Answer in exactly 3 words."),
        PassingGenerator(),
        system_prompt="System instruction",
        generation_config={"max_new_tokens": 32, "do_sample": False},
    )

    assert result["retry_reason"] is None
    del review["retry_reason"]
    with pytest.raises(ValueError, match="differs from generated responses"):
        validate_reviews_against_responses([review], [result])


def test_run_uses_constraint_validation_without_loading_a_real_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(_case("Answer in exactly 3 words.")) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "constrained-output"
    config_path = tmp_path / "constrained.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "baseline_eval": {
                    "name": "fake_constrained_run",
                    "eval_file": str(eval_path),
                    "output_dir": str(output_dir),
                    "runtime_system_prompt": "System instruction",
                    "mechanical_constraints": {"enabled": True},
                    "model": {"name": "fake/model", "dtype": "bfloat16"},
                    "generation": {
                        "max_new_tokens": 32,
                        "do_sample": False,
                        "seed": 1,
                    },
                    "decision_gate": {
                        "minimum_rule_compliance_rate": 0.9,
                        "maximum_critical_failures": 0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeRetryGenerator:
        resolved_revision = "fake-revision"
        dependency_versions = {"fake-backend": "1.0"}

        def __init__(self, *_args, **_kwargs) -> None:
            self.responses = iter(["Only two", "One two three"])

        def generate(self, _messages, _generation_config):
            return next(self.responses)

    monkeypatch.setattr(
        "evaluation.run_baseline.TransformersGenerator",
        FakeRetryGenerator,
    )

    assert run(config_path) == output_dir
    response = load_jsonl(output_dir / "responses.jsonl")[0]
    manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert response["retry_happened"] is True
    assert response["retry_passed"] is True
    assert response["response"] == response["final_response"] == "One two three"
    assert manifest["mechanical_constraints"]["max_retries"] == 1


def test_constrained_config_is_separate_and_v1_v2_remain_frozen() -> None:
    for path, expected_sha256 in FROZEN_CONFIG_SHA256.items():
        normalized_text = path.read_text(encoding="utf-8")
        assert hashlib.sha256(normalized_text.encode("utf-8")).hexdigest() == expected_sha256

    v1 = load_config(BASELINE_V1_CONFIG)
    v2 = load_config(BASELINE_V2_CONFIG)
    constrained = load_config(CONSTRAINED_CONFIG)

    assert mechanical_constraints_enabled(v1) is False
    assert mechanical_constraints_enabled(v2) is False
    assert mechanical_constraints_enabled(constrained) is True
    assert constrained["name"] == "qwen38_27b_base_behavior_v2_constrained"
    assert constrained["output_dir"] == (
        "outputs/eval/qwen38_27b_base_behavior_v2_constrained"
    )
    assert constrained["output_dir"] != v2["output_dir"]
    assert constrained["eval_file"] == v2["eval_file"]
    assert constrained["runtime_system_prompt"] == v2["runtime_system_prompt"]
    assert constrained["model"] == v2["model"]
    assert constrained["generation"] == v2["generation"]
    assert constrained["decision_gate"] == v2["decision_gate"]


@pytest.mark.parametrize(
    "settings",
    [True, {"enabled": "yes"}, {"enabled": True, "max_retries": 2}],
)
def test_mechanical_constraint_config_rejects_ambiguous_settings(settings) -> None:
    with pytest.raises(ValueError, match="mechanical_constraints"):
        mechanical_constraints_enabled({"mechanical_constraints": settings})
