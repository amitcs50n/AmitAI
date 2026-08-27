import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from training.data import normalize_example


BATCH_PATH = Path("data/sft/v1/batch_01.jsonl")
EVAL_PATH = Path("eval/behavior_v1.jsonl")
PLAN_PATH = Path("configs/sft_v1_dataset_plan.yaml")
REVIEW_PATH = Path("data/sft/v1/batch_01_review.yaml")
SPEC_PATH = Path("configs/amitai_spec_v1.yaml")
V0_PATH = Path("data/sft/amitai_sft_v0.jsonl")

EXPECTED_COUNTS = {
    "normal_conversation": 3,
    "technical_coding": 4,
    "reasoning_decision_support": 3,
    "disagreement_correction": 2,
    "uncertainty_hallucination_resistance": 2,
    "emotional_practical_support": 1,
    "tool_behavior": 1,
    "memory_continuity": 1,
    "creative_roleplay": 2,
    "boundaries_concise_refusals": 1,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _collect_rule_ids(value: Any) -> set[str]:
    rule_ids: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("id"), str):
            rule_ids.add(value["id"])
        for child in value.values():
            rule_ids.update(_collect_rule_ids(child))
    elif isinstance(value, list):
        for child in value:
            rule_ids.update(_collect_rule_ids(child))
    return rule_ids


def _message_text(message: dict[str, Any]) -> str:
    return "\n".join(part["text"] for part in message["content"])


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _assistant_word_count(row: dict[str, Any]) -> int:
    text = " ".join(
        _message_text(message)
        for message in row["messages"]
        if message["role"] == "assistant"
    )
    return len(text.split())


def _response_length_bucket(row: dict[str, Any]) -> str:
    word_count = _assistant_word_count(row)
    if word_count < 55:
        return "short"
    if word_count < 170:
        return "medium"
    return "long"


def test_batch_01_schema_metadata_and_exact_category_mix() -> None:
    rows = _load_jsonl(BATCH_PATH)
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    known_rule_ids = _collect_rule_ids(spec)
    canonical_system_message = plan["record_schema"]["canonical_system_message"]
    id_pattern = re.compile(plan["record_schema"]["id_format"])

    assert len(rows) == 20
    assert Counter(row["category"] for row in rows) == Counter(EXPECTED_COUNTS)
    assert len({row["id"] for row in rows}) == len(rows)

    for row in rows:
        assert normalize_example(row) == row
        assert id_pattern.fullmatch(row["id"])
        assert row["spec_version"] == "1.0.0"
        assert 1 <= len(row["primary_rules"]) <= 3
        assert len(row["primary_rules"]) == len(set(row["primary_rules"]))
        assert set(row["primary_rules"]) <= known_rule_ids

        roles = [message["role"] for message in row["messages"]]
        user_turns = roles.count("user")
        assert roles == [
            "system",
            *[role for _ in range(user_turns) for role in ("user", "assistant")],
        ]
        assert _message_text(row["messages"][0]) == canonical_system_message


def test_batch_01_meets_turn_and_response_length_targets() -> None:
    rows = _load_jsonl(BATCH_PATH)
    user_turn_counts = Counter(
        sum(message["role"] == "user" for message in row["messages"]) for row in rows
    )
    length_counts = Counter(_response_length_bucket(row) for row in rows)

    assert user_turn_counts == Counter({1: 12, 2: 6, 4: 2})
    assert length_counts == Counter({"short": 7, "medium": 10, "long": 3})


def test_batch_01_has_no_exact_duplicates_or_eval_leakage() -> None:
    rows = _load_jsonl(BATCH_PATH)
    eval_rows = _load_jsonl(EVAL_PATH)
    v0_rows = _load_jsonl(V0_PATH)

    batch_user_turns = {
        _normalized(_message_text(message))
        for row in rows
        for message in row["messages"]
        if message["role"] == "user"
    }
    batch_conversations = [
        _normalized(
            " ".join(
                _message_text(message)
                for message in row["messages"]
                if message["role"] == "user"
            )
        )
        for row in rows
    ]
    held_out_prompts = {_normalized(row["prompt"]) for row in eval_rows}
    v0_user_turns = {
        _normalized(_message_text(message))
        for row in v0_rows
        for message in row["messages"]
        if message["role"] == "user"
    }

    assert len(batch_user_turns) == sum(
        message["role"] == "user" for row in rows for message in row["messages"]
    )
    assert len(batch_conversations) == len(set(batch_conversations))
    assert not (batch_user_turns & held_out_prompts)
    assert not (batch_user_turns & v0_user_turns)
    assert not ({row["id"] for row in rows} & {row["id"] for row in eval_rows + v0_rows})


def test_batch_01_avoids_repetitive_assistant_scaffolding() -> None:
    rows = _load_jsonl(BATCH_PATH)
    all_assistant_text = [
        _message_text(message)
        for row in rows
        for message in row["messages"]
        if message["role"] == "assistant"
    ]
    opening_counts = Counter(
        " ".join(_normalized(text).split()[:4]) for text in all_assistant_text
    )

    assert max(opening_counts.values()) <= 3
    for text in all_assistant_text:
        lowered = text.casefold()
        assert not lowered.startswith(("absolutely", "certainly", "great question"))
        assert "let me know if you need" not in lowered


def test_batch_01_memory_row_uses_retrieved_memory_semantics() -> None:
    rows = _load_jsonl(BATCH_PATH)
    memory_rows = [row for row in rows if row["category"] == "memory_continuity"]

    assert len(memory_rows) == 1
    row = memory_rows[0]
    assert row["id"] == "sftv1_memory_001"
    assert set(row["primary_rules"]) == {"MEMORY-002", "MEMORY-004", "MEMORY-006"}
    assert sum(message["role"] == "user" for message in row["messages"]) == 4

    user_text = "\n".join(
        _message_text(message)
        for message in row["messages"]
        if message["role"] == "user"
    ).casefold()
    assistant_text = "\n".join(
        _message_text(message)
        for message in row["messages"]
        if message["role"] == "assistant"
    ).casefold()

    assert "retrieved memory supplied by the runtime" in user_text
    assert "source: prior conversation" in user_text
    assert "recorded_at:" in user_text
    assert "correction for this purchase" in user_text
    assert "stale where it conflicts" in assistant_text
    assert "retrieved memory contributed only" in assistant_text
    assert "came from this conversation" in assistant_text
    assert "anything else as remembered" in assistant_text


def test_batch_01_manual_review_is_complete_and_approved() -> None:
    rows = _load_jsonl(BATCH_PATH)
    review = yaml.safe_load(REVIEW_PATH.read_text(encoding="utf-8"))["review"]
    reviewed_rows = review["rows"]

    assert review["batch_id"] == "batch_01"
    assert review["spec_version"] == "1.0.0"
    assert review["status"] == "approved_after_manual_review"
    assert all(review["manual_checks"].values())
    assert {row["id"] for row in rows} == {row["id"] for row in reviewed_rows}
    assert len(reviewed_rows) == 20
    assert all(row["decision"] == "approved" for row in reviewed_rows)
