import json
from pathlib import Path

import yaml


EVAL_PATH = Path("eval/behavior_v1.jsonl")
SFT_PATHS = [
    Path("data/sft/amitai_sft_v0.jsonl"),
    *sorted(Path("data/sft/v1").glob("batch_*.jsonl")),
]
SPEC_PATH = Path("configs/amitai_spec_v1.yaml")


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _collect_rule_ids(value) -> set[str]:
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


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def test_behavior_eval_schema_and_rule_references() -> None:
    rows = _load_jsonl(EVAL_PATH)
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    known_rule_ids = _collect_rule_ids(spec)
    required_fields = {
        "id",
        "spec_version",
        "category",
        "primary_rules",
        "prompt",
        "pass_criteria",
        "failure_signals",
    }

    assert 20 <= len(rows) <= 30
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert required_fields <= row.keys()
        assert row["spec_version"] == "1.1.0"
        assert row["primary_rules"]
        assert set(row["primary_rules"]) <= known_rule_ids
        assert row["prompt"].strip()
        assert row["pass_criteria"]
        assert row["failure_signals"]


def test_eval_ids_and_prompts_are_held_out_from_training() -> None:
    eval_rows = _load_jsonl(EVAL_PATH)
    sft_rows = [row for path in SFT_PATHS for row in _load_jsonl(path)]
    training_ids = {row["id"] for row in sft_rows}
    training_prompts = {
        _normalized(part["text"])
        for row in sft_rows
        for message in row["messages"]
        if message["role"] == "user"
        for part in message["content"]
        if part["type"] == "text"
    }

    assert not ({row["id"] for row in eval_rows} & training_ids)
    assert not ({_normalized(row["prompt"]) for row in eval_rows} & training_prompts)
