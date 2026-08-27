from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PLAN_PATH = Path("configs/sft_v1_dataset_plan.yaml")
SPEC_PATH = Path("configs/amitai_spec_v1.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def test_sft_v1_category_and_scenario_totals() -> None:
    document = _load_yaml(PLAN_PATH)
    plan = document["plan"]
    categories = document["category_mix"]
    expected_counts = {
        "normal_conversation": 15,
        "technical_coding": 20,
        "reasoning_decision_support": 12,
        "disagreement_correction": 10,
        "uncertainty_hallucination_resistance": 8,
        "emotional_practical_support": 7,
        "tool_behavior": 8,
        "memory_continuity": 5,
        "creative_roleplay": 10,
    }

    assert plan["status"] == "frozen_for_batch_creation"
    assert plan["spec_version"] == "1.1.0"
    assert {name: value["examples"] for name, value in categories.items()} == expected_counts
    assert sum(expected_counts.values()) == plan["total_examples"] == 95
    for category in categories.values():
        assert sum(category["scenario_mix"].values()) == category["examples"]

    diversity = document["diversity_targets"]
    assert sum(diversity["turn_depth"].values()) == 95
    assert sum(diversity["assistant_response_length"].values()) == 95
    per_batch = diversity["per_batch_targets"]
    assert sum(value for key, value in per_batch.items() if "user_turn" in key) == 19
    assert sum(value for key, value in per_batch.items() if "responses" in key) == 19

    eval_map = document["evaluation_reporting"]["existing_eval_category_map"]
    assert set(eval_map.values()) <= set(categories)


def test_sft_v1_batches_are_nineteen_rows_and_match_category_mix() -> None:
    document = _load_yaml(PLAN_PATH)
    plan = document["plan"]
    categories = document["category_mix"]
    batches = document["batch_plan"]["batches"]
    aggregate = Counter()

    assert len(batches) == plan["batch_count"] == 5
    for batch in batches:
        assert set(batch["category_counts"]) == set(categories)
        assert sum(batch["category_counts"].values()) == plan["batch_size"] == 19
        aggregate.update(batch["category_counts"])

    assert aggregate == Counter(
        {name: category["examples"] for name, category in categories.items()}
    )


def test_sft_v1_plan_uses_frozen_spec_and_known_rule_ids() -> None:
    document = _load_yaml(PLAN_PATH)
    spec = _load_yaml(SPEC_PATH)
    known_rule_ids = _collect_rule_ids(spec)

    assert spec["spec"]["version"] == document["plan"]["spec_version"]
    assert spec["spec"]["status"] == document["plan"]["required_spec_status"]
    for category in document["category_mix"].values():
        assert category["target_rules"]
        assert set(category["target_rules"]) <= known_rule_ids
