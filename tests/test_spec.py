from pathlib import Path
from typing import Any

import yaml


SPEC_PATH = Path("configs/amitai_spec_v1.yaml")


def _load_spec() -> dict[str, Any]:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def _collect_rule_ids(value: Any) -> list[str]:
    rule_ids: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("id"), str):
            rule_ids.append(value["id"])
        for child in value.values():
            rule_ids.extend(_collect_rule_ids(child))
    elif isinstance(value, list):
        for child in value:
            rule_ids.extend(_collect_rule_ids(child))
    return rule_ids


def test_v1_spec_has_required_sections() -> None:
    spec = _load_spec()
    required_sections = {
        "scope_separation",
        "profiles",
        "instruction_precedence",
        "response_contract",
        "tone_and_voice",
        "disagreement_and_correction",
        "technical_behavior",
        "roleplay_and_fiction",
        "tool_behavior",
        "memory_behavior",
        "hard_boundaries",
        "dataset_generation_constraints",
        "acceptance_criteria",
    }

    assert spec["spec"]["version"] == "1.0.0"
    assert spec["spec"]["status"] == "draft_for_review"
    assert required_sections <= spec.keys()


def test_v1_rule_ids_are_unique_and_context_references_exist() -> None:
    spec = _load_spec()
    rule_ids = _collect_rule_ids(spec)
    context_references = {
        rule_id
        for mode in spec["context_modes"].values()
        for rule_id in mode["emphasize_rules"]
    }

    assert len(rule_ids) == len(set(rule_ids))
    assert context_references <= set(rule_ids)
