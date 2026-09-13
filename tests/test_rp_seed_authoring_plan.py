"""Structural regressions for the design-only, rights-clear RP seed plan."""

import json
from pathlib import Path

import yaml

from scripts.validate_rp_seed_authoring import validate
from training.data import ALLOWED_ROLES

ROOT = Path(__file__).resolve().parents[1]
AUTHORING = ROOT / "data/sft/rp_seed_v1/authoring"


def test_canonical_authoring_plan_has_exact_allocation_and_closed_gates():
    result = validate(ROOT)

    assert result["status"] == "valid_design_only"
    assert result["character_count"] == 16
    assert result["adult_capable_character_count"] == 8
    assert result["scene_count"] == 48
    assert result["category_counts"] == {
        "sfw": 28,
        "mature_nonsexual": 12,
        "adult_capable": 8,
    }
    assert result["length_counts"] == {"short": 12, "medium": 24, "long": 12}
    assert result["author_scene_counts"] == {
        "author_slot_01": 12,
        "author_slot_02": 12,
        "author_slot_03": 12,
        "author_slot_04": 12,
    }
    assert result["rights_status"] == "pending"
    assert result["synthetic_generation_authorized"] is False
    assert result["qlora_training_authorized"] is False
    assert result["training_ready"] is False


def test_scene_matrix_contains_outlines_only_and_adult_cards_have_sfw_coverage():
    roster = yaml.safe_load((AUTHORING / "roster.yaml").read_text(encoding="utf-8"))
    allocation = yaml.safe_load((AUTHORING / "scene_allocation.yaml").read_text(encoding="utf-8"))
    adult_ids = {card["character_id"] for card in roster["characters"] if card["adult_capable"]}

    scene_categories = {character_id: set() for character_id in adult_ids}
    for scene in allocation["scenes"]:
        assert not ({"messages", "dialogue", "conversation", "turns"} & set(scene))
        if scene["character_id"] in adult_ids:
            scene_categories[scene["character_id"]].add(scene["category"])

    assert all({"sfw", "adult_capable"} <= categories for categories in scene_categories.values())


def test_future_conversation_contract_matches_existing_loader_surface():
    schema = json.loads((AUTHORING / "schemas/conversation.schema.json").read_text(encoding="utf-8"))
    required = set(schema["required"])
    roles = set(schema["properties"]["messages"]["items"]["properties"]["role"]["enum"])

    assert {"id", "spec_version", "category", "primary_rules", "messages"} <= required
    assert roles <= ALLOWED_ROLES
    assert schema["properties"]["spec_version"]["const"] == "1.1.0"
    assert schema["properties"]["category"]["const"] == "creative_roleplay"
    assert schema["properties"]["training_ready"]["const"] is False


def test_heldout_plan_is_separate_and_not_falsely_claimed_complete():
    allocation = yaml.safe_load((AUTHORING / "scene_allocation.yaml").read_text(encoding="utf-8"))
    heldout = yaml.safe_load((AUTHORING / "heldout_evaluation.yaml").read_text(encoding="utf-8"))
    training_families = {scene["scenario_family"] for scene in allocation["scenes"]}
    heldout_families = set(heldout["reserved_scenario_families"])

    assert training_families.isdisjoint(heldout_families)
    assert heldout["plan_status"] == "frozen_for_seed_authoring"
    assert heldout["case_content_status"] == "not_authored"
    assert heldout["case_manifest_sha256"] is None
    assert heldout["denylist_sha256"] is None
    assert not any(heldout["freeze_gate"]["before_synthetic_authorization"].values())
