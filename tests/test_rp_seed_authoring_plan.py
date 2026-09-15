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

    assert result["status"] == "ready_for_human_authoring"
    assert result["authoring_spec_version"] == "1.2.0"
    assert result["character_count"] == 16
    assert result["adult_capable_character_count"] == 8
    assert result["scene_count"] == 48
    assert result["category_counts"] == {
        "sfw": 28,
        "mature_nonsexual": 12,
        "adult_capable": 8,
    }
    assert result["length_counts"] == {"short": 12, "medium": 24, "long": 12}
    assert result["category_length_counts"] == {
        "sfw": {"short": 7, "medium": 14, "long": 7},
        "mature_nonsexual": {"short": 3, "medium": 6, "long": 3},
        "adult_capable": {"short": 2, "medium": 4, "long": 2},
    }
    assert result["adult_scene_structures"] == {
        "emotionally_complicated": 1,
        "desire_led": 2,
        "awkward_reconnection": 1,
        "boundary_checkin_heavy": 2,
        "playful_affectionate": 2,
    }
    assert len(result["mature_conflict_shapes"]) >= 8
    assert result["author_scene_counts"] == {
        "author_slot_01": 12,
        "author_slot_02": 12,
        "author_slot_03": 12,
        "author_slot_04": 12,
    }
    assert result["rights_status"] == "pending"
    assert result["permissions_remain_false"] is True
    assert result["synthetic_generation_authorized"] is False
    assert result["qlora_training_authorized"] is False
    assert result["human_scenes_created"] == 0
    assert result["diagnostic_scenes_migrated"] == 0
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


def test_every_scene_and_card_has_a_complete_user_agency_contract():
    roster = yaml.safe_load((AUTHORING / "roster.yaml").read_text(encoding="utf-8"))
    allocation = yaml.safe_load((AUTHORING / "scene_allocation.yaml").read_text(encoding="utf-8"))
    scene_bans = {
        "appearance", "gender", "attraction", "thoughts", "emotions", "decisions",
        "past history beyond known_user_facts", "unprovided physical actions",
    }
    card_bans = {
        "appearance", "gender", "attraction", "thoughts", "emotions", "decisions",
        "unstated history", "unprovided physical actions",
    }
    baseline_bans = {
        "appearance", "gender unless explicitly specified", "attraction", "thoughts",
        "emotions", "unstated history", "unprovided actions", "unexpressed consent",
    }

    assert set(allocation["baseline_forbidden_user_assumptions"]) == baseline_bans
    assert set(roster["baseline_prohibited_user_assumptions"]) == baseline_bans

    for scene in allocation["scenes"]:
        assert scene["user_role"]
        assert scene["known_user_facts"]
        assert scene_bans <= set(scene["forbidden_user_assumptions"])

    for card in roster["characters"]:
        assert set(card["boundaries"]) == {"hard", "soft", "stop_behavior"}
        assert card["boundaries"]["hard"]
        assert isinstance(card["boundaries"]["soft"], list)
        assert card["boundaries"]["stop_behavior"]
        assert len(card["agency_contract"]["user_agency_constraints"]) >= 3
        assert card_bans <= set(card["agency_contract"]["prohibited_user_assumptions"])
        assert card["agency_contract"]["after_rejection_or_disagreement"]
        assert set(card["relationship_context"]) == {
            "default_stage", "permitted_dynamics", "adult_context_constraints",
        }


def test_mara_and_senka_no_longer_have_an_unexplained_cross_setting_history():
    roster = yaml.safe_load((AUTHORING / "roster.yaml").read_text(encoding="utf-8"))
    cards = {card["character_id"]: card for card in roster["characters"]}

    assert "Senka Vale" not in json.dumps(cards["char_mara_venn"], sort_keys=True)
    assert "Mara Venn" not in json.dumps(cards["char_senka_vale"], sort_keys=True)


def test_future_conversation_contract_matches_existing_loader_surface():
    schema = json.loads((AUTHORING / "schemas/conversation.schema.json").read_text(encoding="utf-8"))
    required = set(schema["required"])
    roles = set(schema["properties"]["messages"]["items"]["properties"]["role"]["enum"])

    assert {"id", "spec_version", "category", "primary_rules", "messages"} <= required
    assert roles <= ALLOWED_ROLES
    assert schema["properties"]["spec_version"]["const"] == "1.2.0"
    assert schema["properties"]["category"]["const"] == "creative_roleplay"
    assert schema["properties"]["training_ready"]["const"] is False

    scene_metadata = schema["properties"]["scene_metadata"]
    assert {
        "length_band", "intermediate_state_beats", "shared_history_facts",
        "adult_content_ceiling",
    } <= set(scene_metadata["required"])
    assert scene_metadata["properties"]["forbidden_user_assumptions"]["contains"]["const"] == "unexpressed consent"


def test_v12_progression_relationship_texture_and_setup_facts_are_structural():
    allocation = yaml.safe_load((AUTHORING / "scene_allocation.yaml").read_text(encoding="utf-8"))
    scenes = {scene["scene_id"]: scene for scene in allocation["scenes"]}

    assert allocation["authoring_spec_version"] == "1.2.0"
    long_scenes = [scene for scene in scenes.values() if scene["length_band"] == "long"]
    mature_scenes = [scene for scene in scenes.values() if scene["category"] == "mature_nonsexual"]
    assert len(long_scenes) == 12
    assert all(3 <= len(scene["intermediate_state_beats"]) <= 5 for scene in long_scenes)
    assert all(1 <= len(scene["shared_history_facts"]) <= 2 for scene in mature_scenes)

    setup_ids = {
        "scene_celeste_clockwork_clue",
        "scene_ilyan_low_tide_vault",
        "scene_nadiya_private_evening_detour",
        "scene_priya_empty_studio",
        "scene_safiya_weather_veto",
    }
    assert {scene_id for scene_id, scene in scenes.items() if "setup_facts" in scene} == setup_ids
    assert all(len(scenes[scene_id]["setup_facts"]) >= 3 for scene_id in setup_ids)


def test_v12_targeted_card_voice_changes_do_not_revise_other_cards():
    roster = yaml.safe_load((AUTHORING / "roster.yaml").read_text(encoding="utf-8"))
    cards = {card["character_id"]: card for card in roster["characters"]}
    revised_ids = {
        "char_ilyan_sorrell", "char_jun_park", "char_nadiya_quill", "char_safiya_calder",
    }

    assert roster["authoring_spec_version"] == "1.2.0"
    assert {
        character_id for character_id, card in cards.items()
        if card["revision"]["card_version"] == "1.2.0"
    } == revised_ids
    assert all(cards[character_id]["revision"]["revision"] == 3 for character_id in revised_ids)
    assert all(
        card["revision"] == {"card_version": "1.1.0", "revision": 2, "content_sha256": None}
        for character_id, card in cards.items() if character_id not in revised_ids
    )
    assert "constant procedural restatement" in cards["char_ilyan_sorrell"]["voice_style"]["avoids"]
    assert "serial interrogation" in cards["char_jun_park"]["voice_style"]["avoids"]
    assert "generic compliance language" in cards["char_nadiya_quill"]["voice_style"]["avoids"]
    assert any("read-backs only when safety requires" in rule for rule in cards["char_safiya_calder"]["behavior_constraints"])


def test_v12_review_contract_separates_diagnostics_from_human_approval():
    schema = json.loads((AUTHORING / "schemas/review.schema.json").read_text(encoding="utf-8"))
    diagnostics = schema["properties"]["diagnostic_pre_review"]
    expected = {
        "generic_assistant_leakage", "procedural_dialogue_tendency",
        "excessive_state_recap", "repeated_option_menu_behavior",
        "same_model_voice_convergence",
    }

    assert diagnostics["properties"]["gating"]["const"] is False
    assert expected <= set(diagnostics["required"])
    assert {"opening_third", "middle_third", "final_third"} <= set(
        schema["properties"]["voice_drift_review"]["required"]
    )


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
