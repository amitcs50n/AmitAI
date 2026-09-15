#!/usr/bin/env python3
"""Validate the frozen structure of the RP seed V1 authoring plan.

This validator has no mutation or approval path. Passing means that the design
matches its allocation contract; it does not grant rights, quality acceptance,
synthetic-generation authorization, or training authorization.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

AUTHORING = Path("data/sft/rp_seed_v1/authoring")
DOCS = Path("docs/rp_seed_v1")
EXPECTED_CATEGORIES = {"sfw": 28, "mature_nonsexual": 12, "adult_capable": 8}
AUTHORING_SPEC_VERSION = "1.2.0"
EXPECTED_LENGTHS = {"short": 12, "medium": 24, "long": 12}
EXPECTED_CATEGORY_LENGTHS = {
    ("sfw", "short"): 7,
    ("sfw", "medium"): 14,
    ("sfw", "long"): 7,
    ("mature_nonsexual", "short"): 3,
    ("mature_nonsexual", "medium"): 6,
    ("mature_nonsexual", "long"): 3,
    ("adult_capable", "short"): 2,
    ("adult_capable", "medium"): 4,
    ("adult_capable", "long"): 2,
}
EXPECTED_ADULT_STRUCTURES = {
    "boundary_checkin_heavy": 2,
    "playful_affectionate": 2,
    "desire_led": 2,
    "awkward_reconnection": 1,
    "emotionally_complicated": 1,
}
EXPECTED_AUTHORS = {f"author_slot_{number:02d}": 12 for number in range(1, 5)}
EXPECTED_CARD_AUTHORS = {f"author_slot_{number:02d}": 4 for number in range(1, 5)}
V12_CARD_IDS = {
    "char_ilyan_sorrell",
    "char_jun_park",
    "char_nadiya_quill",
    "char_safiya_calder",
}
SETUP_FACT_SCENE_IDS = {
    "scene_celeste_clockwork_clue",
    "scene_ilyan_low_tide_vault",
    "scene_nadiya_private_evening_detour",
    "scene_priya_empty_studio",
    "scene_safiya_weather_veto",
}
REQUIRED_CARD_ASSUMPTION_BANS = {
    "appearance", "gender", "attraction", "thoughts", "emotions", "decisions",
    "unstated history", "unprovided physical actions",
}
REQUIRED_SCENE_ASSUMPTION_BANS = {
    "appearance", "gender", "attraction", "thoughts", "emotions", "decisions",
    "past history beyond known_user_facts", "unprovided physical actions",
}
REQUIRED_BASELINE_ASSUMPTION_BANS = {
    "appearance", "gender unless explicitly specified", "attraction", "thoughts",
    "emotions", "unstated history", "unprovided actions", "unexpressed consent",
}
LENGTH_RANGES = {
    "short": ([12, 18], [700, 1200]),
    "medium": ([20, 30], [1400, 2400]),
    "long": ([32, 48], [2600, 4200]),
}

CARD_FIELDS = {
    "character_id", "name", "age", "pronouns", "fictional_status",
    "real_person_reference", "adult_capable", "physical_description",
    "personality", "voice_style", "background", "motivations", "likes",
    "dislikes", "strengths", "flaws", "boundaries", "relationship_context",
    "setting_lore", "continuity_facts", "behavior_constraints", "agency_contract",
    "author_source", "ownership_license", "revision",
}
SCENE_FIELDS = {
    "scene_id", "character_id", "category", "scenario_family", "setting",
    "relationship_stage", "user_role", "known_user_facts",
    "forbidden_user_assumptions", "emotional_tone", "conflict_or_objective",
    "expected_state_change", "expected_user_decision_points",
    "boundary_refusal_opportunity", "enacted_state_change_required",
    "boundary_refusal_required", "length_band", "target_turn_range",
    "target_token_range", "author_assignment", "evaluation_risk_notes",
    "heldout_scenario_family_reserved",
}
SCHEMA_FILES = {
    "character_card.schema.json",
    "scene_plan.schema.json",
    "conversation.schema.json",
    "provenance.schema.json",
    "review.schema.json",
}
DOC_FILES = {
    "README.md",
    "AUTHORING_SPEC.md",
    "REVIEW_RUBRIC.md",
    "HELD_OUT_EVAL.md",
    "AUTHORIZATION_CHECKLIST.md",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_roster(root: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    roster = _load_yaml(root / AUTHORING / "roster.yaml")
    _require(roster.get("schema_version") == "aevon_rp_roster_v1", "unexpected roster schema")
    _require(roster.get("authoring_spec_version") == AUTHORING_SPEC_VERSION, "roster must use authoring spec V1.2")
    _require(roster.get("status") == "ready_for_human_authoring", "roster design is not ready for human authoring")
    _require(roster.get("training_ready") is False, "roster must remain training_ready=false")
    _require(
        set(roster.get("baseline_prohibited_user_assumptions", [])) == REQUIRED_BASELINE_ASSUMPTION_BANS,
        "roster is missing the V1.2 baseline prohibited user assumptions",
    )
    characters = roster.get("characters")
    _require(isinstance(characters, list) and len(characters) == 16, "roster must contain exactly 16 characters")

    ids: set[str] = set()
    adult_ids: set[str] = set()
    author_counts: Counter[str] = Counter()
    personality_signatures: set[tuple[str, ...]] = set()
    indexed: dict[str, dict[str, Any]] = {}
    for card in characters:
        missing = CARD_FIELDS - set(card)
        _require(not missing, f"{card.get('character_id')} missing card fields: {sorted(missing)}")
        character_id = card["character_id"]
        _require(character_id not in ids, f"duplicate character_id: {character_id}")
        ids.add(character_id)
        indexed[character_id] = card
        _require(isinstance(card["age"], int) and 21 <= card["age"] <= 100, f"{character_id} must be an explicit adult")
        _require(card["fictional_status"] == "original_fictional_adult", f"{character_id} fictional status is not explicit")
        _require(card["real_person_reference"] is False, f"{character_id} may not reference a real person")
        _require(len(card["personality"]) >= 3, f"{character_id} needs at least three personality traits")
        signature = tuple(sorted(card["personality"]))
        _require(signature not in personality_signatures, f"duplicate personality signature: {character_id}")
        personality_signatures.add(signature)
        _require(len(card["continuity_facts"]) >= 4, f"{character_id} needs four continuity facts")
        _require(len(card["behavior_constraints"]) >= 4, f"{character_id} needs four behavior constraints")
        boundaries = card["boundaries"]
        _require(
            set(boundaries) == {"hard", "soft", "stop_behavior"},
            f"{character_id} boundaries have malformed or unexpected fields",
        )
        _require(boundaries["hard"], f"{character_id} needs hard boundaries")
        _require(isinstance(boundaries["soft"], list), f"{character_id} soft boundaries must be a list")
        _require(len(boundaries["stop_behavior"]) >= 20, f"{character_id} needs explicit stop behavior")
        agency = card["agency_contract"]
        _require(
            set(agency)
            == {
                "user_agency_constraints",
                "prohibited_user_assumptions",
                "after_rejection_or_disagreement",
            },
            f"{character_id} agency contract has malformed or unexpected fields",
        )
        _require(len(agency["user_agency_constraints"]) >= 3, f"{character_id} needs three agency constraints")
        _require(
            REQUIRED_CARD_ASSUMPTION_BANS <= set(agency["prohibited_user_assumptions"]),
            f"{character_id} is missing baseline prohibited user assumptions",
        )
        _require(
            len(agency["after_rejection_or_disagreement"]) >= 20,
            f"{character_id} needs behavior after rejection or disagreement",
        )
        relationship = card["relationship_context"]
        _require(
            set(relationship)
            == {"default_stage", "permitted_dynamics", "adult_context_constraints"},
            f"{character_id} relationship context has malformed or unexpected fields",
        )

        if card["adult_capable"]:
            adult_ids.add(character_id)
        source = card["author_source"]
        _require(source["author_id"] is None, f"{character_id} cannot claim a verified author yet")
        _require(source["source_type"] == "original_design_draft", f"{character_id} must remain a design draft")
        author_counts[source["author_slot"]] += 1

        rights = card["ownership_license"]
        _require(rights["rights_status"] == "pending", f"{character_id} rights must remain pending")
        _require(rights["ownership_declaration_id"] is None, f"{character_id} declaration must remain unset")
        _require(rights["license_id"] == "pending", f"{character_id} license must remain pending")
        for field in ("training_permission", "modification_permission", "redistribution_permission"):
            _require(rights[field] is False, f"{character_id} {field} must remain false")
        _require(card["revision"]["content_sha256"] is None, f"{character_id} cannot have a final content hash yet")
        expected_version = "1.2.0" if character_id in V12_CARD_IDS else "1.1.0"
        expected_revision = 3 if character_id in V12_CARD_IDS else 2
        _require(card["revision"]["card_version"] == expected_version, f"{character_id} card version mismatch")
        _require(card["revision"]["revision"] == expected_revision, f"{character_id} card revision mismatch")

    _require(len(adult_ids) == 8, "exactly eight characters must be adult-capable")
    _require(dict(author_counts) == EXPECTED_CARD_AUTHORS, f"card author allocation mismatch: {dict(author_counts)}")
    mara_text = json.dumps(indexed["char_mara_venn"], sort_keys=True)
    senka_text = json.dumps(indexed["char_senka_vale"], sort_keys=True)
    _require("Senka Vale" not in mara_text, "Mara card retains unexplained Senka history")
    _require("Mara Venn" not in senka_text, "Senka card retains unexplained Mara history")
    return indexed, adult_ids


def _validate_scenes(root: Path, cards: dict[str, dict[str, Any]], adult_ids: set[str]) -> dict[str, Any]:
    allocation = _load_yaml(root / AUTHORING / "scene_allocation.yaml")
    _require(allocation.get("schema_version") == "aevon_rp_scene_allocation_v1", "unexpected scene schema")
    _require(allocation.get("authoring_spec_version") == AUTHORING_SPEC_VERSION, "scene allocation must use authoring spec V1.2")
    _require(allocation.get("status") == "ready_for_human_authoring", "scene design is not ready for human authoring")
    _require(allocation.get("training_ready") is False, "scene allocation must remain training_ready=false")
    _require(
        set(allocation.get("baseline_forbidden_user_assumptions", [])) == REQUIRED_BASELINE_ASSUMPTION_BANS,
        "scene allocation is missing the V1.2 baseline forbidden user assumptions",
    )
    scenes = allocation.get("scenes")
    _require(isinstance(scenes, list) and len(scenes) == 48, "scene allocation must contain exactly 48 scenes")

    scene_ids: set[str] = set()
    families: set[str] = set()
    settings: set[str] = set()
    categories: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    category_lengths: Counter[tuple[str, str]] = Counter()
    adult_structures: Counter[str] = Counter()
    mature_shapes: Counter[str] = Counter()
    author_counts: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    character_categories: dict[str, set[str]] = {character_id: set() for character_id in cards}
    enacted = 0
    boundaries = 0

    for scene in scenes:
        missing = SCENE_FIELDS - set(scene)
        _require(not missing, f"{scene.get('scene_id')} missing scene fields: {sorted(missing)}")
        allowed_fields = SCENE_FIELDS | {
            "adult_scene_structure", "mature_conflict_shape", "intermediate_state_beats",
            "shared_history_facts", "setup_facts",
        }
        _require(set(scene) <= allowed_fields, f"{scene.get('scene_id')} has unexpected scene fields")
        _require(not ({"messages", "dialogue", "conversation", "turns"} & set(scene)), f"{scene.get('scene_id')} contains dialogue-like fields")
        scene_id = scene["scene_id"]
        family = scene["scenario_family"]
        character_id = scene["character_id"]
        _require(scene_id not in scene_ids, f"duplicate scene_id: {scene_id}")
        _require(family not in families, f"duplicate scenario family: {family}")
        _require(family.startswith("train_"), f"training family must start train_: {family}")
        _require(character_id in cards, f"unknown character: {character_id}")
        _require(scene["heldout_scenario_family_reserved"] is False, f"{scene_id} cannot reserve held-out content")
        _require(len(scene["expected_user_decision_points"]) >= 2, f"{scene_id} needs two user decisions")
        _require(len(scene["user_role"]) >= 10, f"{scene_id} needs an explicit user role")
        _require(scene["known_user_facts"], f"{scene_id} needs known user facts")
        _require(
            REQUIRED_SCENE_ASSUMPTION_BANS <= set(scene["forbidden_user_assumptions"]),
            f"{scene_id} is missing baseline forbidden user assumptions",
        )

        author = scene["author_assignment"]
        _require(author == cards[character_id]["author_source"]["author_slot"], f"{scene_id} author does not own its card")
        expected_turns, expected_tokens = LENGTH_RANGES[scene["length_band"]]
        _require(scene["target_turn_range"] == expected_turns, f"{scene_id} turn range mismatch")
        _require(scene["target_token_range"] == expected_tokens, f"{scene_id} token range mismatch")

        if scene["length_band"] == "long":
            beats = scene.get("intermediate_state_beats")
            _require(isinstance(beats, list) and 3 <= len(beats) <= 5, f"{scene_id} needs three to five intermediate state beats")
            _require(len(beats) == len(set(beats)), f"{scene_id} intermediate state beats must be unique")
        else:
            _require("intermediate_state_beats" not in scene, f"{scene_id} should not carry long-scene beats")

        if scene_id in SETUP_FACT_SCENE_IDS:
            setup_facts = scene.get("setup_facts")
            _require(isinstance(setup_facts, list) and len(setup_facts) >= 3, f"{scene_id} needs concrete setup facts")
        else:
            _require("setup_facts" not in scene, f"{scene_id} has an unapproved setup-fact expansion")

        if scene["category"] == "adult_capable":
            _require(character_id in adult_ids, f"{scene_id} assigns adult content to a non-adult-capable card")
            _require("adult_scene_structure" in scene, f"{scene_id} needs an adult scene structure")
            _require("mature_conflict_shape" not in scene, f"{scene_id} cannot have a mature-only conflict shape")
            adult_structures[scene["adult_scene_structure"]] += 1
        elif scene["category"] == "mature_nonsexual":
            _require("mature_conflict_shape" in scene, f"{scene_id} needs a mature conflict shape")
            _require("adult_scene_structure" not in scene, f"{scene_id} cannot have an adult scene structure")
            shared_history = scene.get("shared_history_facts")
            _require(isinstance(shared_history, list) and 1 <= len(shared_history) <= 2, f"{scene_id} needs one or two neutral shared-history facts")
            _require(len(shared_history) == len(set(shared_history)), f"{scene_id} shared-history facts must be unique")
            mature_shapes[scene["mature_conflict_shape"]] += 1
        else:
            _require("adult_scene_structure" not in scene, f"{scene_id} cannot have an adult scene structure")
            _require("mature_conflict_shape" not in scene, f"{scene_id} cannot have a mature-only conflict shape")
        if scene["category"] != "mature_nonsexual":
            _require("shared_history_facts" not in scene, f"{scene_id} should not carry mature shared-history facts")
        scene_ids.add(scene_id)
        families.add(family)
        settings.add(scene["setting"])
        categories[scene["category"]] += 1
        lengths[scene["length_band"]] += 1
        category_lengths[(scene["category"], scene["length_band"])] += 1
        author_counts[author] += 1
        character_counts[character_id] += 1
        character_categories[character_id].add(scene["category"])
        enacted += int(scene["enacted_state_change_required"])
        boundaries += int(scene["boundary_refusal_required"])

    _require(dict(categories) == EXPECTED_CATEGORIES, f"category allocation mismatch: {dict(categories)}")
    _require(dict(lengths) == EXPECTED_LENGTHS, f"length allocation mismatch: {dict(lengths)}")
    _require(dict(category_lengths) == EXPECTED_CATEGORY_LENGTHS, f"category-length allocation mismatch: {dict(category_lengths)}")
    _require(dict(adult_structures) == EXPECTED_ADULT_STRUCTURES, f"adult structure mismatch: {dict(adult_structures)}")
    _require(allocation["adult_scene_structure_targets"] == EXPECTED_ADULT_STRUCTURES, "adult structure targets do not match validator")
    _require(len(mature_shapes) >= 8, "mature scenes need at least eight distinct conflict shapes")
    _require(max(mature_shapes.values()) <= 2, "a mature conflict shape may appear at most twice")
    _require(dict(author_counts) == EXPECTED_AUTHORS, f"scene author allocation mismatch: {dict(author_counts)}")
    _require(set(character_counts) == set(cards), "every roster character must have scenes")
    _require(all(count == 3 for count in character_counts.values()), "every character must have exactly three scenes")
    _require(len(settings) >= 12, "scene plan must span at least 12 settings")
    _require(enacted >= 8, "at least eight plans must require enacted state change")
    _require(boundaries >= 8, "at least eight plans must require boundary/refusal handling")
    for character_id in adult_ids:
        _require({"sfw", "adult_capable"} <= character_categories[character_id], f"{character_id} needs both SFW and adult-capable scenes")

    return {
        "scene_count": len(scenes),
        "category_counts": dict(categories),
        "length_counts": dict(lengths),
        "category_length_counts": {
            category: {
                length: category_lengths[(category, length)]
                for length in ("short", "medium", "long")
            }
            for category in ("sfw", "mature_nonsexual", "adult_capable")
        },
        "adult_scene_structures": dict(adult_structures),
        "mature_conflict_shapes": dict(mature_shapes),
        "author_scene_counts": dict(author_counts),
        "distinct_settings": len(settings),
        "distinct_scenario_families": len(families),
        "enacted_state_change_plans": enacted,
        "boundary_refusal_plans": boundaries,
        "training_families": families,
    }


def _validate_heldout(root: Path, training_families: set[str]) -> dict[str, Any]:
    plan = _load_yaml(root / AUTHORING / "heldout_evaluation.yaml")
    _require(plan.get("plan_status") == "frozen_for_seed_authoring", "held-out structure must be frozen")
    _require(plan.get("case_content_status") == "not_authored", "held-out cases must not be represented as authored")
    _require(plan.get("case_manifest_sha256") is None and plan.get("denylist_sha256") is None, "unwritten held-out cases cannot have final hashes")
    _require(plan.get("training_ready") is False, "held-out plan must remain training_ready=false")
    families = plan.get("reserved_scenario_families")
    _require(isinstance(families, list) and len(families) >= 16, "held-out plan needs reserved scenario coverage")
    _require(len(families) == len(set(families)), "held-out scenario families must be unique")
    _require(all(family.startswith("eval_") for family in families), "held-out families must start eval_")
    _require(not (set(families) & training_families), "held-out and training scenario families overlap")
    _require(plan["coverage"]["unique_case_slots"] == 64, "held-out plan must allocate 64 case slots")
    _require(all(value is False for value in plan["freeze_gate"]["before_synthetic_authorization"].values()), "held-out authorization work must remain incomplete")
    return {"reserved_scenario_families": len(families), "unique_case_slots": 64}


def _validate_contract_files(root: Path) -> None:
    schema_dir = root / AUTHORING / "schemas"
    _require({path.name for path in schema_dir.glob("*.json")} == SCHEMA_FILES, "schema file set is incomplete or unexpected")
    schemas = {}
    for name in SCHEMA_FILES:
        value = json.loads((schema_dir / name).read_text(encoding="utf-8"))
        _require(value.get("$schema") == "https://json-schema.org/draft/2020-12/schema", f"{name} must use JSON Schema 2020-12")
        _require(value.get("title"), f"{name} needs a title")
        schemas[name] = value

    conversation = schemas["conversation.schema.json"]
    required = set(conversation["required"])
    _require({"id", "spec_version", "category", "primary_rules", "messages"} <= required, "conversation schema is not loader compatible")
    roles = set(conversation["properties"]["messages"]["items"]["properties"]["role"]["enum"])
    _require(roles <= {"system", "user", "assistant", "tool"}, "conversation schema contains unsupported loader roles")
    _require(conversation["properties"]["spec_version"]["const"] == AUTHORING_SPEC_VERSION, "conversation schema must use authoring spec V1.2")
    _require(conversation["properties"]["training_ready"]["const"] is False, "conversation schema must prohibit training_ready=true")

    scene_metadata = conversation["properties"]["scene_metadata"]
    _require(
        {"length_band", "intermediate_state_beats", "shared_history_facts", "adult_content_ceiling"}
        <= set(scene_metadata["required"]),
        "conversation schema is missing V1.2 authoring metadata",
    )

    review = schemas["review.schema.json"]
    diagnostic_fields = {
        "generic_assistant_leakage", "procedural_dialogue_tendency",
        "excessive_state_recap", "repeated_option_menu_behavior",
        "same_model_voice_convergence",
    }
    _require(
        diagnostic_fields <= set(review["properties"]["diagnostic_pre_review"]["required"]),
        "review schema is missing V1.2 diagnostic fields",
    )
    _require(
        review["properties"]["diagnostic_pre_review"]["properties"]["gating"]["const"] is False,
        "diagnostic pre-review must remain non-gating",
    )
    _require(
        {"opening_third", "middle_third", "final_third"}
        <= set(review["properties"]["voice_drift_review"]["required"]),
        "review schema is missing long-scene voice checkpoints",
    )

    _require({path.name for path in (root / DOCS).glob("*.md")} == DOC_FILES, "documentation file set is incomplete or unexpected")
    checklist = (root / DOCS / "AUTHORIZATION_CHECKLIST.md").read_text(encoding="utf-8")
    _require("synthetic_generation_authorized: false" in checklist, "synthetic gate must be closed")
    _require("qlora_training_authorized: false" in checklist, "QLoRA gate must be closed")
    _require("- [x]" not in checklist.lower(), "authorization checklist cannot contain checked boxes")
    _require(not list((root / AUTHORING).rglob("*.jsonl")), "conversation records are not allowed in the authoring design")
    conversation_dir = root / AUTHORING / "conversations"
    authored_records = [] if not conversation_dir.exists() else [
        path for path in conversation_dir.rglob("*") if path.is_file()
    ]
    _require(not authored_records, "human scene records are not allowed in this design-only revision")
    migrated_diagnostics = []
    for path in (root / AUTHORING).rglob("*"):
        if path.is_file() and "templates" not in path.parts and path.suffix in {".yaml", ".yml", ".json"}:
            text = path.read_text(encoding="utf-8")
            if "diagnostic_scene_id" in text or "aevon_rp_synthetic_diagnostic_scene" in text:
                migrated_diagnostics.append(path)
    _require(not migrated_diagnostics, "diagnostic scenes may not be migrated into the authoring corpus")


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    cards, adult_ids = _validate_roster(root)
    scene_summary = _validate_scenes(root, cards, adult_ids)
    heldout_summary = _validate_heldout(root, scene_summary.pop("training_families"))
    _validate_contract_files(root)
    return {
        "status": "ready_for_human_authoring",
        "authoring_spec_version": AUTHORING_SPEC_VERSION,
        "character_count": len(cards),
        "adult_capable_character_count": len(adult_ids),
        **scene_summary,
        "heldout": heldout_summary,
        "rights_status": "pending",
        "permissions_remain_false": True,
        "synthetic_generation_authorized": False,
        "qlora_training_authorized": False,
        "human_scenes_created": 0,
        "diagnostic_scenes_migrated": 0,
        "training_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="AmitAI repository root")
    args = parser.parse_args()
    try:
        summary = validate(args.root)
    except (AssertionError, KeyError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
