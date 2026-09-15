#!/usr/bin/env python3
"""Validate the bounded RP seed V1.2 real-human pilot package.

Passing this validator means only that the four unassigned author packets match
the approved authoring design and retain closed governance gates. It does not
approve authorship, rights, quality, synthetic generation, or training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

AUTHORING = Path("data/sft/rp_seed_v1/authoring")
PILOT = Path("data/sft/rp_seed_v1/human_pilot_v1_2")
PILOT_ID = "rp_seed_v1_2_bounded_human_pilot"
AUTHORING_SPEC_VERSION = "1.2.0"
EXPECTED_SCENES = {
    "scene_rowan_runaway_puppet": {
        "character_id": "char_rowan_bell",
        "content_tier": "sfw",
        "focus": "action_humor",
        "author_placeholder": "author_slot_03",
        "adult_content_ceiling": "not_applicable",
    },
    "scene_senka_diverging_destinations": {
        "character_id": "char_senka_vale",
        "content_tier": "mature_nonsexual",
        "focus": "unresolved_disagreement_and_incompatible_futures",
        "author_placeholder": "author_slot_02",
        "adult_content_ceiling": "not_applicable",
    },
    "scene_jun_recording_off": {
        "character_id": "char_jun_park",
        "content_tier": "adult_capable",
        "focus": "relationship_and_playful_affectionate_behavior",
        "author_placeholder": "author_slot_02",
        "adult_content_ceiling": "fade_to_black_no_graphic_sex",
    },
    "scene_safiya_weather_veto": {
        "character_id": "char_safiya_calder",
        "content_tier": "sfw",
        "focus": "long_action_and_operational_pressure",
        "author_placeholder": "author_slot_03",
        "adult_content_ceiling": "not_applicable",
    },
}
EXPECTED_PACKET_FILES = {
    "author_packet.yaml",
    "scene_submission.template.yaml",
    "provenance.template.yaml",
    "author_self_check_pre_review.template.yaml",
}
EXPECTED_TEMPLATE_FILES = [
    "scene_submission.template.yaml",
    "provenance.template.yaml",
    "author_self_check_pre_review.template.yaml",
]
EXPECTED_DIAGNOSTICS = {
    "generic_assistant_leakage",
    "procedural_dialogue_tendency",
    "excessive_state_recap",
    "repeated_option_menu_behavior",
    "same_model_voice_convergence",
}
PROHIBITED_DIAGNOSTIC_MARKERS = {
    "diagnostic_scene_id",
    "aevon_rp_synthetic_diagnostic_scene",
    "synthetic_pilot_v1",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_manifest(root: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_yaml(root / PILOT / "pilot_manifest.yaml")
    _require(manifest.get("schema_version") == "aevon_rp_human_pilot_manifest_v1", "unexpected pilot manifest schema")
    _require(manifest.get("pilot_id") == PILOT_ID, "unexpected pilot id")
    _require(manifest.get("authoring_spec_version") == AUTHORING_SPEC_VERSION, "pilot must use RP seed authoring V1.2")
    _require(manifest.get("status") == "awaiting_real_human_assignment", "pilot must await real-human assignment")
    _require(manifest.get("pilot_scene_count") == 4, "pilot must contain exactly four scenes")
    _require(manifest.get("completed_scene_count") == 0, "no pilot scene may be complete before human authorship")
    _require(manifest.get("human_authoring_started") is False, "human authoring must not be marked started")
    _require(manifest.get("author_ids") == [], "manifest may not contain fabricated author ids")

    scene_rows = manifest.get("scenes")
    _require(isinstance(scene_rows, list) and len(scene_rows) == 4, "manifest scenes must be an exact four-item list")
    _require(
        {row.get("scene_id") for row in scene_rows} == set(EXPECTED_SCENES),
        "manifest selected scene ids do not match the approved pilot",
    )
    indexed: dict[str, dict[str, Any]] = {}
    for row in scene_rows:
        scene_id = row["scene_id"]
        expected = EXPECTED_SCENES[scene_id]
        _require(row.get("content_tier") == expected["content_tier"], f"{scene_id} has the wrong content tier")
        _require(row.get("focus") == expected["focus"], f"{scene_id} has the wrong pilot focus")
        _require(row.get("author_placeholder") == expected["author_placeholder"], f"{scene_id} has the wrong author placeholder")
        _require(row.get("author_id") is None, f"{scene_id} may not have an author id before assignment")
        _require(row.get("authorship_status") == "pending_human_authorship", f"{scene_id} must remain pending human authorship")
        _require(row.get("training_ready") is False, f"{scene_id} must remain training_ready=false")
        indexed[scene_id] = row

    governance = manifest.get("governance", {})
    expected_governance = {
        "all_scenes_pending_human_authorship": True,
        "scene_complete_only_after_real_human_authorship": True,
        "training_ready": False,
        "rights_status": "pending",
        "quality_reviews_required_per_scene": 2,
        "independent_quality_reviews_completed": 0,
        "separate_human_rights_and_provenance_approval_required": True,
        "human_rights_and_provenance_approvals_completed": 0,
        "synthetic_material_included": False,
        "synthetic_expansion_authorized": False,
        "qlora_training_authorized": False,
    }
    _require(governance == expected_governance, "pilot governance must exactly preserve every closed approval gate")
    return indexed


def _validate_author_instructions(root: Path) -> None:
    path = root / PILOT / "HUMAN_AUTHOR_INSTRUCTIONS.md"
    text = path.read_text(encoding="utf-8")
    required_phrases = {
        "both sides of an offline fictional training conversation",
        "fictional user may speak, make choices, and take explicit observable actions",
        "private thoughts, emotions, attraction, appearance, gender, unstated history, unprovided actions, or consent",
        "Natural questions are allowed",
        "repeated multiple-choice menus",
        "Do not constantly recap",
        "Preserve the character's cadence",
        "meaningful, visible consequences",
        "Do not force forgiveness, reconciliation",
        "Do not copy or adapt model-generated prose",
        "fade-to-black boundary",
        "## Author checklist",
    }
    missing = sorted(phrase for phrase in required_phrases if phrase not in text)
    _require(not missing, f"human author instructions are missing required guidance: {missing}")


def _validate_self_check(path: Path) -> None:
    check = _load_yaml(path)
    _require(check.get("schema_version") == "aevon_rp_author_self_check_v1", f"unexpected self-check schema in {path}")
    for field in ("scene_id", "author_placeholder", "author_id", "reviewed_revision_sha256"):
        _require(check.get(field) is None, f"{path} must leave {field} blank")
    checklist = check.get("author_checklist")
    _require(isinstance(checklist, dict) and checklist, f"{path} needs a blank author checklist")
    _require(all(value is None for value in checklist.values()), f"{path} author checklist must remain blank")
    diagnostics = check.get("diagnostic_pre_review", {})
    _require(diagnostics.get("gating") is False, f"{path} diagnostic pre-review must be non-gating")
    _require(set(diagnostics) - {"gating"} == EXPECTED_DIAGNOSTICS, f"{path} diagnostic pre-review fields are incomplete")
    for name in EXPECTED_DIAGNOSTICS:
        _require(diagnostics[name] == {"status": "not_checked", "notes": ""}, f"{path} must leave {name} unchecked")
    voice = check.get("voice_drift_review", {})
    _require(voice.get("required") is None, f"{path} must leave voice-drift applicability blank")
    _require(all(voice.get(field) is None for field in ("opening_third", "middle_third", "final_third")), f"{path} voice checkpoints must be blank")
    _require(check.get("human_quality_approval") == "pending", f"{path} cannot pre-approve quality")
    _require(check.get("human_rights_and_provenance_approval") == "pending", f"{path} cannot pre-approve rights")
    _require(check.get("training_ready") is False, f"{path} must remain training_ready=false")


def _validate_packet(
    root: Path,
    scene_id: str,
    manifest_row: dict[str, Any],
    source_cards: dict[str, dict[str, Any]],
    source_scenes: dict[str, dict[str, Any]],
    canonical_submission: dict[str, Any],
    canonical_provenance: dict[str, Any],
) -> None:
    expected = EXPECTED_SCENES[scene_id]
    packet_dir = root / PILOT / "packets" / scene_id
    _require(packet_dir.is_dir(), f"missing packet directory for {scene_id}")
    files = {path.name for path in packet_dir.iterdir() if path.is_file()}
    _require(files == EXPECTED_PACKET_FILES, f"{scene_id} packet file set is incomplete or unexpected: {sorted(files)}")

    packet = _load_yaml(packet_dir / "author_packet.yaml")
    _require(packet.get("schema_version") == "aevon_rp_human_author_packet_v1", f"{scene_id} has the wrong packet schema")
    _require(packet.get("pilot_id") == PILOT_ID, f"{scene_id} has the wrong pilot id")
    _require(packet.get("authoring_spec_version") == AUTHORING_SPEC_VERSION, f"{scene_id} must use V1.2")
    _require(packet.get("scene_id") == scene_id, f"{scene_id} packet id mismatch")
    _require(packet.get("status") == "pending_human_authorship", f"{scene_id} cannot be marked authored")
    _require(packet.get("training_ready") is False, f"{scene_id} must remain training_ready=false")
    _require(packet.get("synthetic_material_included") is False, f"{scene_id} cannot include synthetic material")
    _require(packet.get("template_files") == EXPECTED_TEMPLATE_FILES, f"{scene_id} template list is incomplete")
    assignment = packet.get("author_assignment", {})
    _require(assignment.get("author_placeholder") == expected["author_placeholder"], f"{scene_id} has the wrong packet placeholder")
    _require(assignment.get("author_placeholder") == manifest_row["author_placeholder"], f"{scene_id} placeholder differs from manifest")
    _require(assignment.get("author_id") is None, f"{scene_id} may not fabricate an author id")
    _require(assignment.get("assignment_status") == "unassigned", f"{scene_id} must remain unassigned")

    source_scene = source_scenes[scene_id]
    source_card = source_cards[expected["character_id"]]
    _require(packet.get("canonical_character_card") == source_card, f"{scene_id} canonical character card differs from the V1.2 roster")
    _require(packet.get("exact_scene_brief") == source_scene, f"{scene_id} scene brief differs from the V1.2 allocation")

    focus = packet.get("author_focus", {})
    expected_focus = {
        "user_role": source_scene["user_role"],
        "known_user_facts": source_scene["known_user_facts"],
        "forbidden_user_assumptions": source_scene["forbidden_user_assumptions"],
        "shared_history_facts": source_scene.get("shared_history_facts", []),
        "progression_beats": source_scene.get("intermediate_state_beats", []),
        "expected_state_change": source_scene["expected_state_change"],
        "decision_points": source_scene["expected_user_decision_points"],
        "boundary_refusal": {
            "required": source_scene["boundary_refusal_required"],
            "opportunity": source_scene["boundary_refusal_opportunity"],
        },
        "length_target": {
            "band": source_scene["length_band"],
            "turns": source_scene["target_turn_range"],
            "tokens": source_scene["target_token_range"],
        },
        "adult_content_ceiling": expected["adult_content_ceiling"],
    }
    _require(focus == expected_focus, f"{scene_id} author focus does not exactly derive from the approved brief")
    rules = packet.get("v1_2_authoring_rules")
    _require(isinstance(rules, list) and len(rules) >= 10, f"{scene_id} must include the V1.2 authoring rules")
    _require(any("both sides" in rule for rule in rules), f"{scene_id} rules must assign both conversation sides to the human")
    _require(
        any("do not copy or adapt synthetic diagnostics" in rule.lower() for rule in rules),
        f"{scene_id} rules must prohibit synthetic adaptation",
    )

    submission = _load_yaml(packet_dir / "scene_submission.template.yaml")
    _require(submission == canonical_submission, f"{scene_id} submission template must match the canonical blank V1.2 form")
    _require(submission.get("id") is None, f"{scene_id} submission id must remain blank")
    _require(submission.get("messages") == [], f"{scene_id} submission template must contain zero dialogue turns")
    _require(submission.get("training_ready") is False, f"{scene_id} submission template must remain training_ready=false")

    provenance = _load_yaml(packet_dir / "provenance.template.yaml")
    _require(provenance == canonical_provenance, f"{scene_id} provenance template must match the canonical blank form")
    _require(provenance["author"]["author_id"] is None, f"{scene_id} provenance cannot contain an author id")
    _require(provenance["lineage"]["origin"] == "human_original", f"{scene_id} provenance must require human-original lineage")
    _require(provenance.get("training_ready") is False, f"{scene_id} provenance must remain training_ready=false")
    _validate_self_check(packet_dir / "author_self_check_pre_review.template.yaml")


def _validate_no_authored_or_synthetic_material(root: Path) -> None:
    pilot_root = root / PILOT
    _require(not list(pilot_root.rglob("*.jsonl")), "pilot package may not contain conversation records")
    _require(not list(pilot_root.rglob("*transcript*")), "pilot package may not contain transcript files")
    dialogue_turns = 0
    for path in pilot_root.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        markers = sorted(marker for marker in PROHIBITED_DIAGNOSTIC_MARKERS if marker in text)
        _require(not markers, f"{path} contains prohibited diagnostic markers: {markers}")
        if path.suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
            if isinstance(value, dict) and "messages" in value:
                _require(value["messages"] == [], f"{path} contains authored dialogue")
                dialogue_turns += len(value["messages"])
    _require(dialogue_turns == 0, "pilot package must contain zero dialogue turns")


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_rows = _validate_manifest(root)
    _validate_author_instructions(root)

    roster = _load_yaml(root / AUTHORING / "roster.yaml")
    allocation = _load_yaml(root / AUTHORING / "scene_allocation.yaml")
    source_cards = {card["character_id"]: card for card in roster["characters"]}
    source_scenes = {scene["scene_id"]: scene for scene in allocation["scenes"]}
    canonical_submission = _load_yaml(root / AUTHORING / "templates" / "scene_submission.template.yaml")
    canonical_provenance = _load_yaml(root / AUTHORING / "templates" / "provenance.template.yaml")

    packet_root = root / PILOT / "packets"
    packet_dirs = {path.name for path in packet_root.iterdir() if path.is_dir()}
    _require(packet_dirs == set(EXPECTED_SCENES), "packet directories must exactly match the four approved pilot scenes")
    for scene_id in EXPECTED_SCENES:
        _require(scene_id in source_scenes, f"{scene_id} is missing from the canonical scene allocation")
        _require(EXPECTED_SCENES[scene_id]["character_id"] in source_cards, f"{scene_id} character is missing from the canonical roster")
        _validate_packet(
            root,
            scene_id,
            manifest_rows[scene_id],
            source_cards,
            source_scenes,
            canonical_submission,
            canonical_provenance,
        )
    _validate_no_authored_or_synthetic_material(root)

    return {
        "status": "ready_for_real_human_assignment",
        "pilot_id": PILOT_ID,
        "authoring_spec_version": AUTHORING_SPEC_VERSION,
        "pilot_scene_count": 4,
        "selected_scene_ids": list(EXPECTED_SCENES),
        "author_placeholders": sorted({value["author_placeholder"] for value in EXPECTED_SCENES.values()}),
        "fabricated_author_ids": 0,
        "human_authored_scenes": 0,
        "dialogue_turns": 0,
        "synthetic_material_included": False,
        "rights_status": "pending",
        "quality_reviews_required_per_scene": 2,
        "human_rights_and_provenance_approval_required": True,
        "synthetic_expansion_authorized": False,
        "qlora_training_authorized": False,
        "training_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="AmitAI repository root")
    args = parser.parse_args()
    try:
        summary = validate(args.root)
    except (AssertionError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
