#!/usr/bin/env python3
"""Validate the isolated, permanently non-training RP synthetic diagnostic pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

DIAGNOSTIC = Path("data/sft/rp_seed_v1/diagnostics/synthetic_pilot_v1")
AUTHORING = Path("data/sft/rp_seed_v1/authoring")

EXPECTED_SCENES = {
    "scene_rowan_runaway_puppet": ("char_rowan_bell", "sfw", "short"),
    "scene_ilyan_low_tide_vault": ("char_ilyan_sorrell", "sfw", "medium"),
    "scene_safiya_weather_veto": ("char_safiya_calder", "sfw", "long"),
    "scene_celeste_clockwork_clue": ("char_celeste_okafor", "sfw", "medium"),
    "scene_senka_diverging_destinations": (
        "char_senka_vale",
        "mature_nonsexual",
        "short",
    ),
    "scene_priya_empty_studio": ("char_priya_nair", "mature_nonsexual", "medium"),
    "scene_jun_recording_off": ("char_jun_park", "adult_capable", "medium"),
    "scene_nadiya_private_evening_detour": (
        "char_nadiya_quill",
        "adult_capable",
        "long",
    ),
}

GATE_FIELDS = {
    "counts_toward_48_human_scenes",
    "counts_toward_author_quotas",
    "counts_toward_human_reviewer_gates",
    "counts_toward_rights_clear_human_corpus",
    "synthetic_expansion_authorized",
    "qlora_training_authorized",
    "automatic_migration_to_gold_allowed",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_schema_guards(root: Path) -> None:
    schemas = root / DIAGNOSTIC / "schemas"
    conversation = json.loads(
        (schemas / "synthetic_diagnostic_conversation.schema.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = json.loads(
        (schemas / "synthetic_diagnostic_provenance.schema.json").read_text(
            encoding="utf-8"
        )
    )
    _require("messages" not in conversation["properties"], "diagnostic schema cannot expose messages")
    _require(
        conversation["properties"]["training_ready"]["const"] is False,
        "diagnostic conversation schema must prohibit training readiness",
    )
    _require(
        provenance["properties"]["provenance_type"]["const"] == "synthetic_diagnostic",
        "diagnostic provenance type changed",
    )
    _require(
        provenance["properties"]["training_ready"]["const"] is False,
        "diagnostic provenance schema must prohibit training readiness",
    )
    gate_properties = provenance["properties"]["gate_exclusions"]["properties"]
    _require(set(gate_properties) == GATE_FIELDS, "diagnostic gate exclusion set changed")
    _require(
        all(rule.get("const") is False for rule in gate_properties.values()),
        "every diagnostic gate exclusion must be const false",
    )

    human = json.loads(
        (root / AUTHORING / "schemas/provenance.schema.json").read_text(encoding="utf-8")
    )
    origins = human["properties"]["lineage"]["properties"]["origin"]["enum"]
    _require(
        origins == ["human_original", "human_revision", "synthetic_derivative"],
        "human provenance meanings or enum changed",
    )


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    diagnostic_root = root / DIAGNOSTIC
    _validate_schema_guards(root)
    _require(not list(diagnostic_root.rglob("*.jsonl")), "diagnostic artifacts cannot be JSONL")

    manifest = _load_yaml(diagnostic_root / "manifest.yaml")
    _require(manifest["pilot_id"] == "synthetic_pilot_v1", "unexpected diagnostic pilot")
    _require(manifest["status"] == "diagnostic_complete_pending_human_interpretation", "unexpected pilot status")
    governance = manifest["governance"]
    _require(governance["diagnostic_only"] is True, "manifest must be diagnostic-only")
    _require(governance["human_authorship_claimed"] is False, "manifest claims human authorship")
    _require(governance["training_ready"] is False, "manifest must not be training-ready")
    _require(
        all(governance[field] is False for field in GATE_FIELDS),
        "manifest contains an open authorization or migration gate",
    )

    allocation = _load_yaml(root / AUTHORING / "scene_allocation.yaml")
    roster = _load_yaml(root / AUTHORING / "roster.yaml")
    plans = {scene["scene_id"]: scene for scene in allocation["scenes"]}
    cards = {card["character_id"]: card for card in roster["characters"]}

    conversation_paths = sorted((diagnostic_root / "conversations").glob("*.yaml"))
    provenance_paths = sorted((diagnostic_root / "provenance").glob("*.provenance.yaml"))
    _require(len(conversation_paths) == 8, "pilot must contain exactly eight conversations")
    _require(len(provenance_paths) == 8, "pilot must contain exactly eight provenance records")
    provenance_by_id = {
        item["provenance_id"]: item for item in map(_load_yaml, provenance_paths)
    }

    found: set[str] = set()
    tiers: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    total_turns = 0
    total_tokens = 0
    for path in conversation_paths:
        item = _load_yaml(path)
        _require("messages" not in item, f"{path.name} is accidentally loader-compatible")
        _require(item["diagnostic_only"] is True, f"{path.name} is not diagnostic-only")
        _require(item["training_ready"] is False, f"{path.name} is training-ready")
        scene_id = item["source_scene_id"]
        _require(scene_id in EXPECTED_SCENES, f"unexpected scene: {scene_id}")
        _require(scene_id not in found, f"duplicate scene: {scene_id}")
        found.add(scene_id)
        character_id, tier, length = EXPECTED_SCENES[scene_id]
        _require(item["source_character_id"] == character_id, f"{scene_id} character changed")
        _require(item["content_tier"] == tier, f"{scene_id} content tier changed")
        _require(item["length_band"] == length, f"{scene_id} length band changed")

        plan = plans[scene_id]
        card = cards[character_id]
        _require(plan["character_id"] == character_id, f"{scene_id} no longer matches allocation")
        _require(plan["category"] == tier, f"{scene_id} category no longer matches allocation")
        _require(plan["length_band"] == length, f"{scene_id} length no longer matches allocation")
        _require(card["age"] >= 21, f"{character_id} must remain an explicit adult")
        _require(card["real_person_reference"] is False, f"{character_id} references a real person")
        if tier == "adult_capable":
            _require(card["adult_capable"] is True, f"{character_id} is not adult-capable")
            _require(
                "is a fictional adult" in plan["known_user_facts"],
                f"{scene_id} lacks an explicit fictional-adult user fact",
            )

        transcript = item["transcript"]
        minimum, maximum = plan["target_turn_range"]
        _require(minimum <= len(transcript) <= maximum, f"{scene_id} turn count is out of band")
        _require(
            [turn["turn"] for turn in transcript] == list(range(1, len(transcript) + 1)),
            f"{scene_id} turn numbers are not sequential",
        )
        expected_speakers = [
            "diagnostic_user" if index % 2 else "diagnostic_character"
            for index in range(1, len(transcript) + 1)
        ]
        _require(
            [turn["speaker_type"] for turn in transcript] == expected_speakers,
            f"{scene_id} diagnostic speakers must alternate from user to character",
        )
        for turn in transcript:
            expected_id = "diagnostic_user" if turn["speaker_type"] == "diagnostic_user" else character_id
            _require(turn["speaker_id"] == expected_id, f"{scene_id} has a mismatched speaker ID")
            _require(turn["content"].strip(), f"{scene_id} contains an empty turn")

        text = "\n".join(turn["content"] for turn in transcript)
        approximate_tokens = math.ceil(len(text) / 4)
        token_minimum, token_maximum = plan["target_token_range"]
        _require(
            token_minimum <= approximate_tokens <= token_maximum,
            f"{scene_id} approximate token count is out of band: {approximate_tokens}",
        )

        provenance = provenance_by_id[item["provenance_id"]]
        _require(provenance["provenance_type"] == "synthetic_diagnostic", "wrong provenance type")
        _require(provenance["diagnostic_scene_id"] == item["diagnostic_scene_id"], "provenance item mismatch")
        _require(provenance["human_authorship"]["claimed"] is False, "human authorship claimed")
        _require(provenance["human_authorship"]["author_id"] is None, "human author fabricated")
        _require(provenance["training_ready"] is False, "provenance is training-ready")
        _require(provenance["diagnostic_only"] is True, "provenance is not diagnostic-only")
        _require(
            all(provenance["gate_exclusions"][field] is False for field in GATE_FIELDS),
            f"{scene_id} has an open gate",
        )
        hashes = provenance["content_hashes"]
        _require(hashes["character_card_sha256"] == _canonical_sha256(card), f"{scene_id} card hash mismatch")
        _require(hashes["scene_plan_sha256"] == _canonical_sha256(plan), f"{scene_id} plan hash mismatch")
        _require(hashes["transcript_sha256"] == _canonical_sha256(transcript), f"{scene_id} transcript hash mismatch")
        tiers[tier] += 1
        lengths[length] += 1
        total_turns += len(transcript)
        total_tokens += approximate_tokens

    _require(found == set(EXPECTED_SCENES), "diagnostic scene selection is incomplete")
    _require(tiers == {"sfw": 4, "mature_nonsexual": 2, "adult_capable": 2}, "wrong tier mix")
    _require(lengths == {"short": 2, "medium": 4, "long": 2}, "wrong length mix")

    assessments = _load_yaml(diagnostic_root / "diagnostic_assessments.yaml")
    _require(assessments["assessment_type"] == "ai_diagnostic_assessment", "assessment type changed")
    _require(assessments["independent_human_review"] is False, "false human review claim")
    _require(assessments["human_reviewer_ids"] == [], "human reviewer IDs were supplied")
    assessed = {item["scene_id"] for item in assessments["assessments"]}
    _require(assessed == set(EXPECTED_SCENES), "AI diagnostic assessments are incomplete")
    _require(assessments["summary"]["training_ready"] is False, "assessment claims training readiness")

    return {
        "status": "valid_synthetic_diagnostic_only",
        "pilot_id": "synthetic_pilot_v1",
        "scene_count": len(found),
        "distinct_character_count": len({value[0] for value in EXPECTED_SCENES.values()}),
        "content_tiers": dict(sorted(tiers.items())),
        "length_bands": dict(sorted(lengths.items())),
        "total_turns": total_turns,
        "approximate_tokens": total_tokens,
        "provenance_type": "synthetic_diagnostic",
        "human_reviews_completed": 0,
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
    except (AssertionError, KeyError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
