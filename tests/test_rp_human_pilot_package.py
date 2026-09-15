"""Regressions for the unassigned RP seed V1.2 real-human pilot package."""

from pathlib import Path

import yaml

from scripts.validate_rp_human_pilot import EXPECTED_SCENES, validate

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/sft/rp_seed_v1/human_pilot_v1_2"


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_human_pilot_has_exact_scene_set_and_closed_gates():
    result = validate(ROOT)

    assert result == {
        "status": "ready_for_real_human_assignment",
        "pilot_id": "rp_seed_v1_2_bounded_human_pilot",
        "authoring_spec_version": "1.2.0",
        "pilot_scene_count": 4,
        "selected_scene_ids": list(EXPECTED_SCENES),
        "author_placeholders": ["author_slot_02", "author_slot_03"],
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


def test_all_scene_and_provenance_forms_are_blank_and_unassigned():
    for scene_id in EXPECTED_SCENES:
        packet_dir = PILOT / "packets" / scene_id
        submission = _load_yaml(packet_dir / "scene_submission.template.yaml")
        provenance = _load_yaml(packet_dir / "provenance.template.yaml")
        self_check = _load_yaml(packet_dir / "author_self_check_pre_review.template.yaml")

        assert submission["id"] is None
        assert submission["messages"] == []
        assert submission["training_ready"] is False
        assert provenance["author"]["author_id"] is None
        assert provenance["ownership"]["declaration_status"] == "pending"
        assert provenance["training_ready"] is False
        assert self_check["author_id"] is None
        assert self_check["human_quality_approval"] == "pending"
        assert self_check["human_rights_and_provenance_approval"] == "pending"
        assert self_check["training_ready"] is False


def test_only_jun_packet_has_the_approved_adult_content_ceiling():
    ceilings = {
        scene_id: _load_yaml(PILOT / "packets" / scene_id / "author_packet.yaml")["author_focus"]["adult_content_ceiling"]
        for scene_id in EXPECTED_SCENES
    }

    assert ceilings == {
        "scene_rowan_runaway_puppet": "not_applicable",
        "scene_senka_diverging_destinations": "not_applicable",
        "scene_jun_recording_off": "fade_to_black_no_graphic_sex",
        "scene_safiya_weather_veto": "not_applicable",
    }


def test_pilot_contains_no_diagnostic_transcripts_or_authored_dialogue():
    assert not list(PILOT.rglob("*.jsonl"))
    assert not list(PILOT.rglob("*transcript*"))
    for path in PILOT.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert "diagnostic_scene_id" not in text
        assert "aevon_rp_synthetic_diagnostic_scene" not in text
        if path.suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
            if isinstance(value, dict) and "messages" in value:
                assert value["messages"] == []
