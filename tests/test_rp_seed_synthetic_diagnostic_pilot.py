"""Regressions for the isolated RP synthetic diagnostic pilot."""

import json
from pathlib import Path

import pytest
import yaml

from scripts.validate_rp_seed_authoring import validate as validate_human_authoring
from scripts.validate_rp_seed_diagnostic_pilot import validate as validate_diagnostic
from training.data import normalize_example

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "data/sft/rp_seed_v1/diagnostics/synthetic_pilot_v1"
AUTHORING = ROOT / "data/sft/rp_seed_v1/authoring"


def test_diagnostic_pilot_has_exact_selection_composition_and_closed_gates():
    result = validate_diagnostic(ROOT)

    assert result == {
        "status": "valid_synthetic_diagnostic_only",
        "pilot_id": "synthetic_pilot_v1",
        "scene_count": 8,
        "distinct_character_count": 8,
        "content_tiers": {"adult_capable": 2, "mature_nonsexual": 2, "sfw": 4},
        "length_bands": {"long": 2, "medium": 4, "short": 2},
        "total_turns": 184,
        "approximate_tokens": 12556,
        "provenance_type": "synthetic_diagnostic",
        "human_reviews_completed": 0,
        "synthetic_expansion_authorized": False,
        "qlora_training_authorized": False,
        "training_ready": False,
    }


def test_diagnostic_conversations_are_not_sft_loader_compatible():
    paths = sorted((DIAGNOSTIC / "conversations").glob("*.yaml"))
    assert len(paths) == 8
    assert not list(DIAGNOSTIC.rglob("*.jsonl"))

    for path in paths:
        item = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "transcript" in item
        assert "messages" not in item
        assert item["diagnostic_only"] is True
        assert item["training_ready"] is False
        with pytest.raises(ValueError):
            normalize_example(item)


def test_diagnostic_provenance_cannot_claim_human_or_training_status():
    schema = json.loads(
        (DIAGNOSTIC / "schemas/synthetic_diagnostic_provenance.schema.json").read_text(
            encoding="utf-8"
        )
    )
    properties = schema["properties"]
    assert properties["provenance_type"]["const"] == "synthetic_diagnostic"
    assert properties["training_ready"]["const"] is False
    assert properties["human_authorship"]["properties"]["claimed"]["const"] is False
    assert properties["human_authorship"]["properties"]["author_id"]["type"] == "null"
    assert all(
        value["const"] is False
        for value in properties["gate_exclusions"]["properties"].values()
    )

    for path in (DIAGNOSTIC / "provenance").glob("*.provenance.yaml"):
        item = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert item["human_authorship"] == {
            "claimed": False,
            "author_id": None,
            "ownership_declaration": None,
            "permission_record": None,
        }
        assert item["review_status"]["human_reviewer_ids"] == []
        assert item["rights_and_permissions"]["training_use_authorized"] is False


def test_human_authoring_provenance_meanings_and_allocation_gates_remain_closed():
    human_schema = json.loads(
        (AUTHORING / "schemas/provenance.schema.json").read_text(encoding="utf-8")
    )
    assert human_schema["properties"]["lineage"]["properties"]["origin"]["enum"] == [
        "human_original",
        "human_revision",
        "synthetic_derivative",
    ]

    result = validate_human_authoring(ROOT)
    assert result["status"] == "ready_for_human_authoring"
    assert result["scene_count"] == 48
    assert result["synthetic_generation_authorized"] is False
    assert result["qlora_training_authorized"] is False
    assert result["training_ready"] is False
