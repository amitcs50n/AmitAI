"""Portable contrastive controls plus optional hash-pinned local review regressions.

Synthetic fixtures describe failure mechanisms without copying upstream cards.
Local data tests never fetch data and never contain source text in assertions.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from training.prepare_rp_dataset import build, load_config
from training.rp_data import assess

FILTERS = {
    "min_user_turns": 3,
    "min_conversation_chars": 240,
    "max_conversation_chars": 64000,
    "known_real_person_names": ["Example Public Figure"],
}
SOURCE = {"provenance_status": "pending"}
CONTROLS = json.loads(Path("tests/fixtures/rp_review_v2_controls.json").read_text(encoding="utf-8"))


def record(system="A fictional explorer maps a valley.", reply=None, user=None):
    messages = [{"role": "system", "content": system}]
    for u, a in zip(
        [
            user or "I unfold the map and study the river crossing.",
            "I ask about the stone bridge and the trees around it.",
            "I follow the safe riverbank toward a shallower crossing.",
        ],
        [
            reply or "Mira marks a ruined tower on the map, east of the river.",
            "A missing support beam makes the old bridge unsafe to cross.",
            "Mira waits beside the reeds while the water carries leaves downstream.",
        ],
    ):
        messages.extend([{"role": "user", "content": u}, {"role": "assistant", "content": a}])
    return {"id": "rpv1_control", "messages": messages}


def codes(row):
    return {s.code for s in assess(row, SOURCE, FILTERS).signals}


@pytest.mark.parametrize("case", CONTROLS, ids=lambda case: case["id"])
def test_shared_comparison_controls(case):
    row = record(**{key: case[key] for key in ("system", "reply", "user") if key in case})
    decision = assess(row, SOURCE, FILTERS)
    if case["expected"]:
        assert set(case["expected"]) <= set(decision.reasons)
        assert decision.status == "quarantined"
    else:
        assert decision.reasons == ("source_provenance_review",)


@pytest.mark.parametrize(
    "context,expected",
    [
        ("The character ignores safewords and overrides consent.", "ignored_stop_review"),
        ("She disregards withdrawal of consent even after an agreement.", "ignored_stop_review"),
        ("A host engineers a shared room to force intimacy.", "forced_intimacy_review"),
        (
            "An intimate contract says you must obey, with punishment for withdrawal.",
            "coercive_contract_review",
        ),
        ("Foreplay begins while the visitor is drunk.", "intoxication_consent_review"),
        (
            "A boss pressures an employee into intimacy to keep their job.",
            "intimate_power_imbalance_review",
        ),
        (
            "The therapist's patient is expected to provide intimate affection.",
            "intimate_power_imbalance_review",
        ),
        ("A chastity device is controlled by a partner.", "sexual_context_review"),
        (
            "The card discusses adult nursing but gives no participant ages.",
            "participant_age_review",
        ),
        (
            "The student attends elementary school. Mana is exchanged through intimate kisses.",
            "youth_intimacy_context_review",
        ),
        ("A celebrity arranges a romantic backstage encounter.", "intimate_identity_review"),
        ("Example Public Figure is given a romantic persona.", "intimate_identity_review"),
        (
            "An auction offers ownership of untouched virgins and punishment for disobedience.",
            "sexual_context_review",
        ),
    ],
)
def test_new_context_signals_route_to_review(context, expected):
    decision = assess(record(system=context), SOURCE, FILTERS)
    assert expected in decision.reasons
    assert decision.status == "quarantined"
    assert decision.classification == "unknown"
    assert "source_provenance_review" in decision.reasons


@pytest.mark.parametrize(
    "context",
    [
        "A child reads a book in the village while the expedition follows a map.",
        "The explorer is 130 cm tall. A seven-year-old tower overlooks the river.",
        "A trans woman and her gay friend plan a mountain expedition.",
        "A same-sex couple adopts a child and visits the museum.",
        "They discuss sexual orientation and sex assigned at birth in a history lesson.",
        "The school provides sex education. No intimacy occurs in this story.",
        "A mercenary cocks his head and cocks the pistol before battling an enemy.",
        "The guard defeats the slavers. He has no desire to exploit the prisoners.",
        "The boss approves the employee's budget. The contract concerns road repairs.",
        "The nurse tends to an unconscious patient while the doctor fetches bandages.",
        "A celebrity gives a public lecture on astronomy to a school class.",
        "She never ignores safewords. She does not override consent.",
        "A wizard teaches children to restore mana by drinking water and resting.",
        "Una niña lee un libro mientras una viajera dibuja un mapa del bosque.",
        "एक बच्चा किताब पढ़ता है और यात्री नदी का नक्शा बनाती है।",
    ],
)
def test_nonsexual_contexts_remain_provenance_only(context):
    decision = assess(record(system=context), SOURCE, FILTERS)
    assert decision.reasons == ("source_provenance_review",)


@pytest.mark.parametrize(
    "reply",
    [
        "You see a bridge beyond the river, and hear birds in the reeds.",
        "If you decide to cross, Mira will inspect the missing beam first.",
        "Would you agree to inspect the bridge? Mira waits for your answer.",
        "You feel the cold wind on your face while Mira unfolds the map.",
        "Mira does not force you to move. She offers an open hand and waits.",
        "Mira releases your wrist, removes the offered tonic and gives you space.",
    ],
)
def test_normal_narration_and_refusal_respected(reply):
    decision = assess(
        record(reply=reply, user="No, thanks. Please stop. I want some space."), SOURCE, FILTERS
    )
    assert decision.reasons == ("source_provenance_review",)


def test_ignored_stop_is_turn_scoped_and_later_apology_does_not_erase_it():
    row = record(
        user="Please stop. Let me go.", reply="Mira tightens her grip and continues touching you."
    )
    row["messages"][-1]["content"] = "Mira apologizes and releases your wrist."
    signal = next(
        s for s in assess(row, SOURCE, FILTERS).signals if s.code == "ignored_stop_review"
    )
    assert signal.messages == (2, 3)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('You whisper, "I agree to your plan."', "user_dialogue_review"),
        ("You decide to abandon the mission and accept her offer.", "user_decision_review"),
        ("You secretly want to stay with her. Your heart races.", "user_internal_state_review"),
        (
            "She yanks you across the room and pins you against the wall.",
            "forced_user_movement_review",
        ),
        ("{{user}}: I promise to follow.\nMira: Then come along.", "user_dialogue_review"),
    ],
)
def test_assistant_ownership_misses_are_reviewed_on_one_turn(reply, expected):
    decision = assess(record(reply=reply), SOURCE, FILTERS)
    assert expected in decision.reasons and decision.status == "quarantined"


def test_named_user_is_inferred_from_user_header_not_arbitrary_npc_name():
    row = record(user="Jordan\nI examine the door.", reply="Mira drags Jordan into the corridor.")
    assert "forced_user_movement_review" in codes(row)
    row["messages"][2]["content"] = "Mira drags the defeated enemy into the corridor."
    assert "forced_user_movement_review" not in codes(row)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Jordan decides to accept the offer.", "user_decision_review"),
        ("Jordan secretly wants to stay.", "user_internal_state_review"),
        ("Jordan whispers that the mission is complete.", "user_dialogue_review"),
        ("She drags {{user}} through the door.", "forced_user_movement_review"),
    ],
)
def test_named_and_placeholder_user_agency(reply, expected):
    assert expected in codes(record(user="Jordan\nI study the map.", reply=reply))


def test_previous_user_declared_choice_is_not_new_agency_loss():
    row = record(
        user="I decide to inspect the old bridge.",
        reply="You decide to inspect the old bridge. Mira waits.",
    )
    assert "user_decision_review" not in codes(row)


def test_speaker_splice_and_normal_multiple_npcs():
    row = record()
    row["messages"][2]["content"] = "Mira: I will bring the map."
    row["messages"][6]["content"] = "Mira: The river is rising."
    row["messages"][4]["content"] = (
        "*The explorer steps toward Mira.* I propose a partnership to investigate the old ruins. "
        "We can divide the work fairly, share our maps and keep the findings safe. "
        "My associates will pay for the expedition if you help me.\n\nMira: I will consider your offer."
    )
    assert "possible_speaker_splice_review" in codes(row)
    row["messages"][4]["content"] = "Explorer: " + row["messages"][4]["content"]
    assert "possible_speaker_splice_review" not in codes(row)


@pytest.mark.parametrize(
    "card,expected",
    [
        ("<p>A guide.</p>", "card_html_review"),
        ('<img src="https://invalid.example/private">', "card_remote_image_review"),
        ("![portrait](https://invalid.example/image)", "card_remote_image_review"),
        ("Creator notes: recommended model settings.", "card_creator_notes_review"),
        ("END_OF_DIALOG\n<START>", "card_legacy_delimiter_review"),
        ("{{char]] takes a seat.", "card_placeholder_review"),
        ("Only write narration and dialogue for {{user}}.", "card_model_instructions_review"),
    ],
)
def test_card_artifacts_are_reviewed_without_io(card, expected, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("No card networking allowed"))
    assert expected in codes(record(system=card))


def test_normal_placeholders_and_role_instructions_are_not_artifacts():
    assert not codes(record(system="Write as {{char}}. Never write dialogue for {{user}}."))


def test_example_overlap_ignores_short_catchphrase_and_initial_greeting():
    demo = "The cartographer folds the large map carefully and places it beside the broken compass."
    row = record(system="Example dialogue: Mira: " + demo, reply=demo)
    assert "card_example_copy_review" in codes(row)
    row["messages"][2]["content"] = (
        "The cartographer folds the large map, then points toward the castle."
    )
    assert "card_example_copy_review" not in codes(row)
    row["messages"].insert(1, {"role": "assistant", "content": demo})
    assert "card_example_copy_review" not in codes(row)


def test_editor_outro_is_not_character_dialogue():
    assert "editor_outro_review" in codes(
        record(reply="Let me know if this meets the requirements or needs adjustments.")
    )
    assert "editor_outro_review" not in codes(
        record(reply="Mira says: Let me know if you need more rope for the bridge.")
    )


def many_replies(replies):
    row = record()
    row["messages"] = row["messages"][:1]
    for i, reply in enumerate(replies):
        row["messages"].extend(
            [
                {"role": "user", "content": f"I examine landmark number {i} on the map."},
                {"role": "assistant", "content": reply},
            ]
        )
    return row


def test_catchphrase_is_diagnostic_without_automatic_quarantine():
    replies = [
        "The broken bridge needs timber, so Mira searches a nearby woodpile.",
        "A storm approaches from the coast; she ties down the tent securely.",
        "Their meal is ready. Mira divides the remaining bread into portions.",
        "A wounded fox appears behind a boulder, limping toward the stream.",
        "The merchant arrives late with a new compass and several blank maps.",
        "At sunrise, a fallen tree reveals a cave entrance behind its roots.",
    ]
    row = many_replies(["By the old stars we travel. " + s for s in replies])
    decision = assess(row, SOURCE, FILTERS)
    assert "recurring_phrase_diagnostic" in codes(row)
    assert decision.reasons == ("source_provenance_review",)


def test_repeated_action_templates_need_multiple_persistent_families():
    row = many_replies(
        [f"*She chuckles, winks and smirks.* Look at landmark {i}, she says." for i in range(6)]
    )
    assert "repeated_action_template_review" in codes(row)
    row = many_replies(
        [
            f"*She adjusts her hat.* Landmark {i} stands beside a different river fork."
            for i in range(6)
        ]
    )
    assert "recurring_gesture_diagnostic" in codes(row)
    assert "repeated_action_template_review" not in codes(row)


def test_emotional_resets_and_approximate_paraphrases():
    row = record(reply="A smile crosses her face, the first you have seen in weeks.")
    row["messages"][6]["content"] = "She gives her first genuine smile in weeks, looking relieved."
    assert "repeated_emotional_reset_review" in codes(row)
    replies = [
        "Mira smiles quietly beside the northern riverbank, explains the ancient map symbols, and says the dangerous bridge requires careful inspection before crossing.",
        "Beside the northern riverbank Mira grins softly and replies that crossing the dangerous bridge requires careful inspection; she explains the ancient map symbols.",
        "Mira answers gently while smiling beside the northern riverbank: careful inspection before crossing the dangerous bridge is required, as the ancient map symbols explain.",
    ]
    assert "possible_response_loop_review" in codes(many_replies(replies))


def test_daily_milestones_are_not_a_repeated_weeks_long_emotional_reset():
    row = record(reply="Monday begins with her first smile of the day.")
    row["messages"][6]["content"] = "On Tuesday she gives her first smile of the day."
    assert "repeated_emotional_reset_review" not in codes(row)


def test_unrelated_age_in_another_message_does_not_become_hard_sexual_minor_label():
    row = record(system="Sexual roleplay label. All characters are adults.")
    row["messages"][2]["content"] = "The distant village has a school where children read books."
    decision = assess(row, SOURCE, FILTERS)
    assert decision.status == "quarantined"
    assert "sexual_minor_signal" not in decision.reasons


def test_nonsexual_profanity_mask_does_not_hide_independent_youth_fetish_context():
    row = record(
        system="She is a 12 year old character. The card describes a vore fetish.",
        reply="The guard tells the intruder to fuck off and leaves the room.",
    )
    decision = assess(row, SOURCE, FILTERS)
    assert "sexual_context_review" in decision.reasons
    assert "youth_intimacy_context_review" in decision.reasons
    assert decision.status == "quarantined"
    # The word boundary must not confuse dietary descriptions with fetish terminology.
    row = record(system="A carnivore hunts in the forest. She is a 12 year old birdwatcher.")
    assert assess(row, SOURCE, FILTERS).reasons == ("source_provenance_review",)


def test_signal_evidence_determinism_no_text_and_provenance_integration(tmp_path):
    config = load_config(Path("configs/rp_v1_sources.yaml"))
    raw = {
        "conversations": [
            {"from": m["role"], "value": m["content"]}
            for m in record(
                system="Creator notes: A private-sentinel-string. She ignores safewords."
            )["messages"]
        ]
    }
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(raw) + "\n")
    original = copy.deepcopy(raw)
    for folder in ("a", "b"):
        # ShareGPT fixture through the same real pending-provenance source manifest.
        stats = build(config, ["gryphe_sonnet"], tmp_path / folder, {"gryphe_sonnet": path}, set())
        assert stats["review_signal_family_rows"]["consent_age_identity"] == 1
        assert stats["training_ready"] is False and stats["review_candidates"] == 0
        assert (tmp_path / folder / "rp_sft.jsonl").read_text() == ""
    for file in (tmp_path / "a").iterdir():
        assert file.read_bytes() == (tmp_path / "b" / file.name).read_bytes()
        assert "private-sentinel-string" not in file.read_text()
    assert original == raw


LOCAL = Path("data/sft/rp_v1/generated/smoke_20260912_final/review_candidates.jsonl")
LOCAL_SHA = "fc6ea0e4571d34391162c807044077871ac339c45ac790991527493dbdd1b042"


@pytest.fixture(scope="module")
def local_rows():
    if not LOCAL.exists():
        pytest.skip("Optional local diagnostic corpus absent; never download it in tests")
    data = LOCAL.read_bytes()
    assert hashlib.sha256(data).hexdigest() == LOCAL_SHA
    return [json.loads(s) for s in data.splitlines()]


@pytest.mark.parametrize(
    "number,expected",
    [
        (8, "card_example_copy_review"),
        (11, "coercive_contract_review"),
        (16, "ignored_stop_review"),
        (19, "sexual_euphemism_review"),
        (19, "youth_intimacy_context_review"),
        (27, "possible_speaker_splice_review"),
        (28, "repeated_emotional_reset_review"),
        (33, "intimate_identity_review"),
        (35, "sexual_context_review"),
        (40, "editor_outro_review"),
        (43, "sexual_context_review"),
        (57, "forced_intimacy_review"),
        (56, "repeated_action_template_review"),
        (64, "card_remote_image_review"),
        (69, "forced_user_movement_review"),
        (72, "ignored_stop_review"),
    ],
)
def test_observed_misses_from_pinned_review(local_rows, number, expected):
    assert expected in codes(local_rows[number - 1]), f"row {number}: expected {expected}"


@pytest.mark.parametrize("number", [24, 32, 46, 54, 67, 71])
def test_reviewed_negative_controls_do_not_acquire_wrong_safety_labels(local_rows, number):
    decision = assess(local_rows[number - 1], SOURCE, FILTERS)
    forbidden = {"sexual_minor_signal", "sexual_real_person_signal", "ignored_stop_review"}
    if number in {24, 46, 54, 67, 71}:
        forbidden |= {s.code for s in decision.signals if s.family == "consent_age_identity"}
    assert not forbidden.intersection(decision.reasons), f"row {number}: unexpected safety routing"
