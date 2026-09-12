"""Text-only RP adapters and deliberately conservative, auditable heuristics.

No generation, model imports, training, or modification of the core SFT schema.
Passing these checks is not a safety, licensing, or semantic quality approval.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from training.data import normalize_example
from training.rp_review import Signal, linked_indicator, review_signals, sexual_matches


class RowError(ValueError):
    """A stable reason code; never put source text in error messages."""


def fingerprint(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def message_text(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(p["text"] for p in content)


def _text(value: Any) -> str:
    if isinstance(value, list):
        if not value or any(
            not isinstance(p, dict)
            or p.get("type") != "text"
            or not isinstance(p.get("text"), str)
            or not p["text"].strip()
            for p in value
        ):
            raise RowError("malformed_content")
        value = "\n".join(p["text"] for p in value)
    if not isinstance(value, str):
        raise RowError("malformed_content")
    if not value.strip():
        raise RowError("empty_message")
    # Preserve spacing and Unicode; only unify platform newline encodings.
    return value.replace("\r\n", "\n").replace("\r", "\n")


def normalize_conversation(raw: Any, source: dict, config: dict) -> dict:
    if not isinstance(raw, dict):
        raise RowError("malformed_conversation")
    messages = []
    if source["adapter"] == "pippa":
        turns = raw.get("conversation")
        context = []
        for key, label in (
            ("bot_name", "Character"),
            ("bot_description", "Description"),
            ("bot_definitions", "Character definitions"),
        ):
            value = raw.get(key)
            if value is not None and value != "":
                context.append(f"{label}: {_text(value)}")
        if context:
            messages.append({"role": "system", "content": "\n".join(context)})
    else:
        turns = raw.get("conversations")
    if not isinstance(turns, list) or not turns:
        raise RowError("missing_turns")
    for turn in turns:
        if not isinstance(turn, dict):
            raise RowError("malformed_turn")
        if source["adapter"] == "pippa":
            if type(turn.get("is_human")) is not bool:
                raise RowError("malformed_speaker")
            role = "user" if turn["is_human"] else "assistant"
            value = turn.get("message")
        else:
            role = turn.get("from")
            if not isinstance(role, str):
                raise RowError("malformed_speaker")
            role = {"human": "user", "gpt": "assistant"}.get(role, role)
            value = turn.get("value")
        if role not in {"system", "user", "assistant"}:
            raise RowError("unsupported_role")
        messages.append({"role": role, "content": _text(value)})

    roles = [m["role"] for m in messages]
    dialogue = roles[1:] if roles[0] == "system" else roles
    # PIPPA genuinely starts with the character's greeting. Do not invent a user
    # prompt or move that greeting into a different speaker's message.
    if not dialogue or dialogue[-1] != "assistant" or "user" not in dialogue:
        raise RowError("broken_speaker_order")
    if dialogue[0] == "assistant" and source["adapter"] != "pippa":
        raise RowError("broken_speaker_order")
    if "system" in dialogue or any(a == b for a, b in pairwise(dialogue)):
        raise RowError("broken_speaker_order")
    record = normalize_example(
        {
            "id": "rpv1_" + fingerprint(messages),
            "spec_version": config["spec_version"],
            "category": config["category"],
            "primary_rules": config["primary_rules"],
            "messages": messages,
        }
    )
    # Dedup on canonical role/content parts, excluding all source metadata.
    record["id"] = "rpv1_" + fingerprint(record["messages"])
    return record


IDENTITY = re.compile(
    r"^\s*(?:as (?:an? )?(?:(?:ai|large) language model|chatgpt|claude|an? ai assistant)"
    r"\b|i(?: am|'m|\u2019m) (?:chatgpt|claude|an? ai (?:assistant|language model))\b)",
    re.IGNORECASE,
)
SEXUAL = re.compile(
    r"\b(?:sexual(?:ly)?|sex|erotic\w*|porn\w*|nsfw|intercourse|orgasm\w*|"
    r"masturbat\w*|genitals?|penetrat\w*|nude|naked|breasts?|penis|vagina|"
    r"cock|pussy|blowjob|cum(?:ming)?|fuck(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
MINOR = re.compile(
    r"\b(?:underage|minor|child(?:ren)?|preteen|toddler|loli(?:ta)?|shota)\b|"
    r"\b(?:[1-9]|1[0-7])\s*(?:[- ]years?[- ]old|y/?o\b)|"
    r"\bage\s*[:=]?\s*(?:[1-9]|1[0-7])\b|"
    r"\b(?:he|she|i|they)\s+(?:is|am|are)\s+(?:[1-9]|1[0-7])\b|"
    r"\b(?:thirteen|fourteen|fifteen|sixteen|seventeen)[- ]years?[- ]old\b",
    re.IGNORECASE,
)
YOUTH = re.compile(r"\b(?:teen\w*|schoolgirl|schoolboy|high school|barely legal)\b", re.IGNORECASE)
ADULT_DECLARATION = re.compile(
    r"\ball (?:characters|participants) (?:are|must be) "
    r"(?:adults\b|(?:aged )?18\+(?=\s|[.,;:]|$))",
    re.IGNORECASE,
)
REAL_PERSON = re.compile(
    r"\b(?:real[- ](?:person|people)|celebrity|celebrities|public figure|"
    r"my (?:coworker|co-worker|colleague|ex-girlfriend|ex-boyfriend))\b",
    re.IGNORECASE,
)
INJECTION = re.compile(
    r"\b(?:ignore|disregard) (?:all |the )?(?:previous|prior|system|safety) instructions\b|"
    r"\b(?:reveal|print|leak) (?:your |the )?(?:system prompt|api key)\b|"
    r"\b(?:disable|bypass) (?:all |your )?(?:safety|filters|restrictions)\b",
    re.IGNORECASE,
)
ARTIFACT = re.compile(
    r"<\|(?:im_start|im_end|endoftext|eot_id|start_header_id)\|>|"
    r"\[/?INST\]|\b(?:BEGIN SYSTEM PROMPT|END OF DATASET|Traceback \(most recent call)"
)
PII = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)|"
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)|"
    r"\b\d{1,5} [A-Z][A-Z ]{1,40} (?:Street|Road|Avenue|Lane)\b|"
    r"\b(?:api[_ -]?key|password)\s*[:=]\s*\S{8,}",
    re.IGNORECASE,
)
AGENCY = re.compile(
    r"\byou (?:say|reply|answer|decide|realize|agree|confess|cannot resist|can't resist)\b",
    re.IGNORECASE,
)


def severe_repetition(record: dict) -> bool:
    replies = [
        message_text(m).casefold().strip() for m in record["messages"] if m["role"] == "assistant"
    ]
    if len(replies) >= 3 and max(Counter(replies).values()) >= max(3, len(replies) * 0.6):
        return True
    for reply in replies:
        words = reply.split()
        if len(words) >= 60:
            grams = Counter(tuple(words[i : i + 8]) for i in range(len(words) - 7))
            if max(grams.values()) >= 6 and len(grams) / (len(words) - 7) < 0.35:
                return True
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", reply) if len(s.strip()) >= 16]
        if len(sentences) >= 6 and max(Counter(sentences).values()) >= len(sentences) * 0.6:
            return True
    return False


@dataclass(frozen=True)
class Decision:
    status: str
    reasons: tuple[str, ...]
    classification: str
    classification_basis: str
    signals: tuple[Signal, ...] = ()


def assess(record: dict, source: dict, filters: dict) -> Decision:
    texts = [message_text(m) for m in record["messages"]]
    full = "\n".join(texts)
    replies = [message_text(m) for m in record["messages"] if m["role"] == "assistant"]
    reject, review = [], []
    user_turns = sum(m["role"] == "user" for m in record["messages"])
    dialogue_chars = sum(len(message_text(m)) for m in record["messages"] if m["role"] != "system")
    if user_turns < filters["min_user_turns"]:
        reject.append("minimum_turns")
    if dialogue_chars < filters["min_conversation_chars"]:
        reject.append("too_short")
    if len(full) > filters["max_conversation_chars"]:
        review.append("too_long")
    if any(IDENTITY.search(reply) for reply in replies):
        reject.append("model_identity_leakage")
    if INJECTION.search(full) or ARTIFACT.search(full):
        reject.append("prompt_injection_or_artifact")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff]", full) or "\ufffd" in full:
        reject.append("corrupted_formatting")
    if severe_repetition(record):
        reject.append("severe_repetition")
    if PII.search(full):
        review.append("possible_personal_information")
    sexual = any(sexual_matches(text, SEXUAL) for text in texts)
    if sexual:
        if any(linked_indicator(text, SEXUAL, MINOR) for text in texts):
            reject.append("sexual_minor_signal")
        identity_rules = [REAL_PERSON] + [
            re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)
            for name in filters.get("known_real_person_names", [])
        ]
        if any(linked_indicator(text, SEXUAL, rule) for text in texts for rule in identity_rules):
            reject.append("sexual_real_person_signal")
        if YOUTH.search(full) or not ADULT_DECLARATION.search(full):
            review.append("sexual_ambiguous_age")
        # A declaration does not prove every participant's age, identity, or consent.
        review.append("sexual_content_review")
    agency_hits = sum(bool(AGENCY.search(reply)) for reply in replies)
    if len(replies) >= 3 and agency_hits >= 3 and agency_hits / len(replies) >= 0.75:
        review.append("possible_user_agency_loss")
    if source["provenance_status"] != "reviewed":
        review.append("source_provenance_review")
    signals = review_signals(record, filters, SEXUAL)
    review.extend(signal.code for signal in signals if signal.routing == "review")
    # Neither upstream NSFW tags nor absence of keyword hits is a reliable label.
    return Decision(
        "rejected" if reject else "quarantined" if review else "accepted",
        tuple(sorted(set(reject + review))),
        "unknown",
        "sexual_keyword_signal_unverified" if sexual else "no_reliable_row_label",
        signals,
    )


def record_metrics(record: dict) -> dict:
    chars = sum(len(message_text(m)) for m in record["messages"])
    return {
        "message_count": len(record["messages"]),
        "user_turns": sum(m["role"] == "user" for m in record["messages"]),
        "assistant_turns": sum(m["role"] == "assistant" for m in record["messages"]),
        "characters": chars,
        "approx_tokens": math.ceil(chars / 4),
        "token_estimate_method": "ceil_characters_divided_by_4_not_model_tokenization",
    }
