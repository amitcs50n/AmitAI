from __future__ import annotations

import re
from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SPEC_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
RULE_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*-\d{3}$")


def _require_matching_string(example: dict[str, Any], field: str, pattern: re.Pattern) -> str:
    value = example.get(field)
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{field} has an invalid format: {value!r}")
    return value


def _normalize_primary_rules(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("primary_rules must be a non-empty list")
    if any(
        not isinstance(rule_id, str) or not RULE_ID_PATTERN.fullmatch(rule_id)
        for rule_id in value
    ):
        raise ValueError("primary_rules must contain valid rule IDs")
    if len(value) != len(set(value)):
        raise ValueError("primary_rules must not contain duplicates")
    return value


def _normalize_content(content: Any) -> list[dict[str, str]]:
    """Normalize content into Qwen3.5/3.8 multimodal-compatible message parts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        raise ValueError(f"Unsupported message content type: {type(content).__name__}")

    normalized: list[dict[str, str]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Each content part must be an object")
        part_type = part.get("type")
        if part_type != "text":
            raise ValueError(
                "AmitAI v0 is text-only. Dataset content parts must use type='text'."
            )
        text = part.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text content must be a non-empty string")
        normalized.append({"type": "text", "text": text})
    return normalized


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    example_id = _require_matching_string(example, "id", SLUG_PATTERN)
    spec_version = _require_matching_string(example, "spec_version", SPEC_VERSION_PATTERN)
    category = _require_matching_string(example, "category", SLUG_PATTERN)
    primary_rules = _normalize_primary_rules(example.get("primary_rules"))

    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Each example must contain at least 2 messages")

    normalized_messages = []
    for message in messages:
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported role: {role!r}")
        normalized_messages.append(
            {"role": role, "content": _normalize_content(message.get("content"))}
        )

    if normalized_messages[-1]["role"] != "assistant":
        raise ValueError("Each SFT conversation must end with an assistant message")

    return {
        "id": example_id,
        "spec_version": spec_version,
        "category": category,
        "primary_rules": primary_rules,
        "messages": normalized_messages,
    }


def load_sft_dataset(path: str):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=path, split="train")
    return dataset.map(normalize_example, desc="Normalizing AmitAI SFT dataset")
