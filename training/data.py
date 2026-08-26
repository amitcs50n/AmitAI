from __future__ import annotations

from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


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

    return {"messages": normalized_messages}


def load_sft_dataset(path: str):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=path, split="train")
    return dataset.map(normalize_example, desc="Normalizing AmitAI SFT dataset")
