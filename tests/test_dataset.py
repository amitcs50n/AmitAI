import pytest

from training.data import normalize_example


def _example(messages):
    return {
        "id": "conversation_001",
        "spec_version": "1.0.0",
        "category": "normal_conversation",
        "primary_rules": ["RESPONSE-001"],
        "messages": messages,
    }


def test_normalize_preserves_metadata_and_string_content():
    result = normalize_example(
        _example(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
    )
    assert result["id"] == "conversation_001"
    assert result["spec_version"] == "1.0.0"
    assert result["category"] == "normal_conversation"
    assert result["primary_rules"] == ["RESPONSE-001"]
    assert result["messages"][0]["content"] == [{"type": "text", "text": "hello"}]


def test_rejects_non_assistant_final_message():
    with pytest.raises(ValueError, match="end with an assistant"):
        normalize_example(
            _example(
                [
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "hi"},
                ]
            )
        )


def test_rejects_missing_metadata():
    with pytest.raises(ValueError, match="id has an invalid format"):
        normalize_example(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            }
        )
