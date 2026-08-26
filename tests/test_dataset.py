import pytest

from training.data import normalize_example


def test_normalize_string_content():
    result = normalize_example(
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }
    )
    assert result["messages"][0]["content"] == [{"type": "text", "text": "hello"}]


def test_rejects_non_assistant_final_message():
    with pytest.raises(ValueError, match="end with an assistant"):
        normalize_example(
            {
                "messages": [
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "hi"},
                ]
            }
        )
