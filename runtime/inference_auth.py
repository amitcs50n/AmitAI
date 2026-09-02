"""Shared validation for inference-service credentials, not local application tokens."""

MIN_INFERENCE_TOKEN_CHARS = 32


def validate_inference_token(token: str) -> str:
    """Require meaningful header-safe material without changing the credential."""

    if (
        not isinstance(token, str)
        or len(token) < MIN_INFERENCE_TOKEN_CHARS
        or any(not 33 <= ord(char) <= 126 for char in token)
    ):
        raise ValueError(
            "Inference token must contain at least 32 printable ASCII characters without whitespace"
        )
    return token
