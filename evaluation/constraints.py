from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


SUPPORTED_CONSTRAINT_TYPES = (
    "exact_words",
    "exact_bullets",
    "at_most_bullets",
    "code_only",
)

_COUNT_PATTERNS = (
    (
        "exact_words",
        re.compile(r"\bexactly[ \t]+(?P<count>[0-9]+)[ \t]+words?\b", re.IGNORECASE),
    ),
    (
        "exact_bullets",
        re.compile(r"\bexactly[ \t]+(?P<count>[0-9]+)[ \t]+bullets?\b", re.IGNORECASE),
    ),
    (
        "at_most_bullets",
        re.compile(r"\bat[ \t]+most[ \t]+(?P<count>[0-9]+)[ \t]+bullets?\b", re.IGNORECASE),
    ),
)
_CODE_ONLY_PATTERN = re.compile(
    r"\breturn[ \t]+code[ \t]+only\b|(?:^|(?<=[.!?]))[ \t]*code[ \t]+only\b",
    re.IGNORECASE | re.MULTILINE,
)
_NEGATED_DIRECTIVE_PATTERN = re.compile(
    r"\b(?:do[ \t]+not|don't|never|not|without)"
    r"(?:[ \t]+(?:use|using|write(?:[ \t]+in)?|return|answer(?:[ \t]+in)?|"
    r"respond(?:[ \t]+in)?|output|provide))?[ \t]*$",
    re.IGNORECASE,
)
_METALINGUISTIC_PATTERN = re.compile(
    r"\b(?:phrase|wording|literal|string|text)[ \t]*$",
    re.IGNORECASE,
)
_BULLET_PATTERN = re.compile(r"(?:[-*+]|[0-9]+[.)])[ \t]+\S")
_FENCE_MARKER_PATTERN = re.compile(r"(?m)^[ \t]*(?:`{3,}|~{3,})")
_FENCED_CODE_PATTERN = re.compile(
    r"(?P<fence>`{3,}|~{3,})[^\r\n]*\r?\n(?P<code>.*?)\r?\n(?P=fence)",
    re.DOTALL,
)


def _is_negated(prompt: str, start: int) -> bool:
    prefix = prompt[max(0, start - 32) : start]
    return _NEGATED_DIRECTIVE_PATTERN.search(prefix) is not None


def _is_quoted(prompt: str, start: int, end: int) -> bool:
    left = prompt[:start].rstrip()
    right = prompt[end:].lstrip()
    if not left or left[-1] not in {'"', "'", "`"}:
        return False
    return right.startswith(left[-1])


def _is_metalinguistic(prompt: str, start: int) -> bool:
    prefix = prompt[max(0, start - 32) : start]
    return _METALINGUISTIC_PATTERN.search(prefix) is not None


def parse_constraints(prompt: str) -> list[dict[str, Any]]:
    """Parse only the explicitly supported mechanical constraint shapes."""

    matches: list[tuple[int, dict[str, Any]]] = []
    for constraint_type, pattern in _COUNT_PATTERNS:
        for match in pattern.finditer(prompt):
            if (
                _is_negated(prompt, match.start())
                or _is_quoted(prompt, match.start(), match.end())
                or _is_metalinguistic(prompt, match.start())
            ):
                continue
            matches.append(
                (
                    match.start(),
                    {"type": constraint_type, "count": int(match.group("count"))},
                )
            )

    for match in _CODE_ONLY_PATTERN.finditer(prompt):
        if not (
            _is_negated(prompt, match.start())
            or _is_quoted(prompt, match.start(), match.end())
            or _is_metalinguistic(prompt, match.start())
        ):
            matches.append((match.start(), {"type": "code_only"}))

    constraints: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for _, constraint in sorted(matches, key=lambda item: item[0]):
        key = (constraint["type"], constraint.get("count"))
        if key not in seen:
            seen.add(key)
            constraints.append(constraint)
    return constraints


def count_words(text: str) -> int:
    return len(text.strip().split())


def count_bullets(text: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if _BULLET_PATTERN.match(line.lstrip()) is not None
    )


def normalize_code_only(text: str) -> str | None:
    """Return code inside one complete outer fence, otherwise return None."""

    match = _FENCED_CODE_PATTERN.fullmatch(text.strip())
    if match is None:
        return None
    code = match.group("code")
    if not code.strip():
        return None
    fence_marker = re.escape(match.group("fence")[0])
    if re.search(rf"(?m)^[ \t]*{fence_marker}{{3,}}[^\r\n]*$", code):
        return None
    return code


def _validate_code_only(text: str, constraint: dict[str, Any]) -> dict[str, Any]:
    normalized_code = normalize_code_only(text)
    if normalized_code is not None:
        return {
            "constraint": constraint,
            "passed": True,
            "actual": "single_fenced_code_block",
            "message": "The answer contained exactly one fenced code block.",
        }
    if _FENCE_MARKER_PATTERN.search(text) is None:
        return {
            "constraint": constraint,
            "passed": True,
            "actual": "unfenced_unverified",
            "message": (
                "The unfenced answer was accepted because code-only content could not be "
                "disproved mechanically."
            ),
        }
    return {
        "constraint": constraint,
        "passed": False,
        "actual": "prose_or_invalid_fence",
        "message": (
            "Expected code only, but the answer was not exactly one fenced code block "
            "with no surrounding prose."
        ),
    }


def validate_response(
    text: str,
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    has_code_only = any(constraint["type"] == "code_only" for constraint in constraints)
    normalized_code = normalize_code_only(text) if has_code_only else None
    validation_text = normalized_code if normalized_code is not None else text
    checks: list[dict[str, Any]] = []
    for constraint in constraints:
        constraint_type = constraint["type"]
        expected = constraint.get("count")
        if constraint_type == "exact_words":
            actual = count_words(validation_text)
            passed = actual == expected
            check = {
                "constraint": constraint,
                "passed": passed,
                "actual": actual,
                "message": (
                    f"Expected exactly {expected} words, but the answer contained "
                    f"{actual} words."
                ),
            }
        elif constraint_type == "exact_bullets":
            actual = count_bullets(validation_text)
            passed = actual == expected
            check = {
                "constraint": constraint,
                "passed": passed,
                "actual": actual,
                "message": (
                    f"Expected exactly {expected} bullets, but the answer contained "
                    f"{actual} bullets."
                ),
            }
        elif constraint_type == "at_most_bullets":
            actual = count_bullets(validation_text)
            passed = actual <= expected
            check = {
                "constraint": constraint,
                "passed": passed,
                "actual": actual,
                "message": (
                    f"Expected at most {expected} bullets, but the answer contained "
                    f"{actual} bullets."
                ),
            }
        elif constraint_type == "code_only":
            check = _validate_code_only(text, constraint)
        else:
            raise ValueError(f"Unsupported mechanical constraint: {constraint_type}")
        checks.append(check)

    failures = [check["message"] for check in checks if not check["passed"]]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "normalized_response": normalized_code,
    }


def build_retry_prompt(
    original_prompt: str,
    previous_response: str,
    validation_result: dict[str, Any],
) -> str:
    failures = validation_result.get("failures")
    if not isinstance(failures, list) or not failures:
        raise ValueError("A corrective retry requires at least one validation failure")
    measured_failure = "\n".join(str(failure) for failure in failures)
    return (
        "Original user request:\n"
        f"{original_prompt}\n\n"
        "Previous answer:\n"
        f"{previous_response}\n\n"
        "Validation failure:\n"
        f"{measured_failure}\n\n"
        "Rewrite the answer so it satisfies the original request and the measured constraint.\n"
        "Preserve the original content, tone, and task requirements as much as possible.\n"
        "Output only the corrected answer."
    )


def validate_with_one_retry(
    original_prompt: str,
    original_response: str,
    retry: Callable[[str], str],
) -> dict[str, Any]:
    constraints = parse_constraints(original_prompt)
    first_validation = validate_response(original_response, constraints)
    first_final_response = first_validation["normalized_response"]
    if first_final_response is None:
        first_final_response = original_response
    result: dict[str, Any] = {
        "original_user_prompt": original_prompt,
        "original_response": original_response,
        "parsed_constraints": constraints,
        "first_validation": first_validation,
        "retry_happened": False,
        "retry_reason": None,
        "retry_prompt": None,
        "retry_response": None,
        "second_validation": None,
        "retry_passed": None,
        "final_response": first_final_response,
    }
    if not constraints or first_validation["passed"]:
        return result

    retry_prompt = build_retry_prompt(
        original_prompt,
        original_response,
        first_validation,
    )
    retry_response = retry(retry_prompt)
    if not isinstance(retry_response, str) or not retry_response.strip():
        raise ValueError("Corrective model retry returned an empty response")
    retry_response = retry_response.strip()
    second_validation = validate_response(retry_response, constraints)
    final_response = retry_response
    if second_validation["passed"] and second_validation["normalized_response"] is not None:
        final_response = second_validation["normalized_response"]
    result.update(
        retry_happened=True,
        retry_reason="\n".join(first_validation["failures"]),
        retry_prompt=retry_prompt,
        retry_response=retry_response,
        second_validation=second_validation,
        retry_passed=second_validation["passed"],
        final_response=final_response,
    )
    return result
