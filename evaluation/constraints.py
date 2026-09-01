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
MAX_MECHANICAL_RETRIES = 2

_WRITTEN_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_WRITTEN_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WRITTEN_UNIT_PATTERN = "|".join(
    word for word, value in _WRITTEN_SMALL_NUMBERS.items() if 1 <= value <= 9
)
_WRITTEN_SMALL_PATTERN = "|".join(_WRITTEN_SMALL_NUMBERS)
_WRITTEN_TENS_PATTERN = "|".join(_WRITTEN_TENS)
_WRITTEN_COUNT_PATTERN = (
    rf"(?:one[ \t]+hundred|"
    rf"(?:{_WRITTEN_TENS_PATTERN})(?:(?:[ \t]+|-)(?:{_WRITTEN_UNIT_PATTERN}))?|"
    rf"(?:{_WRITTEN_SMALL_PATTERN}))"
)
_COUNT_TOKEN_PATTERN = rf"(?:[0-9]+|{_WRITTEN_COUNT_PATTERN})"

_COUNT_PATTERNS = (
    (
        "exact_words",
        re.compile(
            rf"\bexactly[ \t]+(?P<count>{_COUNT_TOKEN_PATTERN})[ \t]+words?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exact_bullets",
        re.compile(
            rf"\bexactly[ \t]+(?P<count>{_COUNT_TOKEN_PATTERN})[ \t]+bullets?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "at_most_bullets",
        re.compile(
            rf"\bat[ \t]+most[ \t]+(?P<count>{_COUNT_TOKEN_PATTERN})[ \t]+bullets?\b",
            re.IGNORECASE,
        ),
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
_QUOTE_DELIMITERS = ('"', "'", "`")


def _is_negated(prompt: str, start: int) -> bool:
    prefix = prompt[max(0, start - 32) : start]
    return _NEGATED_DIRECTIVE_PATTERN.search(prefix) is not None


def _is_word_apostrophe(text: str, index: int) -> bool:
    return (
        text[index] == "'"
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isalnum()
        and text[index + 1].isalnum()
    )


def _paired_quote_spans(prompt: str) -> list[tuple[int, int]] | None:
    spans: list[tuple[int, int]] = []
    for delimiter in _QUOTE_DELIMITERS:
        positions = [
            index
            for index, character in enumerate(prompt)
            if character == delimiter and not _is_word_apostrophe(prompt, index)
        ]
        if len(positions) % 2:
            return None
        spans.extend(zip(positions[::2], positions[1::2]))

    spans.sort()
    for index, (_, closing) in enumerate(spans):
        for later_opening, later_closing in spans[index + 1 :]:
            if later_opening < closing < later_closing:
                return None
    return spans


def _is_quoted(prompt: str, start: int, end: int) -> bool:
    spans = _paired_quote_spans(prompt)
    if spans is None:
        return True
    return any(opening < start and end <= closing for opening, closing in spans)


def _is_metalinguistic(prompt: str, start: int) -> bool:
    prefix = prompt[max(0, start - 32) : start]
    return _METALINGUISTIC_PATTERN.search(prefix) is not None


def _parse_count(value: str) -> int:
    if value.isdigit():
        return int(value)

    normalized = re.sub(r"[ \t]+", " ", value.lower().replace("-", " ")).strip()
    if normalized == "one hundred":
        return 100
    if normalized in _WRITTEN_SMALL_NUMBERS:
        return _WRITTEN_SMALL_NUMBERS[normalized]
    if normalized in _WRITTEN_TENS:
        return _WRITTEN_TENS[normalized]

    tens, unit = normalized.split(" ")
    return _WRITTEN_TENS[tens] + _WRITTEN_SMALL_NUMBERS[unit]


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
                    {
                        "type": constraint_type,
                        "count": _parse_count(match.group("count")),
                    },
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


def _build_count_retry_guidance(check: Any) -> str | None:
    if not isinstance(check, dict) or check.get("passed") is not False:
        return None
    constraint = check.get("constraint")
    actual = check.get("actual")
    if not isinstance(constraint, dict) or not isinstance(actual, int):
        return None
    expected = constraint.get("count")
    if not isinstance(expected, int):
        return None

    constraint_type = constraint.get("type")
    if constraint_type == "exact_words" and actual != expected:
        difference = abs(expected - actual)
        difference_unit = "word" if difference == 1 else "words"
        if actual < expected:
            direction = f"The answer is {difference} {difference_unit} short."
            edit = (
                f"Edit the previous answer minimally and add exactly {difference} "
                f"{difference_unit}."
            )
        else:
            direction = f"The answer is {difference} {difference_unit} too long."
            edit = (
                f"Edit the previous answer minimally and remove exactly {difference} "
                f"{difference_unit}."
            )
        return "\n".join(
            [
                f"The previous answer contains {actual} whitespace-separated words.",
                f"The required total is exactly {expected} words.",
                direction,
                "Count words exactly as whitespace-separated tokens.",
                edit,
                "Do not rewrite it from scratch unless unavoidable.",
                (
                    "Before returning, internally recount using whitespace-separated "
                    f"tokens and ensure the final total is exactly {expected} words."
                ),
            ]
        )

    if constraint_type == "exact_bullets" and actual != expected:
        difference = abs(expected - actual)
        difference_unit = "bullet" if difference == 1 else "bullets"
        if actual < expected:
            direction = f"The answer is {difference} {difference_unit} short."
            edit = (
                f"Edit the previous answer minimally and add exactly {difference} "
                f"{difference_unit}."
            )
            preservation = "Preserve the original task and content."
        else:
            direction = f"The answer has {difference} excess {difference_unit}."
            edit = (
                f"Edit the previous answer minimally and remove exactly {difference} "
                f"{difference_unit}."
            )
            preservation = "Preserve the strongest relevant content."
        return "\n".join(
            [
                f"The previous answer contains {actual} Markdown list-item bullets.",
                f"The required total is exactly {expected} bullets.",
                direction,
                edit,
                preservation,
                (
                    "Do not invent unnecessary services or details merely to fill the "
                    "bullet count."
                ),
                (
                    "Before returning, internally recount the Markdown list-item bullets "
                    f"and ensure the final total is exactly {expected}."
                ),
            ]
        )

    if constraint_type == "at_most_bullets" and actual > expected:
        difference = actual - expected
        difference_unit = "bullet" if difference == 1 else "bullets"
        return "\n".join(
            [
                f"The previous answer contains {actual} Markdown list-item bullets.",
                f"The maximum allowed total is {expected} bullets.",
                f"The answer has {difference} excess {difference_unit}.",
                (
                    f"Edit the previous answer minimally and remove exactly {difference} "
                    f"{difference_unit} so the final count is no more than {expected}."
                ),
                "Preserve the most important content.",
                "Do not invent unnecessary services or details.",
                (
                    "Before returning, internally recount the Markdown list-item bullets "
                    f"and ensure the final total is no more than {expected}."
                ),
            ]
        )

    return None


def build_retry_prompt(
    original_prompt: str,
    previous_response: str,
    validation_result: dict[str, Any],
) -> str:
    failures = validation_result.get("failures")
    if not isinstance(failures, list) or not failures:
        raise ValueError("A corrective retry requires at least one validation failure")
    measured_failure = "\n".join(str(failure) for failure in failures)
    checks = validation_result.get("checks")
    count_guidance = (
        [
            guidance
            for check in checks
            if (guidance := _build_count_retry_guidance(check)) is not None
        ]
        if isinstance(checks, list)
        else []
    )
    directional_guidance = (
        "\n\nCount-aware correction:\n" + "\n\n".join(count_guidance) + "\n"
        if count_guidance
        else "\n"
    )
    return (
        "Original user request:\n"
        f"{original_prompt}\n\n"
        "Previous answer:\n"
        f"{previous_response}\n\n"
        "Validation failure:\n"
        f"{measured_failure}\n\n"
        "Rewrite the answer so it satisfies the original request and the measured constraint."
        f"{directional_guidance}"
        "Preserve the original content, tone, and task requirements as much as possible.\n"
        "Output only the corrected answer."
    )


def validate_with_bounded_retries(
    original_prompt: str,
    original_response: str,
    retry: Callable[[str], str],
    max_retries: int = MAX_MECHANICAL_RETRIES,
    *,
    retry_original_prompt: str | None = None,
) -> dict[str, Any]:
    # Constraints always come from the original request. Runtime callers may supply
    # a provider-safe projection solely for the corrective model-facing prompt.
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or not 0 <= max_retries <= MAX_MECHANICAL_RETRIES
    ):
        raise ValueError(
            f"max_retries must be an integer between 0 and {MAX_MECHANICAL_RETRIES}"
        )

    constraints = parse_constraints(original_prompt)
    first_validation = validate_response(original_response, constraints)
    retry_attempts: list[dict[str, Any]] = []
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
        "retry_attempts": retry_attempts,
        "retry_count": 0,
        "final_validation": first_validation,
        "final_response": original_response,
    }
    if not constraints or first_validation["passed"]:
        return result

    latest_response = original_response
    latest_validation = first_validation
    for attempt_number in range(1, max_retries + 1):
        retry_reason = "\n".join(latest_validation["failures"])
        retry_prompt = build_retry_prompt(
            original_prompt if retry_original_prompt is None else retry_original_prompt,
            latest_response,
            latest_validation,
        )
        retry_response = retry(retry_prompt)
        if not isinstance(retry_response, str) or not retry_response.strip():
            raise ValueError("Corrective model retry returned an empty response")
        retry_response = retry_response.strip()
        retry_validation = validate_response(retry_response, constraints)
        retry_attempts.append(
            {
                "attempt": attempt_number,
                "reason": retry_reason,
                "prompt": retry_prompt,
                "response": retry_response,
                "validation": retry_validation,
                "passed": retry_validation["passed"],
            }
        )
        result.update(
            retry_happened=True,
            retry_count=attempt_number,
            final_response=retry_response,
            final_validation=retry_validation,
        )
        if attempt_number == 1:
            result.update(
                retry_reason=retry_reason,
                retry_prompt=retry_prompt,
                retry_response=retry_response,
                second_validation=retry_validation,
                retry_passed=retry_validation["passed"],
            )
        if retry_validation["passed"]:
            break
        latest_response = retry_response
        latest_validation = retry_validation

    return result


def validate_with_one_retry(
    original_prompt: str,
    original_response: str,
    retry: Callable[[str], str],
) -> dict[str, Any]:
    """Compatibility wrapper for the bounded two-retry production policy."""

    return validate_with_bounded_retries(
        original_prompt,
        original_response,
        retry,
        max_retries=MAX_MECHANICAL_RETRIES,
    )
