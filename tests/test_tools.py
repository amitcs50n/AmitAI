import json

import pytest

from runtime.calculator import (
    MAX_EXPONENT_MAGNITUDE,
    MAX_EXPRESSION_LENGTH,
    MAX_LITERAL_DIGITS,
    MAX_NESTING_DEPTH,
    MAX_TOKENS,
    CalculatorTool,
    evaluate_expression,
)
from runtime.tooling import (
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
    LateToolProtocolFilter,
    ToolCall,
    ToolFailure,
    ToolRegistry,
    classify_tool_protocol_prefix,
    failed_tool_attempt,
    format_tool_result,
    is_reserved_tool_candidate,
    parse_tool_call,
    sanitize_late_tool_protocol,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("10 / 4", "2.5"),
        ("2.5 * 3", "7.5"),
        ("2 ** 8", "256"),
        ("2 ** 3 ** 2", "512"),
        ("15%", "0.15"),
        ("15% of 200", "30"),
    ],
)
def test_calculator_supported_arithmetic(expression: str, expected: str) -> None:
    assert evaluate_expression(expression) == expected


def test_calculator_of_matches_multiplication_division_precedence_left_to_right() -> None:
    assert evaluate_expression("100 / 50% of 10") == "2000"


def test_calculator_rejects_division_by_zero() -> None:
    with pytest.raises(ToolFailure) as raised:
        evaluate_expression("10 / (3 - 3)")

    assert raised.value.code == "division_by_zero"


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("whoami")',
        'open("secret.txt")',
        "a.b",
        "lambda value: value",
        "[item for item in values]",
        "{item: item for item in values}",
        "answer = 2 + 2",
        "sum([1, 2])",
        "2 // 1",
        "2 << 1",
    ],
)
def test_calculator_rejects_unsafe_or_unsupported_syntax(expression: str) -> None:
    with pytest.raises(ToolFailure) as raised:
        evaluate_expression(expression)

    assert raised.value.code == "unsupported_expression"


def test_calculator_restricts_exponents_to_bounded_integers() -> None:
    with pytest.raises(ToolFailure) as fractional:
        evaluate_expression("9 ** 0.5")
    with pytest.raises(ToolFailure) as oversized:
        evaluate_expression(f"2 ** {MAX_EXPONENT_MAGNITUDE + 1}")

    assert fractional.value.code == "invalid_exponent"
    assert oversized.value.code == "invalid_exponent"


def test_calculator_enforces_expression_nesting_and_result_bounds() -> None:
    with pytest.raises(ToolFailure) as length_failure:
        evaluate_expression("1" * (MAX_EXPRESSION_LENGTH + 1))
    with pytest.raises(ToolFailure) as nesting_failure:
        evaluate_expression(
            "(" * (MAX_NESTING_DEPTH + 1)
            + "1"
            + ")" * (MAX_NESTING_DEPTH + 1)
        )
    with pytest.raises(ToolFailure) as magnitude_failure:
        evaluate_expression("100 ** 100")
    with pytest.raises(ToolFailure) as literal_failure:
        evaluate_expression("9" * (MAX_LITERAL_DIGITS + 1))
    token_heavy_expression = "+".join("1" for _ in range((MAX_TOKENS // 2) + 1))
    with pytest.raises(ToolFailure) as token_failure:
        evaluate_expression(token_heavy_expression)

    assert length_failure.value.code == "expression_limit"
    assert nesting_failure.value.code == "expression_limit"
    assert magnitude_failure.value.code == "result_limit"
    assert literal_failure.value.code == "expression_limit"
    assert token_failure.value.code == "expression_limit"


def test_tool_call_protocol_accepts_only_one_whole_json_envelope() -> None:
    call = parse_tool_call(
        '  \n<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}'
        "</tool_call>\t"
    )

    assert call == ToolCall(name="calculator", arguments={"expression": "2 + 2"})

    for malformed in (
        'Use this: <tool_call>{"name":"calculator","arguments":{}}</tool_call>',
        '<tool_call>{"name":"calculator","arguments":{}}</tool_call> done',
        "<tool_call>{not-json}</tool_call>",
        '<tool_call>{"name":"calculator"}</tool_call>',
        '<tool_call>{"name":"calculator","arguments":{},"extra":true}</tool_call>',
    ):
        with pytest.raises(ToolFailure, match="Tool call"):
            parse_tool_call(malformed)


def test_reserved_protocol_detection_uses_prefix_commit_semantics() -> None:
    assert is_reserved_tool_candidate("  <tool_call broken") is True
    assert is_reserved_tool_candidate("\n<tool_result>{}") is True
    assert is_reserved_tool_candidate("Sure: <tool_call broken") is False
    assert is_reserved_tool_candidate("Result: <tool_result>{}") is False
    assert is_reserved_tool_candidate("The result is 4") is False


@pytest.mark.parametrize("prefix", ["", " \n", "<", "<too", "<tool_"])
def test_tool_protocol_prefix_remains_ambiguous_only_while_necessary(
    prefix: str,
) -> None:
    assert classify_tool_protocol_prefix(prefix) == "ambiguous"


def test_late_protocol_filter_flushes_divergence_and_suppresses_envelopes() -> None:
    protocol_filter = LateToolProtocolFilter()

    visible = protocol_filter.feed("I'll calculate that. <too")
    visible += protocol_filter.feed(
        'l_call>{"name":"calculator","arguments":{"expression":"2+2"}}'
        "</tool_call> Done."
    )
    visible += protocol_filter.finish()

    assert visible == "I'll calculate that.  Done."


def test_late_protocol_filter_flushes_an_ambiguous_prefix_that_diverges() -> None:
    protocol_filter = LateToolProtocolFilter()

    assert protocol_filter.feed("<to") == ""
    assert protocol_filter.feed("ast") == "<toast"
    assert protocol_filter.finish() == ""


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (
            'Before <tool_call>{"unsafe":"raw"}</tool_call> after',
            "Before  after",
        ),
        (
            'Before <tool_result>{"unsafe":"raw"}</tool_result> after',
            "Before  after",
        ),
        ("Before <tool_call", "Before "),
        ("Ordinary <toast> text", "Ordinary <toast> text"),
    ],
)
def test_complete_candidate_sanitizer_matches_stream_filter(
    candidate: str,
    expected: str,
) -> None:
    assert sanitize_late_tool_protocol(candidate) == expected


def test_tool_registry_reports_unknown_invalid_and_execution_failures_safely() -> None:
    registry = ToolRegistry([CalculatorTool()])

    unknown = registry.execute(ToolCall("missing", {"unsafe": "raw"}), attempt=1)
    invalid = registry.execute(ToolCall("calculator", {"wrong": "raw"}), attempt=2)
    division = registry.execute(
        ToolCall("calculator", {"expression": "1 / 0"}),
        attempt=3,
    )

    assert unknown.as_record() == {
        "attempt": 1,
        "name": "missing",
        "success": False,
        "error": {
            "code": "unknown_tool",
            "message": "Requested tool is not available",
        },
    }
    assert "arguments" not in invalid.as_record()
    assert invalid.error_code == "invalid_arguments"
    assert division.arguments == {"expression": "1 / 0"}
    assert division.error_code == "division_by_zero"


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("whoami")',
        "sqrt(4)",
        "unknown_identifier + 1",
    ],
)
def test_unsafe_calculator_arguments_are_not_retained_in_metadata(
    expression: str,
) -> None:
    attempt = ToolRegistry([CalculatorTool()]).execute(
        ToolCall("calculator", {"expression": expression}),
        attempt=1,
    )

    record = attempt.as_record()
    assert record["success"] is False
    assert "arguments" not in record
    assert expression not in json.dumps(record)


def test_safe_calculator_attempts_retain_validated_arguments_and_results() -> None:
    registry = ToolRegistry([CalculatorTool()])

    division = registry.execute(
        ToolCall("calculator", {"expression": " 1 / 0 "}),
        attempt=1,
    )
    success = registry.execute(
        ToolCall("calculator", {"expression": " 17 * 83 "}),
        attempt=2,
    )

    assert division.arguments == {"expression": "1 / 0"}
    assert division.success is False
    assert division.error_code == "division_by_zero"
    assert success.arguments == {"expression": "17 * 83"}
    assert success.result == "1411"


def test_malformed_tool_json_is_not_retained_in_failure_metadata() -> None:
    raw = '<tool_call>{"name":"calculator","arguments":not-json}</tool_call>'

    with pytest.raises(ToolFailure) as raised:
        parse_tool_call(raw)
    record = failed_tool_attempt(attempt=1, failure=raised.value).as_record()

    assert record["success"] is False
    assert "arguments" not in record
    assert raw not in json.dumps(record)


def test_internal_tool_result_is_deterministic_structured_json() -> None:
    attempt = ToolRegistry([CalculatorTool()]).execute(
        ToolCall("calculator", {"expression": "17 * 83"}),
        attempt=1,
    )

    message = format_tool_result(attempt)
    assert message.startswith(TOOL_RESULT_OPEN)
    assert message.endswith(TOOL_RESULT_CLOSE)
    payload = json.loads(message[len(TOOL_RESULT_OPEN) : -len(TOOL_RESULT_CLOSE)])
    assert payload == {
        "arguments": {"expression": "17 * 83"},
        "attempt": 1,
        "name": "calculator",
        "result": "1411",
        "success": True,
    }
