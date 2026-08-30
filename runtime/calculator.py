"""Deterministic decimal calculator with a strict expression grammar."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
from typing import Any

from .tooling import ToolDefinition, ToolFailure

MAX_EXPRESSION_LENGTH = 256
MAX_TOKENS = 128
MAX_NESTING_DEPTH = 16
MAX_LITERAL_DIGITS = 64
MAX_EXPONENT_MAGNITUDE = 100
MAX_VALUE_MAGNITUDE = Decimal("1e100")


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


def _tokenize(expression: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    while index < len(expression):
        character = expression[index]
        if character.isspace():
            index += 1
            continue
        if expression.startswith("**", index):
            tokens.append(_Token("POWER", "**"))
            index += 2
            continue
        if character in "+-*/()%":
            tokens.append(
                _Token(
                    {
                        "+": "PLUS",
                        "-": "MINUS",
                        "*": "MULTIPLY",
                        "/": "DIVIDE",
                        "(": "LPAREN",
                        ")": "RPAREN",
                        "%": "PERCENT",
                    }[character],
                    character,
                )
            )
            index += 1
            continue
        if character.isdigit() or character == ".":
            start = index
            seen_decimal = False
            seen_digit = False
            while index < len(expression):
                current = expression[index]
                if current.isdigit():
                    seen_digit = True
                    index += 1
                    continue
                if current == "." and not seen_decimal:
                    seen_decimal = True
                    index += 1
                    continue
                break
            value = expression[start:index]
            if not seen_digit:
                raise ToolFailure("unsupported_expression", "Invalid numeric literal")
            if sum(item.isdigit() for item in value) > MAX_LITERAL_DIGITS:
                raise ToolFailure("expression_limit", "Numeric literal is too large")
            tokens.append(_Token("NUMBER", value))
            continue
        if character.isalpha() or character == "_":
            start = index
            while index < len(expression) and (
                expression[index].isalnum() or expression[index] == "_"
            ):
                index += 1
            identifier = expression[start:index]
            if identifier.lower() != "of":
                raise ToolFailure("unsupported_expression", "Identifiers are not supported")
            tokens.append(_Token("OF", identifier))
            continue
        raise ToolFailure("unsupported_expression", "Expression contains unsupported syntax")

    if len(tokens) > MAX_TOKENS:
        raise ToolFailure("expression_limit", "Expression contains too many tokens")
    tokens.append(_Token("EOF", ""))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.index = 0
        self.nesting = 0

    @property
    def current(self) -> _Token:
        return self.tokens[self.index]

    def advance(self, kind: str) -> _Token:
        token = self.current
        if token.kind != kind:
            raise ToolFailure("unsupported_expression", "Expression grammar is invalid")
        self.index += 1
        return token

    @staticmethod
    def bounded(value: Decimal) -> Decimal:
        if not value.is_finite() or abs(value) > MAX_VALUE_MAGNITUDE:
            raise ToolFailure("result_limit", "Calculator magnitude limit exceeded")
        return value

    def parse(self) -> Decimal:
        value = self.additive()
        if self.current.kind != "EOF":
            raise ToolFailure("unsupported_expression", "Expression grammar is invalid")
        return self.bounded(value)

    def additive(self) -> Decimal:
        value = self.multiplicative()
        while self.current.kind in {"PLUS", "MINUS"}:
            operator = self.current.kind
            self.index += 1
            right = self.multiplicative()
            value = value + right if operator == "PLUS" else value - right
            value = self.bounded(value)
        return value

    def multiplicative(self) -> Decimal:
        value = self.unary()
        while self.current.kind in {"MULTIPLY", "DIVIDE", "OF"}:
            operator = self.current.kind
            self.index += 1
            right = self.unary()
            if operator == "DIVIDE":
                if right == 0:
                    raise ToolFailure("division_by_zero", "Division by zero")
                value /= right
            else:
                value *= right
            value = self.bounded(value)
        return value

    def unary(self) -> Decimal:
        if self.current.kind == "PLUS":
            self.index += 1
            return self.unary()
        if self.current.kind == "MINUS":
            self.index += 1
            return self.bounded(-self.unary())
        return self.postfix()

    def postfix(self) -> Decimal:
        value = self.power()
        while self.current.kind == "PERCENT":
            self.index += 1
            value = self.bounded(value / Decimal(100))
        return value

    def power(self) -> Decimal:
        value = self.primary()
        if self.current.kind != "POWER":
            return value
        self.index += 1
        exponent = self.unary()
        integral = exponent.to_integral_value()
        if exponent != integral:
            raise ToolFailure("invalid_exponent", "Exponent must be an integer")
        exponent_value = int(integral)
        if abs(exponent_value) > MAX_EXPONENT_MAGNITUDE:
            raise ToolFailure("invalid_exponent", "Exponent magnitude limit exceeded")
        if value == 0 and exponent_value < 0:
            raise ToolFailure("division_by_zero", "Division by zero")
        try:
            return self.bounded(value**exponent_value)
        except DecimalException as exc:
            raise ToolFailure("result_limit", "Calculator magnitude limit exceeded") from exc

    def primary(self) -> Decimal:
        if self.current.kind == "NUMBER":
            token = self.advance("NUMBER")
            try:
                return self.bounded(Decimal(token.value))
            except DecimalException as exc:
                raise ToolFailure("unsupported_expression", "Invalid numeric literal") from exc
        if self.current.kind == "LPAREN":
            self.advance("LPAREN")
            self.nesting += 1
            if self.nesting > MAX_NESTING_DEPTH:
                raise ToolFailure("expression_limit", "Parenthesis nesting limit exceeded")
            try:
                value = self.additive()
                self.advance("RPAREN")
                return value
            finally:
                self.nesting -= 1
        raise ToolFailure("unsupported_expression", "Expression grammar is invalid")


class _GrammarValidator:
    """Validate the complete expression grammar without performing arithmetic."""

    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.index = 0
        self.nesting = 0

    @property
    def current(self) -> _Token:
        return self.tokens[self.index]

    def advance(self, kind: str) -> None:
        if self.current.kind != kind:
            raise ToolFailure("unsupported_expression", "Expression grammar is invalid")
        self.index += 1

    def parse(self) -> None:
        self.additive()
        if self.current.kind != "EOF":
            raise ToolFailure("unsupported_expression", "Expression grammar is invalid")

    def additive(self) -> None:
        self.multiplicative()
        while self.current.kind in {"PLUS", "MINUS"}:
            self.index += 1
            self.multiplicative()

    def multiplicative(self) -> None:
        self.unary()
        while self.current.kind in {"MULTIPLY", "DIVIDE", "OF"}:
            self.index += 1
            self.unary()

    def unary(self) -> None:
        if self.current.kind in {"PLUS", "MINUS"}:
            self.index += 1
            self.unary()
            return
        self.postfix()

    def postfix(self) -> None:
        self.power()
        while self.current.kind == "PERCENT":
            self.index += 1

    def power(self) -> None:
        self.primary()
        if self.current.kind == "POWER":
            self.index += 1
            self.unary()

    def primary(self) -> None:
        if self.current.kind == "NUMBER":
            self.advance("NUMBER")
            return
        if self.current.kind == "LPAREN":
            self.advance("LPAREN")
            self.nesting += 1
            if self.nesting > MAX_NESTING_DEPTH:
                raise ToolFailure("expression_limit", "Parenthesis nesting limit exceeded")
            try:
                self.additive()
                self.advance("RPAREN")
                return
            finally:
                self.nesting -= 1
        raise ToolFailure("unsupported_expression", "Expression grammar is invalid")


def _format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def evaluate_expression(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolFailure("invalid_arguments", "Calculator expression must be non-empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolFailure("expression_limit", "Calculator expression is too long")
    tokens = _tokenize(expression)
    _GrammarValidator(tokens).parse()
    with localcontext() as context:
        context.prec = 80
        value = _Parser(tokens).parse()
    return _format_decimal(value)


class CalculatorTool:
    definition = ToolDefinition(
        name="calculator",
        description=(
            "Evaluate deterministic decimal arithmetic using +, -, *, /, **, parentheses, "
            "postfix percentages, and 'of'."
        ),
        arguments={"expression": "string"},
    )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if set(arguments) != {"expression"}:
            raise ToolFailure(
                "invalid_arguments",
                "Calculator requires exactly one expression argument",
            )
        expression = arguments["expression"]
        if not isinstance(expression, str) or not expression.strip():
            raise ToolFailure("invalid_arguments", "Calculator expression must be non-empty")
        if len(expression) > MAX_EXPRESSION_LENGTH:
            raise ToolFailure("expression_limit", "Calculator expression is too long")
        expression = expression.strip()
        try:
            evaluate_expression(expression)
        except ToolFailure as exc:
            if exc.code != "division_by_zero":
                raise
        return {"expression": expression}

    def execute(self, arguments: Mapping[str, Any]) -> str:
        return evaluate_expression(str(arguments["expression"]))
