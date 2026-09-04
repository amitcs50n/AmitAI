"""High-precision text preflight for structurally unavailable answers.

Only bounded, projected compiler output enters this module. This is deliberately
an English request grammar, not a semantic classifier or hallucination judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from .context import CompiledModelContext

EpistemicGuardKind = Literal["missing_history", "ambiguous_reference", "unknown_internal_env_var"]


@dataclass(frozen=True)
class EpistemicGuardDecision:
    kind: EpistemicGuardKind
    deterministic_response: str
    reason: str


_UNAVAILABLE = (
    "That part of the conversation isn't available in the current context. "
    "Paste it here if you want me to use it."
)
_RECALL_START = re.compile(
    r"^(?:what\b|which\b|repeat\b|quote\b|tell me\b|remind me\b|restate\b|give me\b)",
)
_OPENING = re.compile(r"\b(?:first|opening|earliest) (?:user )?(?:message|turn|line)\b")
_OPENING_RECALL = re.compile(
    r"^(?:what (?:did i|was (?:my|the|our))\b|(?:repeat|quote)\b|"
    r"tell me (?:exactly )?(?:what i|my omitted|my very|my first)\b)|"
    r"\b(?:of this conversation|in this chat|i sent|i wrote)\b",
)
_BEFORE_VISIBLE = re.compile(
    r"\bbefore (?:all )?(?:the |this )?(?:"
    r"(?:messages|exchanges|turns) you can (?:currently )?see|"
    r"oldest (?:one|message|turn)|retained (?:history|exchange)|"
    r"messages (?:currently )?available)\b",
)
_PREVIOUS = re.compile(
    r"\b(?:(?:immediately |just )?before (?:this|my current)(?:\b)|"
    r"(?:previous|preceding|last) (?:user |assistant )?(?:message|turn|task|request|exchange)\b)",
)
_TURN_CONTENT = re.compile(
    r"\b(?:message|turn|task|request|exchange)\b|\b(?:did i|i) (?:ask(?:ed)?|say|said|write|wrote|sent|gave)\b",
)
_PAST_SPEAKER = re.compile(
    r"\b(?:did (?:i|you) (?:ask|say|write)|(?:i|you) (?:asked|said|wrote|sent|gave))\b",
)

# Full matches deliberately reject clauses supplying quotes, code, URLs, objects,
# examples or hypothetical requests. Do not broaden the tail to arbitrary prose.
_TARGET = r"(?:this|that|it|these|those|them)"
_DATE = (
    r"(?:today|tomorrow|(?:next |this )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))"
)
_ZERO_REFERENT = re.compile(
    rf"(?:"
    rf"(?:(?:could|can|would|will) you |please )?"
    rf"(?:shift|move|reschedule|cancel|rewrite|revise|summarize|explain|translate|delete) {_TARGET}"
    rf"(?: (?:to|for) {_DATE})?|explain {_TARGET} error|"
    rf"should i take {_TARGET}(?: {_DATE})?|"
    rf"what (?:does {_TARGET}(?: error)? mean|is {_TARGET}|are (?:these|those))"
    rf")[?.!]*",
)
_ENV = re.compile(r"\b(?:environment[ -]variable|env[ -]var)(?:s)?\b")
_PROJECT = re.compile(
    r"\b(?:our|my)\s+(?:[a-z0-9-]+\s+){0,3}"
    r"(?:app|application|project|service|server|system|deployment|config)\b|"
    r"\bthis (?:app|application|project|service|deployment)\b",
)
_ENV_QUERY = re.compile(r"\b(?:what(?:'s| is)?|which)\b[^?.!]*\b(?:environment[ -]variable|env[ -]var)")
_ACTUAL_ENV = re.compile(r"\b(?:exact|actual|actually|reads?|uses?|configures?)\b")
_PROPOSED_ENV = re.compile(r"\b(?:should|could|recommend|suggest|example|hypothetical)\b")
_ENV_NAME = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_BARE_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,63}")


def _opening_requested(text: str) -> bool:
    opening = _OPENING.search(text)
    if opening is None or not _OPENING_RECALL.search(text):
        return False
    # Bind the requested message to this conversation, not a queue, speech, or
    # other object's ordering mentioned in a question about an earlier topic.
    owned = re.search(r"\b(?:my|our)(?: (?:very|omitted|exact))* $", text[:opening.start()])
    conversational = re.search(r"\b(?:this (?:conversation|chat)|i sent|i wrote)\b", text)
    bare_recall = re.fullmatch(
        r"(?:repeat|quote|what was) (?:the )?(?:very )?(?:first|opening|earliest) "
        r"(?:message|turn|line)[?.!]*", text,
    )
    return bool(owned or conversational or bare_recall)


def _previous_requested(text: str) -> bool:
    if not _PREVIOUS.search(text) or not _TURN_CONTENT.search(text):
        return False
    return bool(_PAST_SPEAKER.search(text) or re.fullmatch(
        r"(?:repeat|quote|what was) (?:the |my |your )?(?:immediately )?"
        r"(?:previous|preceding|last) (?:user |assistant )?"
        r"(?:message|turn|task|request|exchange)[?.!]*", text,
    ))


def _has_env_candidate(text: str) -> bool:
    # Accept supplied identifiers, not their truth. The provider still reasons
    # about conflicting/negated candidates. Bare names need a config association.
    return bool(_ENV_NAME.search(text) or re.search(
        r"(?:\b(?:variable|env[ -]var)\s+(?:is\s+|named\s+)?|"
        r"\bconfig(?:uration)?\s+(?:says|reads|specifies|uses)\s+)"
        r"[`\"']?[A-Z][A-Z0-9]{1,63}\b", text,
    ) or re.search(
        r"\b(?:environ|getenv)\s*[\[(]\s*[\"'][A-Za-z_][A-Za-z0-9_]*[\"']", text,
    ))


def _trusted_memory_has_candidate(context: CompiledModelContext) -> bool:
    for frame in context.messages[1:1 + context.trusted_context_count]:
        content = frame["content"]
        if not content.startswith("MEMORY_CONTEXT_V1\n"):
            continue
        # Parse values only: frame labels and keys are not project identifiers.
        start, end = content.find("<memory_context>"), content.rfind("</memory_context>")
        if start < 0 or end < start:
            continue
        try:
            payload = json.loads(content[start + len("<memory_context>"):end])
        except (ValueError, TypeError):
            continue
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            continue
        for item in items:
            value = item.get("value") if isinstance(item, dict) else None
            key = item.get("key", "") if isinstance(item, dict) else ""
            if isinstance(value, str) and (
                _has_env_candidate(value) or (
                    isinstance(key, str) and re.search(r"\benv(?:[_. -]|$)|environment", key)
                    and _BARE_ENV_NAME.fullmatch(value.strip())
                )
            ):
                return True
    return False


def epistemic_preflight(context: CompiledModelContext) -> EpistemicGuardDecision | None:
    """Decide without access to omitted/raw/private history or any external state."""
    prompt = context.messages[-1]["content"]
    text = " ".join(prompt.casefold().replace("’", "'").split())
    if _RECALL_START.search(text):
        if _opening_requested(text) and (
            context.history_truncated or context.retained_user_turn_count == 0
        ):
            return EpistemicGuardDecision("missing_history", _UNAVAILABLE, "opening_unavailable")
        if context.history_truncated and _BEFORE_VISIBLE.search(text) and _TURN_CONTENT.search(text):
            return EpistemicGuardDecision("missing_history", _UNAVAILABLE, "before_window_unavailable")
        if _previous_requested(text):
            asks_user = bool(re.search(r"\b(?:i|my)\b", text))
            retained = (context.latest_prior_user_turn_retained if asks_user
                        else context.latest_prior_turn_retained)
            if not retained:
                return EpistemicGuardDecision("missing_history", _UNAVAILABLE, "previous_unavailable")

    if context.retained_history_count == 0 and _ZERO_REFERENT.fullmatch(text):
        return EpistemicGuardDecision(
            "ambiguous_reference", "What are you referring to?", "no_conversational_referent",
        )

    if (_ENV.search(text) and _PROJECT.search(text) and _ENV_QUERY.search(text)
            and _ACTUAL_ENV.search(text) and not _PROPOSED_ENV.search(text)):
        # Retained user messages and trusted memory values are potential evidence.
        # Assistant guesses, runtime instructions and memory command frames are not.
        if any(_has_env_candidate(message["content"]) for message in context.messages
               if message["role"] == "user") or _trusted_memory_has_candidate(context):
            return None
        return EpistemicGuardDecision(
            "unknown_internal_env_var",
            "The available context doesn't contain the exact environment-variable name. "
            "Share the relevant config or source code to identify it.",
            "internal_env_name_unavailable",
        )
    return None
