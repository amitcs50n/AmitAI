"""Deterministic RP review signals, not semantic verdicts or training approval.

No network, model, tokenizer or card execution. Evidence contains only one-based
message numbers, never source excerpts. English heuristics have limited recall.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise

VERSION = "rp_review_v2"


def pattern(value: str) -> re.Pattern:
    return re.compile(value, re.IGNORECASE)


def text_of(message: dict) -> str:
    value = message["content"]
    return value if isinstance(value, str) else "\n".join(p["text"] for p in value)


@dataclass(frozen=True)
class Signal:
    code: str
    family: str
    messages: tuple[int, ...]
    routing: str = "review"

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "family": self.family,
            "messages": list(self.messages),
            "routing": self.routing,
        }


# Mask identity/educational senses, not the participant's identity itself.
NONSEXUAL_SENSE = pattern(
    r"\b(?:same[- ]sex|opposite[- ]sex|sexual (?:orientation|identity|health|education)|"
    r"sex (?:assigned at birth|education)|sex[- ]based discrimination|"
    r"fuck (?:off|you)|fucking (?=(?:weather|door|bridge|sword)\b)|"
    r"cocks? (?:his|her|their|the|your) (?:head|gun|rifle|pistol|hammer)|"
    r"arous\w* (?:curiosity|suspicion|interest))\b"
)


def sexual_matches(text: str, sexual_pattern: re.Pattern) -> list[re.Match]:
    masked = NONSEXUAL_SENSE.sub(lambda m: " " * len(m[0]), text)
    return list(sexual_pattern.finditer(masked))


def linked_indicator(text: str, sexual_pattern: re.Pattern, indicator: re.Pattern) -> bool:
    """Legacy hard rules require local evidence, not distant unrelated mentions."""
    return any(
        indicator.search(text[max(0, m.start() - 180) : m.end() + 180])
        for m in sexual_matches(text, sexual_pattern)
    )


EUPHEMISM = pattern(
    r"\b(?:chastity (?:cage|contract|device)|paypig|foreplay|aphrodisiac\w*|"
    r"adult nursing|foot fetish|fetish\w*|kinks?|(?:soft|hard)?vore|blowjobs|dicks|cocks|"
    r"gets? off on|getting off on|mating response|breeding stock|"
    r"pleasure[- ](?:inducing|pods?|coils?|restraints?)|arous(?:al|ed|e)|"
    r"nipple\w*|carnal|libido|untouched virgins?|exploit (?:their|your) bodies)\b"
)
INTIMACY = pattern(
    r"\b(?:kiss\w*|intimacy|intimate|seduc\w*|seduct\w*|romantic|romance|"
    r"flirt\w*|arous\w*|foreplay|chastity|caress\w*|pleasure|sensual)\b"
)
YOUTH_CONTEXT = pattern(
    r"\b(?:schoolgirl|schoolboy|underage|preteen|elementary (?:school|branch)|"
    r"high school|barely legal|nursery|pacifier|baby talk)\b|"
    r"\b(?:she is|he is|i am|they are|character age[:=]?)\s+(?:an?\s+)?"
    r"(?:[1-9]|1[0-7])\s*(?:[- ]years?[- ]old|years? old|y/?o\b)"
)
ALL_ADULTS = pattern(r"\ball (?:characters|participants) (?:are|must be) (?:adults|(?:aged )?18\+)")
IGNORE_STOP = pattern(
    r"\b(?:ignor\w*|disregard\w*|dismiss\w*|overrid\w*)\b[^.!?\n]{0,85}"
    r"\b(?:safe[- ]?words?|stop signals?|refusals?|withdrawal|consent|free will|protests?)\b|"
    r"\b(?:even if|despite)\b[^.!?\n]{0,70}\b(?:beg\w*|ask\w*|say\w*)\b"
    r"[^.!?\n]{0,30}\bstop\b"
)
FORCED_INTIMACY = pattern(
    r"\b(?:forc\w*|coerc\w*|compel\w*)\b[^.!?\n]{0,65}"
    r"\b(?:intimacy|intimate contact|kissing|affection|romance)\b|"
    r"\b(?:kiss\w*|intimate contact)\b[^.!?\n]{0,50}\bwithout consent\b"
)
NEGATED = pattern(
    r"\b(?:never|not|don't|doesn't|do not|does not|won't|will not|wouldn't|mustn't)\s+(?:ever\s+)?$"
)
REFUSAL = pattern(
    r"(?:^|[.!?\n])\s*(?:[\w{}]+:\s*)?(?:no[,.!]|stop\b)|"
    r"\b(?:please stop|let me go|leave me alone|do not touch|don't touch|"
    r"not comfortable|withdraw my consent|take (?:it|that|the \w+) away|"
    r"(?:just|only) the \w+ is enough|maybe just the \w+ is enough)\b"
)
PERSIST = pattern(
    r"\b(?:refus\w* to (?:stop|release)|(?:keep\w*|continu\w*) (?:touch\w*|kiss\w*|restrain\w*)|"
    r"tighten\w* (?:\w+ ){0,3}grip|won't let (?:you|\w+) go|will not let (?:you|\w+) go)\b"
)
CONTRACT = pattern(r"\b(?:contract|debt|payment|paypig|purchased?|bought|auction|ownership)\b")
CONTROL = pattern(
    r"\b(?:no (?:choice|escape)|must obey|belong to me|own you|own that part|"
    r"extend\w*|punish\w*|property|obedience|no matter|war loot|exploit\w*)\b"
)
INTOXICATED = pattern(r"\b(?:drunk\w*|intoxicat\w*|inebriat\w*|too impaired|drugged|unconscious)\b")
POWER = pattern(
    r"\b(?:boss|employee|employer|priest|confessor|therapist|patient|caregiver|"
    r"dependent|dependency|financial control|slaver\w*|servitude|blackmail\w*)\b"
)
IDENTITY_CONTEXT = pattern(
    r"\b(?:real[- ]person|celebrity|celebrities|famouspeople|public figure|"
    r"vtuber|hololive|my (?:coworker|colleague|ex-partner))\b"
)

USER_TARGET = r"(?:you|\{\{user\}\}|the user)"
FORCED = pattern(
    r"\b(?:forc\w*|drag\w*|haul\w*|yank\w*|shov\w*|pin\w*|restrain\w*|"
    r"push\w*|pull\w*)\s+(?:\w+\s+){0,2}" + USER_TARGET + r"(?!\w)|"
    r"\b(?:grabs?|grips?|seizes?)\s+your\s+(?:collar|wrist|hair|neck|arm)\b|"
    r"\b(?:forces?|pushes?|pulls?|pins?)\s+your\s+(?:body|head|arms?|legs?|hands?)\b"
)
USER_SPEECH = pattern(r"(?<!\w)" + USER_TARGET + r"\s+(?:say|reply|answer|whisper|confess)(?:s)?\b")
USER_CHOICE = pattern(
    r"(?<!\w)" + USER_TARGET + r"\s+(?:decide|agree|choose|consent|realize|accept)(?:s)?\b"
)
USER_INNER = pattern(
    r"(?<!\w)" + USER_TARGET + r"\s+(?:secretly (?:wants?|desires?)|can't resist|cannot resist|"
    r"feel (?:a surge of |an? (?:overwhelming|irresistible) |suddenly )?"
    r"(?:desire|arousal|love|attracted|compelled)|know deep down)\b|"
    r"\byour (?:heart races|pulse accelerates|knees buckle|body betrays)\b"
)
CONDITIONAL = pattern(r"\b(?:if|when|unless|whether|should|would|could|will|do|did|can)\s+$")


def asserted(text: str, rule: re.Pattern, previous_user: str = "") -> bool:
    for match in rule.finditer(text):
        prefix = text[max(0, match.start() - 60) : match.start()]
        if NEGATED.search(prefix) or CONDITIONAL.search(prefix):
            continue
        # Quoted reminders of a previous user's own declared action are not new control.
        tail = re.findall(r"\w+", text[match.start() : match.end() + 45].casefold())[:6]
        declared = re.sub(r"\bi\b", "you", previous_user.casefold())
        if len(tail) >= 4 and " ".join(tail) in " ".join(re.findall(r"\w+", declared)):
            continue
        return True
    return False


def nearby(text: str, first: re.Pattern, second: re.Pattern, radius: int = 240) -> bool:
    return any(
        second.search(text[max(0, m.start() - radius) : m.end() + radius])
        for m in first.finditer(text)
    )


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.casefold())


def review_signals(record: dict, filters: dict, sexual_pattern: re.Pattern) -> tuple[Signal, ...]:
    messages = [(i + 1, m["role"], text_of(m)) for i, m in enumerate(record["messages"])]
    evidence: dict[tuple[str, str, str], set[int]] = defaultdict(set)

    def add(code: str, family: str, ids: list[int], routing: str = "review") -> None:
        evidence[code, family, routing].update(ids)

    intimate_ids = [i for i, _, s in messages if INTIMACY.search(s)]
    sexual_ids = [
        i
        for i, _, s in messages
        if sexual_matches(s, sexual_pattern) or sexual_matches(s, EUPHEMISM)
    ]
    system = [(i, s) for i, role, s in messages if role == "system"]
    sexual_card = any(i in sexual_ids for i, _ in system)
    # User-supplied speaker headers, never a guessed proper-name dictionary.
    user_names = set()
    assistant_labels = Counter()
    for _, role, text in messages:
        label = re.match(r"^([A-Z][\w '-]{1,35})(?::\s|\n)", text)
        if label and role == "user":
            user_names.add(label[1])
        if label and role == "assistant":
            assistant_labels[label[1]] += 1
    named_forced = None
    named_rules = []
    if user_names:
        targets = "(?:" + "|".join(re.escape(n) for n in sorted(user_names)) + ")"
        named_forced = pattern(
            r"\b(?:forces?|drags?|hauls?|yanks?|shoves?|pins?|restrains?|pushes?|pulls?)\s+"
            + targets
            + r"\b|\b(?:grabs?|grips?|seizes?)\s+"
            + targets
            + r"['\u2019]s\s+"
            r"(?:collar|wrist|hair|neck|arm)\b"
        )
        named_rules = [
            (pattern(rule.pattern.replace(USER_TARGET, targets)), code)
            for rule, code in (
                (USER_SPEECH, "user_dialogue_review"),
                (USER_CHOICE, "user_decision_review"),
                (USER_INNER, "user_internal_state_review"),
            )
        ]
    previous_user = ""
    for i, role, text in messages:
        if sexual_matches(text, EUPHEMISM):
            add("sexual_context_review", "consent_age_identity", [i])
        if asserted(text, IGNORE_STOP):
            add("ignored_stop_review", "consent_age_identity", [i])
        if asserted(text, FORCED_INTIMACY):
            add("forced_intimacy_review", "consent_age_identity", [i])
        if nearby(text, CONTRACT, CONTROL) and (INTIMACY.search(text) or EUPHEMISM.search(text)):
            add("coercive_contract_review", "consent_age_identity", [i])
        if INTOXICATED.search(text) and (sexual_card or nearby(text, INTOXICATED, INTIMACY)):
            add("intoxication_consent_review", "consent_age_identity", [i])
        if nearby(text, POWER, INTIMACY) or (nearby(text, POWER, EUPHEMISM)):
            add("intimate_power_imbalance_review", "consent_age_identity", [i])
        if role == "assistant":
            if REFUSAL.search(previous_user) and (
                asserted(text, PERSIST) or asserted(text, FORCED)
            ):
                add("ignored_stop_review", "consent_age_identity", [i - 1, i])
            for rule, code in (
                (USER_SPEECH, "user_dialogue_review"),
                (USER_CHOICE, "user_decision_review"),
                (USER_INNER, "user_internal_state_review"),
                (FORCED, "forced_user_movement_review"),
            ):
                if asserted(text, rule, previous_user):
                    add(code, "agency", [i])
            if named_forced and asserted(text, named_forced, previous_user):
                add("forced_user_movement_review", "agency", [i])
            for rule, code in named_rules:
                if asserted(text, rule, previous_user):
                    add(code, "agency", [i])
            if re.search(r"(?im)^\s*(?:\{\{user\}\}|user|human|you)\s*:", text):
                add("user_dialogue_review", "agency", [i])
            # Detect a later character label after a substantial unlabelled speech block.
            # This is only a possible splice: some multi-NPC scenes legitimately use it.
            labels = list(re.finditer(r"(?m)^([\w .'-]{2,50}):\s", text))
            for label in labels[:1]:
                if (
                    label.start() > 180
                    and assistant_labels[label[1]] >= 2
                    and re.search(r"\b(?:I|my|we|our)\b", text[: label.start()])
                ):
                    add("possible_speaker_splice_review", "agency", [i])
            if re.search(
                r"(?i)let me know if (?:this|you).{0,120}(?:requirements|adjustments|changes)|"
                r"(?:as requested|here is) (?:the|your) (?:rewritten|revised) (?:response|scene)",
                text,
            ):
                add("editor_outro_review", "contamination", [i])
        if role == "user":
            previous_user = text
        if role == "system":
            for rule, code in (
                (r"</?(?:p|div|span|strong|img|script|iframe)\b", "card_html_review"),
                (r"<img\b|!\[[^\]]*\]\(https?://", "card_remote_image_review"),
                (
                    r"\b(?:END_OF_DIALOG|START_OF_DIALOG)\b|<START>|</?EXAMPLES>",
                    "card_legacy_delimiter_review",
                ),
                (
                    r"\{\{(?:char|user)[\]}](?!\})|(?<!\{)\{(?:char|user)\}\}",
                    "card_placeholder_review",
                ),
                (
                    (
                        r"\b(?:creator'?s? notes?|author'?s? notes?|recommended (?:model|settings|gpt[\w.-]*)|"
                        r"SillyTavern|\d+% (?:context |token )?capacity)\b"
                    ),
                    "card_creator_notes_review",
                ),
                (
                    r"\bonly write.{0,60}(?:for|as) \{\{user\}\}|\b(?:model|editor) instructions\b",
                    "card_model_instructions_review",
                ),
            ):
                if re.search(rule, text, re.IGNORECASE):
                    add(code, "contamination", [i])

    full = "\n".join(s for _, _, s in messages)
    # Magical-energy intimacy is a contextual euphemism, not a universal mana ban.
    mana_ids = [
        i
        for i, _, s in messages
        if nearby(s, pattern(r"\b(?:mana|magic(?:al)? energy)\b"), INTIMACY)
    ]
    if mana_ids:
        add("sexual_euphemism_review", "consent_age_identity", mana_ids)
        sexual_ids += mana_ids
    youth_ids = [i for i, _, s in messages if YOUTH_CONTEXT.search(s)]
    if sexual_ids:
        if not ALL_ADULTS.search(full):
            add("participant_age_review", "consent_age_identity", sexual_ids)
        if youth_ids:
            add("youth_intimacy_context_review", "consent_age_identity", youth_ids + sexual_ids)
    for i, role, text in messages:
        identity = IDENTITY_CONTEXT.search(text) or any(
            re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE)
            for name in filters.get("known_real_person_names", [])
        )
        if identity and (sexual_ids or intimate_ids) and role == "system":
            add("intimate_identity_review", "consent_age_identity", [i] + intimate_ids + sexual_ids)

    # Partial example copying: contiguous twelve-word windows cover enough of a
    # substantive reply, ignoring short common phrases and all initial greetings.
    card_grams = set()
    for _, s in system:
        words = tokens(s)
        card_grams.update(tuple(words[j : j + 12]) for j in range(len(words) - 11))
    seen_user = False
    for i, role, text in messages:
        seen_user |= role == "user"
        if role != "assistant" or not seen_user:
            continue
        words = tokens(text)
        covered = set()
        for j in range(len(words) - 11):
            if tuple(words[j : j + 12]) in card_grams:
                covered.update(range(j, j + 12))
        if len(covered) >= 12 and len(covered) / max(1, len(words)) >= 0.55:
            add("card_example_copy_review", "contamination", [i])

    _repetition(messages, add)
    return tuple(
        Signal(code, family, tuple(sorted(ids)), routing)
        for (code, family, routing), ids in sorted(evidence.items())
    )


GESTURES = {
    "laugh": pattern(r"\b(?:laugh\w*|chuckl\w*|giggl\w*)\b"),
    "smile": pattern(r"\b(?:smil\w*|smirk\w*|grin\w*)\b"),
    "wink": pattern(r"\bwink\w*\b"),
    "blush": pattern(r"\b(?:blush\w*|flush\w*)\b"),
    "gaze": pattern(r"\b(?:gaze|eyes?)\b.{0,45}\b(?:lock\w*|glint\w*|spark\w*|narrow\w*)\b"),
    "lean": pattern(r"\blean\w*\b"),
    "adjust": pattern(r"\badjust\w*\b"),
    "distress": pattern(r"\b(?:sob\w*|crying|cries|tears|trembl\w*)\b"),
}
RESET = pattern(
    r"\b(?:first (?:real |genuine |true )?(?:smile|laugh)[^.!?\n]{0,55}"
    r"\b(?:weeks|months|years)|"
    r"(?:smil\w*|laugh\w*)[^.!?\n]{0,85}first[^.!?\n]{0,35}"
    r"\b(?:weeks|months|years))\b"
)
STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "in",
    "on",
    "with",
    "is",
    "it",
    "he",
    "she",
    "his",
    "her",
    "you",
    "your",
    "i",
    "my",
    "as",
    "that",
    "this",
    "for",
    "at",
    "but",
}
# Small transparent normalization for approximate lexical loops, not embeddings.
SYNONYMS = {
    word: root
    for root, group in {
        "smile": "smiles smiled smiling grins grinned smirks",
        "laugh": "laughs laughed chuckles chuckled giggles",
        "quiet": "quietly softly gently",
        "fear": "afraid frightened scared fearful",
        "reply": "says replies answers responds",
    }.items()
    for word in group.split()
}


def _repetition(messages: list[tuple[int, str, str]], add) -> None:
    replies = [(i, s) for i, role, s in messages if role == "assistant"]
    if len(replies) < 3:
        return
    phrase_ids = defaultdict(list)
    action_ids = defaultdict(list)
    reset_ids = []
    vocab = []
    for i, s in replies:
        words = tokens(s)
        for gram in {tuple(words[j : j + 6]) for j in range(len(words) - 5)}:
            phrase_ids[gram].append(i)
        actions = " ".join(re.findall(r"\*([^*]+)\*", s))
        for key, rule in GESTURES.items():
            if rule.search(actions):
                action_ids[key].append(i)
        if RESET.search(s):
            reset_ids.append(i)
        vocab.append((i, {SYNONYMS.get(w, w) for w in words if w not in STOPWORDS}))
    recurring = [
        ids for ids in phrase_ids.values() if len(ids) >= 4 and len(ids) >= len(replies) * 0.5
    ]
    if recurring:
        add(
            "recurring_phrase_diagnostic",
            "repetition",
            sorted({i for ids in recurring for i in ids}),
            "diagnostic",
        )
    gestures = [
        ids for ids in action_ids.values() if len(ids) >= 4 and len(ids) >= len(replies) * 0.6
    ]
    if gestures:
        add(
            "recurring_gesture_diagnostic",
            "repetition",
            sorted({i for ids in gestures for i in ids}),
            "diagnostic",
        )
    # Several persistent gesture families, not a lone catchphrase or action.
    if len(gestures) >= 3 and len(replies) >= 6:
        add(
            "repeated_action_template_review",
            "repetition",
            sorted({i for ids in gestures for i in ids}),
        )
    if len(reset_ids) >= 2:
        add("repeated_emotional_reset_review", "repetition", reset_ids)
    pairs = []
    for (i, a), (j, b) in pairwise(vocab):
        if min(len(a), len(b)) >= 14 and len(a & b) / len(a | b) >= 0.62:
            pairs.append((i, j))
    if len(pairs) >= 2 and len(pairs) >= (len(replies) - 1) * 0.5:
        add(
            "possible_response_loop_review",
            "repetition",
            sorted({i for pair in pairs for i in pair}),
        )
