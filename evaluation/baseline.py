from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol

from evaluation.constraints import validate_with_one_retry


EVAL_REQUIRED_FIELDS = {
    "id",
    "spec_version",
    "category",
    "primary_rules",
    "prompt",
    "pass_criteria",
    "failure_signals",
}
VALID_SCORES = {0, 1, 2}


class TextGenerator(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        generation_config: dict[str, Any],
    ) -> str: ...


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: each JSONL row must be an object")
        rows.append(row)
    return rows


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = load_jsonl(path)
    if not cases:
        raise ValueError("Evaluation file contains no cases")

    seen_ids: set[str] = set()
    for index, case in enumerate(cases, 1):
        missing = EVAL_REQUIRED_FIELDS - case.keys()
        if missing:
            raise ValueError(f"Evaluation row {index} is missing fields: {sorted(missing)}")

        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Evaluation row {index} has an invalid id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation id: {case_id}")
        seen_ids.add(case_id)

        if not isinstance(case["prompt"], str) or not case["prompt"].strip():
            raise ValueError(f"{case_id}: prompt must be a non-empty string")
        for field in ("primary_rules", "pass_criteria", "failure_signals"):
            value = case[field]
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                raise ValueError(f"{case_id}: {field} must be a non-empty string list")
    return cases


def build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": system_prompt.strip(),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    return messages


def generate_case(
    case: dict[str, Any],
    generator: TextGenerator,
    *,
    system_prompt: str | None,
    generation_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = generator.generate(
        build_messages(case["prompt"], system_prompt),
        generation_config,
    )
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{case['id']}: model returned an empty response")
    response = response.strip()

    result = {
        "schema_version": 1,
        "id": case["id"],
        "spec_version": case["spec_version"],
        "category": case["category"],
        "primary_rules": case["primary_rules"],
        "prompt": case["prompt"],
        "response": response,
    }
    review = {
        **result,
        "pass_criteria": case["pass_criteria"],
        "failure_signals": case["failure_signals"],
        "rule_scores": {rule_id: None for rule_id in case["primary_rules"]},
        "critical_failure": None,
        "notes": "",
    }
    return result, review


def generate_constrained_case(
    case: dict[str, Any],
    generator: TextGenerator,
    *,
    system_prompt: str | None,
    generation_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_response = generator.generate(
        build_messages(case["prompt"], system_prompt),
        generation_config,
    )
    if not isinstance(original_response, str) or not original_response.strip():
        raise ValueError(f"{case['id']}: model returned an empty response")
    original_response = original_response.strip()

    def retry(corrective_prompt: str) -> str:
        return generator.generate(
            build_messages(corrective_prompt, system_prompt),
            generation_config,
        )

    constraint_metadata = validate_with_one_retry(
        case["prompt"],
        original_response,
        retry,
    )
    result = {
        "schema_version": 1,
        "id": case["id"],
        "spec_version": case["spec_version"],
        "category": case["category"],
        "primary_rules": case["primary_rules"],
        "prompt": case["prompt"],
        "response": constraint_metadata["final_response"],
        **constraint_metadata,
    }
    review = {
        **result,
        "pass_criteria": case["pass_criteria"],
        "failure_signals": case["failure_signals"],
        "rule_scores": {rule_id: None for rule_id in case["primary_rules"]},
        "critical_failure": None,
        "notes": "",
    }
    return result, review


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_reviews_against_cases(
    reviews: Iterable[dict[str, Any]],
    cases: Iterable[dict[str, Any]],
) -> None:
    case_by_id = {case["id"]: case for case in cases}
    immutable_fields = (
        "spec_version",
        "category",
        "primary_rules",
        "prompt",
        "pass_criteria",
        "failure_signals",
    )
    for review in reviews:
        case_id = review.get("id")
        if case_id not in case_by_id:
            raise ValueError(f"Unknown review id: {case_id}")
        case = case_by_id[case_id]
        for field in immutable_fields:
            if review.get(field) != case[field]:
                raise ValueError(f"{case_id}: review field {field} differs from the eval case")
        if not isinstance(review.get("response"), str) or not review["response"].strip():
            raise ValueError(f"{case_id}: review response must be a non-empty string")


def validate_reviews_against_responses(
    reviews: Iterable[dict[str, Any]],
    responses: Iterable[dict[str, Any]],
) -> None:
    response_rows = list(responses)
    response_by_id = {row.get("id"): row for row in response_rows}
    if len(response_by_id) != len(response_rows):
        raise ValueError("Responses contain duplicate ids")

    review_rows = list(reviews)
    review_by_id = {row.get("id"): row for row in review_rows}
    if len(review_by_id) != len(review_rows):
        raise ValueError("Reviews contain duplicate ids")
    if set(review_by_id) != set(response_by_id):
        raise ValueError("Review ids do not exactly match response ids")

    for case_id, review in review_by_id.items():
        response = response_by_id[case_id]
        for field in response:
            if field not in review or review[field] != response[field]:
                raise ValueError(
                    f"{case_id}: review field {field} differs from generated responses"
                )


def summarize_reviews(
    rows: Iterable[dict[str, Any]],
    *,
    minimum_rule_compliance_rate: float,
    maximum_critical_failures: int,
    expected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    reviews = list(rows)
    if not 0.0 <= minimum_rule_compliance_rate <= 1.0:
        raise ValueError("minimum_rule_compliance_rate must be between 0 and 1")
    if maximum_critical_failures < 0:
        raise ValueError("maximum_critical_failures must be non-negative")

    seen_ids: set[str] = set()
    review_order: list[str] = []
    incomplete_present: set[str] = set()
    completed: list[dict[str, Any]] = []
    for row in reviews:
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every review row must have an id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate review id: {case_id}")
        seen_ids.add(case_id)
        review_order.append(case_id)

        rule_scores = row.get("rule_scores")
        expected_rule_ids = row.get("primary_rules")
        if (
            not isinstance(expected_rule_ids, list)
            or not isinstance(rule_scores, dict)
            or set(rule_scores) != set(expected_rule_ids)
        ):
            raise ValueError(f"{case_id}: rule_scores must match primary_rules exactly")
        for rule_id, score in rule_scores.items():
            if score is None:
                continue
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or score not in VALID_SCORES
            ):
                raise ValueError(f"{case_id}/{rule_id}: score must be 0, 1, or 2")

        critical_failure = row.get("critical_failure")
        if any(score is None for score in rule_scores.values()) or critical_failure is None:
            incomplete_present.add(case_id)
            continue
        if not isinstance(critical_failure, bool):
            raise ValueError(f"{case_id}: critical_failure must be true or false")
        completed.append(row)

    if expected_ids is None:
        if not reviews:
            raise ValueError("Review file contains no rows")
        expected_order = review_order
    else:
        expected_order = list(expected_ids)
        if len(expected_order) != len(set(expected_order)):
            raise ValueError("Expected review ids contain duplicates")
        unexpected = seen_ids - set(expected_order)
        if unexpected:
            raise ValueError(f"Review file contains unexpected ids: {sorted(unexpected)}")
    incomplete_ids = [
        case_id
        for case_id in expected_order
        if case_id not in seen_ids or case_id in incomplete_present
    ]

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        reviewed = len(group)
        rule_scores = [
            score
            for row in group
            for score in row["rule_scores"].values()
        ]
        case_scores = [min(row["rule_scores"].values()) for row in group]
        passed = sum(
            min(row["rule_scores"].values()) == 2 and not row["critical_failure"]
            for row in group
        )
        critical = sum(row["critical_failure"] for row in group)
        return {
            "reviewed": reviewed,
            "passed": passed,
            "case_pass_rate": round(passed / reviewed, 4) if reviewed else None,
            "mean_case_score": (
                round(sum(case_scores) / reviewed, 4)
                if reviewed
                else None
            ),
            "rule_assessments": len(rule_scores),
            "rules_met": sum(score == 2 for score in rule_scores),
            "rule_compliance_rate": (
                round(sum(score == 2 for score in rule_scores) / len(rule_scores), 4)
                if rule_scores
                else None
            ),
            "mean_rule_score": (
                round(sum(rule_scores) / len(rule_scores), 4)
                if rule_scores
                else None
            ),
            "critical_failures": critical,
        }

    by_category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_rule_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_category_rows[str(row["category"])].append(row)
        for rule_id in row["primary_rules"]:
            by_rule_rows[str(rule_id)].append(row)

    overall = aggregate(completed)
    if incomplete_ids:
        decision = "review_incomplete"
    elif (
        overall["rule_compliance_rate"] is not None
        and overall["rule_compliance_rate"] >= minimum_rule_compliance_rate
        and overall["critical_failures"] <= maximum_critical_failures
    ):
        decision = "baseline_meets_gate"
    else:
        decision = "fine_tuning_candidate"

    return {
        "schema_version": 1,
        "decision": decision,
        "gate": {
            "minimum_rule_compliance_rate": minimum_rule_compliance_rate,
            "maximum_critical_failures": maximum_critical_failures,
        },
        "total_cases": len(expected_order),
        "completed_reviews": len(completed),
        "incomplete_ids": incomplete_ids,
        "overall": overall,
        "by_category": {
            name: aggregate(group) for name, group in sorted(by_category_rows.items())
        },
        "by_rule": {
            name: {
                "reviewed": len(group),
                "meets_rule": sum(row["rule_scores"][name] == 2 for row in group),
                "compliance_rate": round(
                    sum(row["rule_scores"][name] == 2 for row in group) / len(group),
                    4,
                ),
                "mean_score": round(
                    sum(row["rule_scores"][name] for row in group) / len(group),
                    4,
                ),
                "associated_critical_failures": sum(
                    row["critical_failure"] for row in group
                ),
            }
            for name, group in sorted(by_rule_rows.items())
        },
    }
