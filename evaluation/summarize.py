from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evaluation.baseline import (
    load_eval_cases,
    load_jsonl,
    sha256_file,
    summarize_reviews,
    validate_reviews_against_cases,
    validate_reviews_against_responses,
    write_json,
)
from evaluation.run_baseline import evaluation_code_sha256, load_config


def summarize_run(
    config_path: str | Path,
    *,
    reviews_override: str | Path | None = None,
    output_override: str | Path | None = None,
) -> tuple[Path, dict]:
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    reviews_path = (
        Path(reviews_override) if reviews_override else output_dir / "reviews.jsonl"
    )
    summary_path = (
        Path(output_override) if output_override else output_dir / "summary.json"
    )
    manifest_path = output_dir / "run.json"
    eval_path = Path(config["eval_file"])
    cases = load_eval_cases(eval_path)
    reviews = load_jsonl(reviews_path)
    validate_reviews_against_cases(reviews, cases)

    if not manifest_path.exists():
        raise ValueError(f"Missing baseline manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = [case["id"] for case in cases]
    if manifest.get("status") != "complete":
        raise ValueError("Baseline generation is not complete")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("The baseline config changed after generation")
    if manifest.get("code_sha256") != evaluation_code_sha256():
        raise ValueError("Evaluation code changed after baseline generation")
    if manifest.get("eval_sha256") != sha256_file(eval_path):
        raise ValueError("The held-out eval file changed after baseline generation")
    if manifest.get("case_ids") != expected_ids:
        raise ValueError("Baseline run does not cover the complete held-out eval set")
    if manifest.get("completed_case_count") != len(expected_ids):
        raise ValueError("Baseline manifest has an incomplete case count")

    responses_path = Path(manifest["responses_file"])
    if manifest.get("responses_sha256") != sha256_file(responses_path):
        raise ValueError("Generated responses changed after the baseline run")
    validate_reviews_against_responses(reviews, load_jsonl(responses_path))

    gate = config["decision_gate"]
    summary = summarize_reviews(
        reviews,
        minimum_rule_compliance_rate=float(gate["minimum_rule_compliance_rate"]),
        maximum_critical_failures=int(gate["maximum_critical_failures"]),
        expected_ids=expected_ids,
    )
    write_json(summary_path, summary)
    return summary_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manually reviewed baseline results")
    parser.add_argument(
        "--config",
        default="configs/baseline_eval.yaml",
        help="Path to the baseline evaluation YAML",
    )
    parser.add_argument("--reviews", help="Override the reviews JSONL path")
    parser.add_argument("--output", help="Override the summary JSON path")
    args = parser.parse_args()

    summary_path, summary = summarize_run(
        args.config,
        reviews_override=args.reviews,
        output_override=args.output,
    )
    print(yaml.safe_dump(summary, sort_keys=False).strip())
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
