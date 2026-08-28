from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evaluation.baseline import (
    append_jsonl,
    generate_case,
    load_eval_cases,
    load_jsonl,
    sha256_file,
    stable_fingerprint,
    write_json,
)
from evaluation.hf_backend import TransformersGenerator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def evaluation_code_sha256() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("baseline_eval"), dict):
        raise ValueError("Config must contain a baseline_eval object")
    return config["baseline_eval"]


def select_eval_cases(
    cases: list[dict[str, Any]],
    *,
    ids: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")

    selected = list(cases)
    if ids is not None:
        requested_tokens = [token.strip() for token in ids.split(",")]
        if any(not token for token in requested_tokens):
            raise ValueError("--ids contains an empty eval ID token")

        requested_ids = set(requested_tokens)
        available_ids = {case["id"] for case in cases}
        unknown_ids = sorted(requested_ids - available_ids)
        if unknown_ids:
            raise ValueError(f"Unknown eval ID(s): {', '.join(unknown_ids)}")
        selected = [case for case in cases if case["id"] in requested_ids]

    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Evaluation case selection produced zero cases")
    return selected


def run(
    config_path: str | Path,
    *,
    ids: str | None = None,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")

    config = load_config(config_path)
    runtime_system_prompt = config.get("runtime_system_prompt")
    if not isinstance(runtime_system_prompt, str) or not runtime_system_prompt.strip():
        raise ValueError("Config must contain a non-empty runtime_system_prompt")
    eval_path = Path(config["eval_file"])
    output_dir = Path(config["output_dir"])
    responses_path = output_dir / "responses.jsonl"
    reviews_path = output_dir / "reviews.jsonl"
    manifest_path = output_dir / "run.json"
    summary_path = output_dir / "summary.json"
    cases = load_eval_cases(eval_path)
    cases = select_eval_cases(cases, ids=ids, limit=limit)

    fingerprint_payload = {
        "run_name": config.get("name"),
        "model": config["model"],
        "eval_sha256": sha256_file(eval_path),
        "case_ids": [case["id"] for case in cases],
        "runtime_system_prompt": runtime_system_prompt,
        "generation": config["generation"],
        "decision_gate": config["decision_gate"],
    }
    fingerprint = stable_fingerprint(fingerprint_payload)
    output_files = (responses_path, reviews_path, manifest_path, summary_path)

    if overwrite:
        for path in output_files:
            path.unlink(missing_ok=True)
    elif any(path.exists() for path in output_files) and not resume:
        raise FileExistsError(
            f"{output_dir} already contains a run; use --resume or --overwrite"
        )

    existing_responses = load_jsonl(responses_path) if responses_path.exists() else []
    existing_reviews = load_jsonl(reviews_path) if reviews_path.exists() else []
    completed_ids = {row["id"] for row in existing_responses}
    if len(completed_ids) != len(existing_responses):
        raise ValueError("Existing responses contain duplicate ids")
    expected_ids = {case["id"] for case in cases}
    if not completed_ids <= expected_ids:
        raise ValueError("Existing responses do not match the selected evaluation cases")

    existing_manifest: dict[str, Any] = {}
    if resume and (responses_path.exists() or reviews_path.exists()) and not manifest_path.exists():
        raise ValueError("Cannot resume partial artifacts without run.json")
    if resume and manifest_path.exists():
        loaded_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_manifest, dict):
            raise ValueError("Cannot resume because run.json is invalid")
        existing_manifest = loaded_manifest
        if existing_manifest.get("fingerprint") != fingerprint:
            raise ValueError("Cannot resume because the config or evaluation set changed")
        current_code_revision = git_revision()
        if (
            existing_manifest.get("code_revision") is not None
            and current_code_revision != existing_manifest["code_revision"]
        ):
            raise ValueError("Cannot resume after the repository revision changed")
        if existing_manifest.get("code_sha256") != evaluation_code_sha256():
            raise ValueError("Cannot resume after evaluation code changed")
        started_at = existing_manifest["started_at_utc"]
    else:
        started_at = utc_now()

    review_ids = {row["id"] for row in existing_reviews}
    if len(review_ids) != len(existing_reviews):
        raise ValueError("Existing reviews contain duplicate ids")
    if not review_ids <= completed_ids:
        raise ValueError("Existing reviews contain ids without matching responses")
    case_by_id = {case["id"]: case for case in cases}
    for response in existing_responses:
        if response["id"] in review_ids:
            continue
        case = case_by_id[response["id"]]
        append_jsonl(
            reviews_path,
            {
                **response,
                "pass_criteria": case["pass_criteria"],
                "failure_signals": case["failure_signals"],
                "rule_scores": {
                    rule_id: None for rule_id in case["primary_rules"]
                },
                "critical_failure": None,
                "notes": "",
            },
        )

    pending = [case for case in cases if case["id"] not in completed_ids]
    if pending:
        summary_path.unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_type": "base_model_baseline",
        "run_name": config.get("name"),
        "status": "running",
        "fingerprint": fingerprint,
        "started_at_utc": started_at,
        "updated_at_utc": utc_now(),
        "code_revision": git_revision(),
        "code_sha256": evaluation_code_sha256(),
        "dependency_versions": existing_manifest.get("dependency_versions"),
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model": config["model"],
        "resolved_model_revision": existing_manifest.get("resolved_model_revision"),
        "eval_file": str(eval_path),
        "eval_sha256": fingerprint_payload["eval_sha256"],
        "case_ids": fingerprint_payload["case_ids"],
        "runtime_system_prompt": runtime_system_prompt,
        "generation": config["generation"],
        "selected_case_count": len(cases),
        "completed_case_count": len(completed_ids),
        "responses_file": str(responses_path),
        "reviews_file": str(reviews_path),
    }
    write_json(manifest_path, manifest)

    if not pending:
        manifest.update(
            status="complete",
            updated_at_utc=utc_now(),
            responses_sha256=sha256_file(responses_path),
        )
        write_json(manifest_path, manifest)
        return output_dir

    try:
        model_config = dict(config["model"])
        if existing_manifest.get("resolved_model_revision"):
            model_config["revision"] = existing_manifest["resolved_model_revision"]
        generator = TransformersGenerator(
            model_config,
            seed=int(config["generation"].get("seed", 3407)),
        )
        if (
            existing_manifest.get("dependency_versions") is not None
            and generator.dependency_versions != existing_manifest["dependency_versions"]
        ):
            raise ValueError("Cannot resume after inference dependencies changed")
        manifest["resolved_model_revision"] = generator.resolved_revision
        manifest["dependency_versions"] = generator.dependency_versions
        write_json(manifest_path, manifest)

        for case in pending:
            response, review = generate_case(
                case,
                generator,
                system_prompt=runtime_system_prompt,
                generation_config=config["generation"],
            )
            append_jsonl(responses_path, response)
            append_jsonl(reviews_path, review)
            completed_ids.add(case["id"])
            manifest.update(
                completed_case_count=len(completed_ids),
                updated_at_utc=utc_now(),
            )
            write_json(manifest_path, manifest)
            print(f"[{len(completed_ids)}/{len(cases)}] {case['id']}")
    except Exception as exc:
        manifest.update(
            status="failed",
            updated_at_utc=utc_now(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        write_json(manifest_path, manifest)
        raise

    manifest.update(
        status="complete",
        updated_at_utc=utc_now(),
        responses_sha256=sha256_file(responses_path),
    )
    write_json(manifest_path, manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate held-out responses from the untouched base model"
    )
    parser.add_argument(
        "--config",
        default="configs/baseline_eval.yaml",
        help="Path to the baseline evaluation YAML",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated evaluation case IDs",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N cases")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--resume", action="store_true")
    run_mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = run(
        args.config,
        ids=args.ids,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(f"Baseline artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
