"""Bounded, deterministic RP candidate preparation; never imports training code."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

import httpx
import yaml

from training.rp_data import (
    RowError,
    assess,
    fingerprint,
    message_text,
    normalize_conversation,
    record_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/rp_v1_sources.yaml"
OUTPUT_ROOT = ROOT / "data/sft/rp_v1/generated"


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported RP manifest schema")
    # Validate the same required metadata as the existing loader.
    from training.data import normalize_example

    normalize_example(
        {
            "id": "rpv1_config_check",
            "spec_version": config["spec_version"],
            "category": config["category"],
            "primary_rules": config["primary_rules"],
            "messages": [
                {"role": "user", "content": "check"},
                {"role": "assistant", "content": "check"},
            ],
        }
    )
    for key, maximum in (
        ("accepted_per_source", 100),
        ("max_candidates_per_source", 5000),
        ("max_bytes_per_source", 16 * 1024 * 1024),
        ("max_record_bytes", 2 * 1024 * 1024),
        ("review_samples_per_bucket", 20),
    ):
        value = config["smoke"][key]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"Invalid smoke limit: {key}")
    for key in ("min_user_turns", "min_conversation_chars", "max_conversation_chars"):
        if type(config["filters"][key]) is not int or config["filters"][key] < 1:
            raise ValueError(f"Invalid filter limit: {key}")
    if config["filters"]["max_conversation_chars"] < config["filters"]["min_conversation_chars"]:
        raise ValueError("Inverted character limits")
    names = config["filters"].get("known_real_person_names", [])
    if not isinstance(names, list) or any(
        not isinstance(name, str) or len(name.strip()) < 3 for name in names
    ):
        raise ValueError("Known real-person names must be a list of non-empty names")
    if not isinstance(config["sources"], dict) or not config["sources"]:
        raise ValueError("No configured sources")
    for name, source in config["sources"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Invalid source name")
        for key in (
            "hf_repo",
            "file",
            "intended_usage",
            "declared_license",
            "provenance_notes",
            "evidence_url",
        ):
            if not isinstance(source[key], str) or not source[key].strip():
                raise ValueError(f"Missing source field: {name}.{key}")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", source["hf_repo"]):
            raise ValueError("Invalid Hugging Face repo")
        if not re.fullmatch(r"[\w.-]+\.jsonl", source["file"]):
            raise ValueError("Only a pinned JSONL file is supported")
        if not isinstance(source["revision"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", source["revision"]
        ):
            raise ValueError("Source revision must be a full immutable commit hash")
        if source["adapter"] not in {"pippa", "sharegpt"}:
            raise ValueError("Unsupported source adapter")
        for key in ("enabled_by_default", "manual_review_required"):
            if type(source[key]) is not bool:
                raise ValueError(f"Source field must be a boolean: {key}")
        if source["provenance_status"] not in {"pending", "reviewed"}:
            raise ValueError("Invalid provenance status")
        if source["provenance_status"] == "reviewed" and not (
            source.get("reviewed_on") and source.get("review_notes")
        ):
            raise ValueError("Reviewed provenance requires a date and review evidence")
    return config


class BoundedInput:
    """Read a prefix only, with no Hub scripts, whole-file cache, or retries.

    Hashes describe bytes actually read, not the full upstream file. A partial
    final line at a byte cap is never interpreted as a complete record.
    """

    def __init__(self, source: dict, limits: dict, local_path: Path | None = None):
        self.source, self.limits, self.local_path = source, limits, local_path
        self.bytes_read = 0
        self.digest = hashlib.sha256()
        self.stop_reason = "consumer_closed"

    def _chunks(self) -> Iterable[bytes]:
        cap = self.limits["max_bytes_per_source"]
        if self.local_path is not None:
            with self.local_path.open("rb") as stream:
                while self.bytes_read < cap:
                    chunk = stream.read(min(16384, cap - self.bytes_read))
                    if not chunk:
                        self.stop_reason = "exhausted"
                        return
                    self._count(chunk)
                    yield chunk
                self.stop_reason = "byte_limit" if stream.read(1) else "exhausted"
            return
        source = self.source
        url = (
            f"https://huggingface.co/datasets/{source['hf_repo']}/resolve/"
            f"{source['revision']}/{quote(source['file'])}"
        )
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=30,
            headers={"Range": f"bytes=0-{cap - 1}", "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            if response.status_code not in {200, 206}:
                raise ValueError("Unexpected source response")
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("Compressed responses are not supported by bounded ingestion")
            total = None
            if response.status_code == 206:
                match = re.fullmatch(
                    r"bytes 0-(\d+)/(\d+)", response.headers.get("content-range", "")
                )
                if not match or int(match[1]) >= cap:
                    raise ValueError("Invalid prefix range response")
                total = int(match[2])
            elif response.headers.get("content-length", "").isdigit():
                total = int(response.headers["content-length"])
            # iter_raw avoids decompression; a non-range server is still closed
            # at the client cap. At most one 16 KiB transport chunk is prefetched.
            for chunk in response.iter_raw(chunk_size=min(16384, cap)):
                remaining = cap - self.bytes_read
                portion = chunk[:remaining]
                self._count(portion)
                yield portion
                if self.bytes_read >= cap:
                    self.stop_reason = "exhausted" if total == self.bytes_read else "byte_limit"
                    return
            if total is not None and self.bytes_read < min(total, cap):
                raise ValueError("Truncated source response")
            self.stop_reason = (
                "exhausted" if total is None or self.bytes_read == total else "byte_limit"
            )

    def _count(self, chunk: bytes) -> None:
        self.bytes_read += len(chunk)
        self.digest.update(chunk)

    def records(self) -> Iterable[tuple[int, object, str | None]]:
        buffer = b""
        oversized = False
        line_number = 0
        with closing(self._chunks()) as chunks:
            for chunk in chunks:
                pieces = chunk.split(b"\n")
                for index, piece in enumerate(pieces):
                    if not oversized:
                        buffer += piece
                        if len(buffer) > self.limits["max_record_bytes"]:
                            oversized, buffer = True, b""
                    if index == len(pieces) - 1:
                        continue
                    line_number += 1
                    yield self._parse(line_number, buffer, oversized)
                    buffer, oversized = b"", False
            if (buffer or oversized) and self.stop_reason == "exhausted":
                yield self._parse(line_number + 1, buffer, oversized)

    @staticmethod
    def _parse(index: int, line: bytes, oversized: bool) -> tuple[int, object, str | None]:
        if oversized:
            return index, None, "record_too_large"
        try:
            # Reject non-finite JSON values; do not silently coerce source data.
            row = json.loads(line.decode("utf-8"), parse_constant=_invalid_constant)
            return index, row, None
        except (ValueError, RecursionError):
            return index, None, "malformed_json"

    def report(self) -> dict:
        return {
            "input_kind": "local" if self.local_path else "hugging_face",
            "bytes_read": self.bytes_read,
            "read_prefix_sha256": self.digest.hexdigest(),
            "stop_reason": self.stop_reason,
        }


def _invalid_constant(value: str):
    raise ValueError("Non-finite JSON number")


def _write_jsonl(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _distributions() -> dict:
    return {
        "turn_depth_distribution": Counter(),
        "character_length_distribution": Counter(),
        "approx_token_length_distribution": Counter(),
        "classification_counts": Counter({"adult": 0, "sfw": 0, "unknown": 0}),
    }


def _update_distributions(distribution: dict, metrics: dict, classification: str) -> None:
    distribution["turn_depth_distribution"][str(metrics["user_turns"])] += 1
    for field, value, bounds in (
        ("character_length_distribution", metrics["characters"], (1000, 4000, 16000, 64000)),
        ("approx_token_length_distribution", metrics["approx_tokens"], (256, 1024, 4096, 16384)),
    ):
        bucket = next((f"lt_{b}" for b in bounds if value < b), f"gte_{bounds[-1]}")
        distribution[field][bucket] += 1
    distribution["classification_counts"][classification] += 1


def protected_prompts() -> set[str]:
    """Read core SFT and held-out prompts only to prevent exact contamination."""
    prompts = set()
    paths = sorted((ROOT / "data/sft").glob("*.jsonl"))
    paths += sorted((ROOT / "data/sft/v1").glob("*.jsonl"))
    paths += sorted((ROOT / "eval").glob("*.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("prompt"), str):
                prompts.add(" ".join(row["prompt"].casefold().split()))
            for message in row.get("messages", []):
                if message.get("role") == "user":
                    prompts.add(" ".join(message_text(message).casefold().split()))
    return prompts


def build(
    config: dict,
    selected: list[str],
    output_dir: Path,
    local_inputs: dict[str, Path] | None = None,
    blocked_prompts: set[str] | None = None,
) -> dict:
    local_inputs = local_inputs or {}
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(name not in config["sources"] for name in selected)
    ):
        raise ValueError("Select unique configured source names")
    if set(local_inputs) - set(selected):
        raise ValueError("Local input was supplied for an unselected source")
    # Fail before output creation or networking on missing local input.
    if any(not path.is_file() for path in local_inputs.values()):
        raise ValueError("A local input file is missing")
    blocked_prompts = protected_prompts() if blocked_prompts is None else blocked_prompts
    output_dir.mkdir(parents=True, exist_ok=False)
    limits = config["smoke"]
    stats = {
        "pipeline_version": config["pipeline_version"],
        "status": "complete",
        "training_ready": False,
        "manifest_sha256": fingerprint(config),
        "protected_prompts_sha256": fingerprint(sorted(blocked_prompts)),
        "code_sha256": fingerprint(
            {
                name: hashlib.sha256((ROOT / "training" / name).read_bytes()).hexdigest()
                for name in ("rp_data.py", "prepare_rp_dataset.py", "data.py")
            }
        ),
        "limits": limits,
        "filters": config["filters"],
        "sources": {},
        "totals": Counter(),
        "rejection_counts": Counter(),
        "quarantine_counts": Counter(),
        **_distributions(),
        "normalized_distributions_by_status": {
            status: _distributions() for status in ("accepted", "rejected", "quarantined")
        },
        "distribution_population": "accepted_only",
        "classification_policy": "unknown unless independently reviewed; no keyword SFW inference",
        "review_candidates": 0,
    }
    seen, sample_counts = {}, Counter()
    with (
        (output_dir / "rp_sft.jsonl").open("w", encoding="utf-8", newline="\n") as accepted,
        (output_dir / "decisions.jsonl").open("w", encoding="utf-8", newline="\n") as decisions,
        (output_dir / "review_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as review,
        (output_dir / "review_candidates.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as candidates,
    ):
        # Manifest order defines duplicate precedence, regardless of CLI order.
        for name, source in config["sources"].items():
            if name not in selected:
                continue
            counts = Counter(
                {"processed": 0, "accepted": 0, "quarantined": 0, "rejected": 0, "duplicates": 0}
            )
            reader = BoundedInput(source, limits, local_inputs.get(name))
            source_report = {
                "manifest": source,
                "counts": counts,
                "error": None,
                "reasons": Counter(),
                "review_candidates": 0,
            }
            stats["sources"][name] = source_report
            try:
                with closing(reader.records()) as rows:
                    while counts["processed"] < limits["max_candidates_per_source"]:
                        try:
                            index, raw, error = next(rows)
                        except StopIteration:
                            break
                        counts["processed"] += 1
                        decision = {"source": name, "source_line": index}
                        record = None
                        try:
                            if error:
                                raise RowError(error)
                            record = normalize_conversation(raw, source, config)
                            assessment = assess(record, source, config["filters"])
                            status, reasons = assessment.status, list(assessment.reasons)
                            decision.update(
                                {
                                    "id": record["id"],
                                    "raw_sha256": fingerprint(raw),
                                    "metrics": record_metrics(record),
                                }
                            )
                            if any(
                                " ".join(message_text(m).casefold().split()) in blocked_prompts
                                for m in record["messages"]
                                if m["role"] == "user"
                            ):
                                status = "rejected"
                                reasons.append("core_or_eval_prompt_overlap")
                            if record["id"] in seen:
                                status = "rejected"
                                reasons.append("exact_duplicate")
                                decision["duplicate_of"] = seen[record["id"]]
                                counts["duplicates"] += 1
                            else:
                                seen[record["id"]] = {"source": name, "source_line": index}
                            source_id = raw.get("id")
                            record["metadata"] = {
                                "source": name,
                                "hf_repo": source["hf_repo"],
                                "revision": source["revision"],
                                "source_file": source["file"],
                                "input_kind": "local" if name in local_inputs else "hugging_face",
                                "source_id": str(source_id)[:256]
                                if isinstance(source_id, (str, int))
                                else None,
                                "source_line": index,
                                "raw_sha256": decision["raw_sha256"],
                                "source_character_id": raw.get("bot_id")
                                if isinstance(raw.get("bot_id"), str)
                                else None,
                                "classification": assessment.classification,
                                "classification_basis": assessment.classification_basis,
                                "metrics": decision["metrics"],
                                "human_review": "pending",
                                "primary_rules_status": "coverage_targets_not_verified",
                                "provenance_status": source["provenance_status"],
                                "declared_license": source["declared_license"],
                                "filter_version": config["pipeline_version"],
                            }
                        except RowError as exc:
                            status, reasons = "rejected", [str(exc)]
                        reasons = sorted(set(reasons))
                        decision.update({"status": status, "reasons": reasons})
                        source_report["reasons"].update(reasons)
                        counts[status] += 1
                        stats["totals"][status] += 1
                        if record is not None and "metadata" in record:
                            _update_distributions(
                                stats["normalized_distributions_by_status"][status],
                                decision["metrics"],
                                record["metadata"]["classification"],
                            )
                        if status != "accepted":
                            key = (
                                "rejection_counts" if status == "rejected" else "quarantine_counts"
                            )
                            stats[key].update(reasons)
                        if status == "accepted":
                            _write_jsonl(accepted, record)
                            _update_distributions(
                                stats, decision["metrics"], record["metadata"]["classification"]
                            )
                        elif status == "quarantined" and reasons == ["source_provenance_review"]:
                            # Retain text only for mechanically eligible provenance candidates.
                            # Safety/PII quarantine and rejection logs contain locators, not text.
                            _write_jsonl(candidates, record)
                            stats["review_candidates"] += 1
                            source_report["review_candidates"] += 1
                        _write_jsonl(decisions, decision)
                        for reason in reasons or ["human_review_pending"]:
                            bucket = (name, status, reason)
                            if sample_counts[bucket] < limits["review_samples_per_bucket"]:
                                _write_jsonl(
                                    review,
                                    {
                                        **decision,
                                        "review_reason": reason,
                                        "human_review": "pending",
                                    },
                                )
                                sample_counts[bucket] += 1
                        if counts["accepted"] >= limits["accepted_per_source"]:
                            reader.stop_reason = "accepted_target"
                            break
                    else:
                        reader.stop_reason = "candidate_limit"
            except (httpx.HTTPError, OSError, ValueError, RecursionError) as exc:
                # Never serialize exception text: HTTP errors may contain signed URLs.
                source_report["error"] = type(exc).__name__
                reader.stop_reason = "source_error"
                stats["status"] = "partial_failure"
            source_report["input"] = reader.report()
            source_report["target_met"] = counts["accepted"] >= limits["accepted_per_source"]
    for name in ("processed", "duplicates"):
        stats["totals"][name] = sum(s["counts"][name] for s in stats["sources"].values())
    for name in ("accepted", "rejected", "quarantined"):
        stats["totals"].setdefault(name, 0)
    stats["output_sha256"] = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in (
            "rp_sft.jsonl",
            "decisions.jsonl",
            "review_manifest.jsonl",
            "review_candidates.jsonl",
        )
    }
    temporary = output_dir / "stats.json.tmp"
    temporary.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "stats.json")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source", action="append", help="Explicitly inspect a source, even if disabled"
    )
    parser.add_argument("--local-input", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument(
        "--offline", action="store_true", help="Require local input for every source"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Fresh directory under RP generated/"
    )
    parser.add_argument("--accepted-per-source", type=int)
    parser.add_argument("--max-candidates-per-source", type=int)
    parser.add_argument("--max-bytes-per-source", type=int)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        for key, maximum in (
            ("accepted_per_source", 100),
            ("max_candidates_per_source", 5000),
            ("max_bytes_per_source", 16 * 1024 * 1024),
        ):
            value = getattr(args, key)
            if value is not None:
                if not 1 <= value <= maximum:
                    raise ValueError(f"Invalid smoke override: {key}")
                config["smoke"][key] = value
        selected = args.source or [
            k for k, v in config["sources"].items() if v["enabled_by_default"]
        ]
        local = {}
        for item in args.local_input:
            name, separator, path = item.partition("=")
            if not separator or name in local or not path:
                raise ValueError("Local input must use a unique SOURCE=PATH")
            local[name] = Path(path)
        if args.offline and set(selected) != set(local):
            raise ValueError("Offline mode requires local input for every selected source")
        destination = args.output_dir.resolve()
        # ROOT is already resolved. Compare against the intended lexical output
        # namespace so a symlink/junction cannot redirect generated/ into core SFT.
        if not destination.is_relative_to(OUTPUT_ROOT) or destination == OUTPUT_ROOT:
            raise ValueError("Output must be a fresh subdirectory of data/sft/rp_v1/generated")
        stats = build(config, selected, destination, local)
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        # Configuration errors are safe to identify by type without dumping inputs.
        parser.exit(
            2, f"RP build failed ({type(exc).__name__}); check config, paths, and limits.\n"
        )
    print(
        json.dumps(
            {
                "status": stats["status"],
                "totals": stats["totals"],
                "review_candidates": stats["review_candidates"],
                "training_ready": False,
            },
            sort_keys=True,
        )
    )
    if stats["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
