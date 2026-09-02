"""Validated model settings with separate production and frozen evaluation prompts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from string import Template
from typing import Any

import yaml

DEFAULT_RUNTIME_CONFIG_PATH = Path("configs/baseline_eval_v2_constrained.yaml")
DEFAULT_PRODUCTION_PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs/production_runtime.yaml"
EXPECTED_MODEL_NAME = "OBLITERATUS/Qwen3.8-27B-OBLITERATED"
EXPECTED_MODEL_REVISION = "a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8"


@dataclass(frozen=True)
class RuntimeConfig:
    source_path: Path
    runtime_system_prompt: str
    model: dict[str, Any]
    generation: dict[str, Any]
    mechanical_constraints_enabled: bool


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Runtime config {field} must be an object")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Runtime config {field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Runtime config {field} must be an integer >= {minimum}")
    return value


def load_runtime_config(path: str | Path = DEFAULT_RUNTIME_CONFIG_PATH) -> RuntimeConfig:
    """Load the literal config, including its original prompt, without overrides."""
    source_path = Path(path)
    try:
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read runtime config: {source_path}") from exc
    if not isinstance(document, Mapping):
        raise TypeError("Runtime config must be a YAML object")

    baseline = _mapping(document.get("baseline_eval"), "baseline_eval")
    runtime_system_prompt = _nonempty_string(
        baseline.get("runtime_system_prompt"),
        "baseline_eval.runtime_system_prompt",
    )
    model = dict(_mapping(baseline.get("model"), "baseline_eval.model"))
    generation = dict(_mapping(baseline.get("generation"), "baseline_eval.generation"))
    mechanical = _mapping(
        baseline.get("mechanical_constraints"),
        "baseline_eval.mechanical_constraints",
    )

    if model.get("name") != EXPECTED_MODEL_NAME:
        raise ValueError(f"Runtime model must remain {EXPECTED_MODEL_NAME}")
    if model.get("revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("Runtime model revision must remain pinned to the tested commit")
    if model.get("dtype") != "bfloat16":
        raise ValueError("Runtime model dtype must remain bfloat16")
    if model.get("load_in_4bit") is not False:
        raise ValueError("Runtime model must use BF16 loading, not 4-bit loading")
    if model.get("device_map") != "auto":
        raise ValueError("Runtime model device_map must remain auto")
    if model.get("trust_remote_code") is not False:
        raise ValueError("Runtime model trust_remote_code must remain false")

    max_new_tokens = _integer(
        generation.get("max_new_tokens"),
        "baseline_eval.generation.max_new_tokens",
        minimum=1,
    )
    seed = _integer(generation.get("seed"), "baseline_eval.generation.seed")
    if max_new_tokens != 512:
        raise ValueError("Runtime max_new_tokens must remain 512")
    if seed != 3407:
        raise ValueError("Runtime generation seed must remain 3407")
    if generation.get("enable_thinking") is not False:
        raise ValueError("Runtime generation must explicitly disable thinking")
    if generation.get("do_sample") is not False:
        raise ValueError("Runtime generation must remain deterministic")
    repetition_penalty = generation.get("repetition_penalty")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or repetition_penalty <= 0
    ):
        raise ValueError(
            "Runtime config baseline_eval.generation.repetition_penalty must be positive"
        )
    if float(repetition_penalty) != 1.15:
        raise ValueError("Runtime repetition_penalty must remain 1.15")
    if mechanical.get("enabled") is not True:
        raise ValueError("Runtime mechanical constraint validation must be enabled")

    return RuntimeConfig(
        source_path=source_path,
        runtime_system_prompt=runtime_system_prompt,
        model=model,
        generation=generation,
        mechanical_constraints_enabled=True,
    )


def load_production_runtime_config(
    path: str | Path = DEFAULT_RUNTIME_CONFIG_PATH,
    *,
    profile_path: str | Path = DEFAULT_PRODUCTION_PROFILE_PATH,
) -> RuntimeConfig:
    """Compose prompt-only production identity with the unchanged validated settings.

    source_path still identifies the model/generation settings file. The profile
    cannot override model, sampling or validator settings, and template values
    come only from the validated config, never from the process environment.
    """
    config = load_runtime_config(path)
    try:
        document = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("Unable to read production runtime profile") from exc
    profile = _mapping(document, "production profile")
    if set(profile) != {"schema_version", "runtime_system_prompt"}:
        raise ValueError("Production runtime profile must contain only version and prompt")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("Unsupported production runtime profile version")
    template = Template(_nonempty_string(profile["runtime_system_prompt"], "production prompt"))
    if not template.is_valid() or template.get_identifiers() != ["model_name"]:
        raise ValueError("Production runtime prompt must reference only the configured model_name")
    return replace(
        config,
        runtime_system_prompt=template.substitute(model_name=config.model["name"]),
    )
