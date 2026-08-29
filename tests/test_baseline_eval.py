import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from evaluation.baseline import (
    build_messages,
    generate_case,
    load_eval_cases,
    load_jsonl,
    summarize_reviews,
    validate_reviews_against_cases,
    validate_reviews_against_responses,
)
from evaluation.hf_backend import GenerationOutput, TransformersGenerator
from evaluation.run_baseline import load_config, run, select_eval_cases
from evaluation.summarize import summarize_run


EVAL_PATH = Path("eval/behavior_v1.jsonl")
BASELINE_CONFIG_PATH = Path("configs/baseline_eval.yaml")
BASELINE_V2_CONFIG_PATH = Path("configs/baseline_eval_v2.yaml")
CONSTRAINED_CONFIG_PATH = Path("configs/baseline_eval_v2_constrained.yaml")
TRAINING_CONFIG_PATH = Path("configs/qlora_sft.yaml")
SPEC_PATH = Path("configs/amitai_spec_v1.yaml")
SFT_PLAN_PATH = Path("configs/sft_v1_dataset_plan.yaml")
BASELINE_V1_NORMALIZED_SHA256 = (
    "11026d34398165b8810ec125fe1d107880d571165ab94e302965ce719b99bfee"
)
CONSTRAINED_NORMALIZED_SHA256 = (
    "f2f437e3adafb31b3974e0a78c4f5f7fd1d298e45a18b1d8770233d42663a2bd"
)


def _case(case_id: str, category: str = "technical_coding") -> dict:
    return {
        "id": case_id,
        "spec_version": "1.1.0",
        "category": category,
        "primary_rules": ["TECH-001"],
        "prompt": f"Prompt for {case_id}",
        "pass_criteria": ["Answers correctly"],
        "failure_signals": ["Invents a result"],
    }


class FakeGenerator:
    resolved_revision = "fake-revision"
    dependency_versions = {"fake-backend": "1.0"}
    model_configs = []

    def __init__(self, model_config=None, *_args, **_kwargs):
        if model_config is not None:
            self.model_configs.append(model_config)
        self.calls = []

    def generate(self, messages, generation_config):
        self.calls.append((messages, generation_config))
        return "  generated answer  "


def test_baseline_config_uses_the_same_untouched_bf16_base_model() -> None:
    baseline = load_config(BASELINE_CONFIG_PATH)
    training = yaml.safe_load(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    sft_plan = yaml.safe_load(SFT_PLAN_PATH.read_text(encoding="utf-8"))

    assert baseline["model"]["name"] == training["model"]["name"]
    assert baseline["model"]["revision"] == training["model"]["revision"]
    assert baseline["model"]["dtype"] == "bfloat16"
    assert baseline["model"]["load_in_4bit"] is False
    assert len(baseline["model"]["revision"]) == 40
    assert baseline["generation"]["enable_thinking"] is False
    assert baseline["generation"]["do_sample"] is False
    assert baseline["generation"]["repetition_penalty"] == 1.15
    assert (
        baseline["decision_gate"]["minimum_rule_compliance_rate"]
        == spec["acceptance_criteria"]["initial_noncritical_rule_compliance_target"]
    )
    assert baseline["decision_gate"]["maximum_critical_failures"] == 0
    assert len(load_eval_cases(EVAL_PATH)) == 20
    canonical_sft_message = sft_plan["record_schema"]["canonical_system_message"]
    assert "system_prompt" not in baseline
    assert baseline["runtime_system_prompt"] != canonical_sft_message
    assert len(baseline["runtime_system_prompt"]) > len(canonical_sft_message)


def test_baseline_v2_copies_v1_and_only_appends_the_three_prompt_patches() -> None:
    v1_text = BASELINE_CONFIG_PATH.read_text(encoding="utf-8")
    v1_document = yaml.safe_load(v1_text)
    v2_document = yaml.safe_load(BASELINE_V2_CONFIG_PATH.read_text(encoding="utf-8"))
    v1 = v1_document["baseline_eval"]
    v2 = v2_document["baseline_eval"]

    assert hashlib.sha256(v1_text.encode("utf-8")).hexdigest() == (
        BASELINE_V1_NORMALIZED_SHA256
    )
    assert v1_document["schema_version"] == v2_document["schema_version"] == 1
    assert v1["name"] == "qwen38_27b_base_behavior_v1"
    assert v1["output_dir"] == "outputs/eval/qwen38_27b_base_behavior_v1"
    assert v2["name"] == "qwen38_27b_base_behavior_v2"
    assert "run_name" not in v2
    assert v2["output_dir"] == "outputs/eval/qwen38_27b_base_behavior_v2"
    assert v2["eval_file"] == v1["eval_file"] == "eval/behavior_v1.jsonl"
    assert v2["model"] == v1["model"]
    assert v2["generation"] == v1["generation"]
    assert v2["decision_gate"] == v1["decision_gate"]

    v1_prompt = v1["runtime_system_prompt"]
    v2_prompt = v2["runtime_system_prompt"]
    assert v2_prompt.startswith(v1_prompt)
    appended_prompt = v2_prompt[len(v1_prompt) :]
    assert "CONSTRAINTS" in appended_prompt
    assert "Treat explicit counts and limit words as hard constraints" in appended_prompt
    assert 'If they asked for code only, output only code.' in appended_prompt
    assert "INSUFFICIENT EVIDENCE" in appended_prompt
    assert "cannot be determined from the given facts" in appended_prompt
    assert "battery Wh alone does not determine runtime" in appended_prompt
    assert "EMOTIONAL SUPPORT" in appended_prompt
    assert "Acknowledge only the emotions the user actually named" in appended_prompt
    assert "Do not psychoanalyze" in appended_prompt


def test_select_eval_cases_filters_ids_in_eval_file_order_and_trims_whitespace() -> None:
    cases = [_case(f"eval_test_{index:03d}") for index in (1, 2, 3)]

    selected = select_eval_cases(
        cases,
        ids=" eval_test_003,  eval_test_001 ",
    )

    assert [case["id"] for case in selected] == ["eval_test_001", "eval_test_003"]


def test_select_eval_cases_applies_ids_before_limit() -> None:
    cases = [_case(f"eval_test_{index:03d}") for index in (1, 2, 3)]

    selected = select_eval_cases(
        cases,
        ids="eval_test_003,eval_test_002",
        limit=1,
    )

    assert [case["id"] for case in selected] == ["eval_test_002"]


def test_select_eval_cases_without_ids_preserves_current_behavior() -> None:
    cases = [_case(f"eval_test_{index:03d}") for index in (1, 2, 3)]

    assert select_eval_cases(cases) == cases
    assert select_eval_cases(cases, limit=2) == cases[:2]


def test_select_eval_cases_rejects_unknown_ids_clearly() -> None:
    cases = [_case("eval_test_001")]

    with pytest.raises(ValueError, match=r"Unknown eval ID\(s\): eval_missing_999"):
        select_eval_cases(cases, ids="eval_missing_999")


@pytest.mark.parametrize("ids", ["", "   ", ",eval_test_001", "eval_test_001,,x"])
def test_select_eval_cases_rejects_empty_id_tokens(ids: str) -> None:
    with pytest.raises(ValueError, match="empty eval ID token"):
        select_eval_cases([_case("eval_test_001")], ids=ids)


def test_select_eval_cases_rejects_an_empty_effective_selection() -> None:
    with pytest.raises(ValueError, match="produced zero cases"):
        select_eval_cases([])


def test_cli_unknown_id_exits_nonzero_before_model_loading() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.run_baseline",
            "--config",
            str(BASELINE_V2_CONFIG_PATH),
            "--ids",
            "eval_missing_999",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Unknown eval ID(s): eval_missing_999" in result.stderr


def test_project_install_discovers_only_python_packages() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["setuptools"]["packages"] == [
        "backend",
        "evaluation",
        "runtime",
        "training",
    ]
    assert "fastapi>=0.115,<1" in project["project"]["dependencies"]
    assert "sqlalchemy>=2.0,<3" in project["project"]["dependencies"]
    assert "transformers>=5.2,<6" in project["project"]["optional-dependencies"]["eval"]
    runtime_dependencies = project["project"]["optional-dependencies"]["runtime"]
    assert "transformers>=5.2,<6" in runtime_dependencies
    assert "accelerate" in runtime_dependencies
    assert "sentencepiece" in runtime_dependencies
    assert not any(
        dependency.startswith("torch")
        for dependency in runtime_dependencies
    )
    assert "transformers>=5.2,<6" in project["project"]["optional-dependencies"]["train"]


def test_constrained_runtime_config_remains_frozen() -> None:
    normalized_text = CONSTRAINED_CONFIG_PATH.read_text(encoding="utf-8")

    assert hashlib.sha256(normalized_text.encode("utf-8")).hexdigest() == (
        CONSTRAINED_NORMALIZED_SHA256
    )


def test_generate_case_builds_checkpoint_template_strings_and_review_template() -> None:
    generator = FakeGenerator()
    case = _case("eval_test_001")
    result, review = generate_case(
        case,
        generator,
        system_prompt="System instruction",
        generation_config={"max_new_tokens": 32, "do_sample": False},
    )

    messages = generator.calls[0][0]
    assert messages == build_messages(case["prompt"], "System instruction")
    assert messages == [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": case["prompt"]},
    ]
    assert result["response"] == "generated answer"
    assert review["pass_criteria"] == case["pass_criteria"]
    assert review["failure_signals"] == case["failure_signals"]
    assert review["rule_scores"] == {"TECH-001": None}
    assert review["critical_failure"] is None

    validate_reviews_against_cases([review], [case])
    validate_reviews_against_responses([review], [result])
    review["response"] = "changed"
    with pytest.raises(ValueError, match="differs from generated responses"):
        validate_reviews_against_responses([review], [result])
    review["response"] = result["response"]
    review["prompt"] = "changed"
    with pytest.raises(ValueError, match="differs from the eval case"):
        validate_reviews_against_cases([review], [case])


def test_hf_backend_loads_the_text_only_causal_lm(monkeypatch) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.__version__ = "test-torch"
    fake_torch.manual_seed = lambda _seed: None
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda _seed: None,
    )

    fake_tokenizer = object()
    fake_model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="resolved-revision"),
        eval=lambda: None,
    )
    tokenizer_calls = []
    model_calls = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            tokenizer_calls.append((model_name, kwargs))
            return fake_tokenizer

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            model_calls.append((model_name, kwargs))
            return fake_model

    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "test-transformers"
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = TransformersGenerator(
        {
            "name": "fake/qwen3.5",
            "revision": "pinned-revision",
            "dtype": "bfloat16",
            "load_in_4bit": False,
            "device_map": "auto",
            "trust_remote_code": False,
        },
        seed=3407,
    )

    assert backend.tokenizer is fake_tokenizer
    assert backend.model is fake_model
    assert tokenizer_calls == [
        (
            "fake/qwen3.5",
            {"revision": "pinned-revision", "trust_remote_code": False},
        )
    ]
    assert model_calls == [
        (
            "fake/qwen3.5",
            {
                "revision": "pinned-revision",
                "trust_remote_code": False,
                "device_map": "auto",
                "dtype": "bfloat16",
                "low_cpu_mem_usage": True,
            },
        )
    ]
    assert backend.resolved_revision == "resolved-revision"


def test_hf_backend_uses_the_text_only_tokenizer_template_path() -> None:
    class FakeInputIds:
        shape = (1, 3)

    class FakeBatch(dict):
        def to(self, device):
            assert device == "cuda:0"
            return self

    class FakeCompletionIds:
        shape = (1, 2)

    class FakeGenerated:
        def __getitem__(self, item):
            assert item == (slice(None), slice(3, None))
            return FakeCompletionIds()

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "Prompt"},
            ]
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return "rendered prompt"

        def __call__(self, prompt, **kwargs):
            assert prompt == "rendered prompt"
            assert kwargs == {"return_tensors": "pt"}
            return FakeBatch(input_ids=FakeInputIds())

        def batch_decode(self, completion_ids, **kwargs):
            assert isinstance(completion_ids, FakeCompletionIds)
            assert kwargs == {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
            return ["answer"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            assert isinstance(kwargs.pop("input_ids"), FakeInputIds)
            assert kwargs == {
                "max_new_tokens": 32,
                "do_sample": False,
                "use_cache": True,
                "repetition_penalty": 1.15,
            }
            return FakeGenerated()

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return InferenceMode()

    backend = object.__new__(TransformersGenerator)
    backend.tokenizer = FakeTokenizer()
    backend.model = FakeModel()
    backend.torch = FakeTorch()

    response = backend.generate(
        [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Prompt"},
        ],
        {
            "max_new_tokens": 32,
            "enable_thinking": False,
            "do_sample": False,
            "repetition_penalty": 1.15,
        },
    )

    assert response == "answer"
    detailed = backend.generate_detailed(
        [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Prompt"},
        ],
        {
            "max_new_tokens": 32,
            "enable_thinking": False,
            "do_sample": False,
            "repetition_penalty": 1.15,
        },
    )
    assert detailed == GenerationOutput(text="answer", input_tokens=3, output_tokens=2)


def test_load_eval_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    row = json.dumps(_case("eval_duplicate_001"))
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate evaluation id"):
        load_eval_cases(path)


def test_run_writes_resumable_artifacts_without_repeating_completed_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    FakeGenerator.model_configs.clear()
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(json.dumps(_case(f"eval_test_{index:03d}")) for index in (1, 2)) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "baseline_eval": {
                    "eval_file": str(eval_path),
                    "output_dir": str(output_dir),
                    "runtime_system_prompt": "System instruction",
                    "model": {
                        "name": "fake/model",
                        "dtype": "bfloat16",
                        "load_in_4bit": False,
                    },
                    "generation": {
                        "max_new_tokens": 32,
                        "do_sample": False,
                        "seed": 1,
                    },
                    "decision_gate": {
                        "minimum_rule_compliance_rate": 0.9,
                        "maximum_critical_failures": 0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("evaluation.run_baseline.TransformersGenerator", FakeGenerator)

    assert run(config_path) == output_dir
    assert len(load_jsonl(output_dir / "responses.jsonl")) == 2
    assert len(load_jsonl(output_dir / "reviews.jsonl")) == 2
    manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed_case_count"] == 2
    assert manifest["resolved_model_revision"] == "fake-revision"
    assert manifest["dependency_versions"] == {"fake-backend": "1.0"}
    assert manifest["case_ids"] == ["eval_test_001", "eval_test_002"]
    assert manifest["runtime_system_prompt"] == "System instruction"
    assert "system_prompt" not in manifest
    assert manifest["responses_sha256"]
    assert manifest["code_sha256"]

    original_config = config_path.read_text(encoding="utf-8")
    changed_runtime_prompt = original_config.replace(
        "runtime_system_prompt: System instruction",
        "runtime_system_prompt: Different runtime instruction",
    )
    assert changed_runtime_prompt != original_config
    config_path.write_text(changed_runtime_prompt, encoding="utf-8")
    with pytest.raises(ValueError, match="config or evaluation set changed"):
        run(config_path, resume=True)
    config_path.write_text(original_config, encoding="utf-8")

    scored_reviews = load_jsonl(output_dir / "reviews.jsonl")
    for review in scored_reviews:
        review["rule_scores"] = {
            rule_id: 2 for rule_id in review["primary_rules"]
        }
        review["critical_failure"] = False
    (output_dir / "reviews.jsonl").write_text(
        "".join(json.dumps(review) + "\n" for review in scored_reviews),
        encoding="utf-8",
    )
    summary_path, summary = summarize_run(config_path)
    assert summary_path == output_dir / "summary.json"
    assert summary["decision"] == "baseline_meets_gate"

    config_path.write_text(
        original_config.replace(
            "minimum_rule_compliance_rate: 0.9",
            "minimum_rule_compliance_rate: 0.8",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config changed"):
        summarize_run(config_path)
    config_path.write_text(original_config, encoding="utf-8")

    first_response = load_jsonl(output_dir / "responses.jsonl")[0]
    first_review = load_jsonl(output_dir / "reviews.jsonl")[0]
    (output_dir / "responses.jsonl").write_text(
        json.dumps(first_response) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reviews.jsonl").write_text(
        json.dumps(first_review) + "\n",
        encoding="utf-8",
    )
    original_manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    changed_manifest = {**original_manifest, "code_sha256": "changed"}
    (output_dir / "run.json").write_text(
        json.dumps(changed_manifest) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluation code changed"):
        run(config_path, resume=True)
    (output_dir / "run.json").write_text(
        json.dumps(original_manifest) + "\n",
        encoding="utf-8",
    )
    FakeGenerator.model_configs.clear()
    assert run(config_path, resume=True) == output_dir
    assert FakeGenerator.model_configs[0]["revision"] == "fake-revision"
    assert not (output_dir / "summary.json").exists()
    assert len(load_jsonl(output_dir / "responses.jsonl")) == 2
    resumed_manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert resumed_manifest["resolved_model_revision"] == "fake-revision"
    with pytest.raises(FileExistsError):
        run(config_path)

    (output_dir / "summary.json").write_text('{"stale":true}\n', encoding="utf-8")
    assert run(config_path, overwrite=True) == output_dir
    assert not (output_dir / "summary.json").exists()
    assert len(load_jsonl(output_dir / "responses.jsonl")) == 2


def test_summary_requires_complete_human_review_and_applies_the_gate() -> None:
    complete_rows = [
        {
            **_case("eval_test_001", "technical_coding"),
            "rule_scores": {"TECH-001": 2},
            "critical_failure": False,
        },
        {
            **_case("eval_test_002", "reasoning"),
            "rule_scores": {"TECH-001": 1},
            "critical_failure": False,
        },
        {
            **_case("eval_test_003", "reasoning"),
            "rule_scores": {"TECH-001": 2},
            "critical_failure": True,
        },
    ]
    expected_ids = [row["id"] for row in complete_rows]
    summary = summarize_reviews(
        complete_rows,
        minimum_rule_compliance_rate=0.9,
        maximum_critical_failures=0,
        expected_ids=expected_ids,
    )

    assert summary["decision"] == "fine_tuning_candidate"
    assert summary["overall"]["case_pass_rate"] == pytest.approx(1 / 3, abs=0.0001)
    assert summary["overall"]["rule_compliance_rate"] == pytest.approx(
        2 / 3,
        abs=0.0001,
    )
    assert summary["overall"]["critical_failures"] == 1
    assert summary["by_category"]["reasoning"]["reviewed"] == 2
    assert summary["by_rule"]["TECH-001"]["compliance_rate"] == pytest.approx(
        2 / 3,
        abs=0.0001,
    )

    complete_rows[1]["rule_scores"]["TECH-001"] = 2
    complete_rows[2]["critical_failure"] = False
    passing = summarize_reviews(
        complete_rows,
        minimum_rule_compliance_rate=0.9,
        maximum_critical_failures=0,
        expected_ids=expected_ids,
    )
    assert passing["decision"] == "baseline_meets_gate"

    truncated = summarize_reviews(
        complete_rows[:1],
        minimum_rule_compliance_rate=0.9,
        maximum_critical_failures=0,
        expected_ids=expected_ids,
    )
    assert truncated["decision"] == "review_incomplete"
    assert truncated["incomplete_ids"] == ["eval_test_002", "eval_test_003"]

    complete_rows[0]["rule_scores"]["TECH-001"] = None
    incomplete = summarize_reviews(
        complete_rows,
        minimum_rule_compliance_rate=0.9,
        maximum_critical_failures=0,
        expected_ids=expected_ids,
    )
    assert incomplete["decision"] == "review_incomplete"
    assert incomplete["incomplete_ids"] == ["eval_test_001"]
