"""CPU-only pinned-template captures of synthetic V4 inputs; no tokenizer or model."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.aevon_text_quality import DEFAULT_CASES, generation_messages, load_cases
from evaluation.context_layouts import LAYOUTS, OMISSION_SECTION, layout_messages
from runtime.calculator import CalculatorTool
from runtime.config import load_production_runtime_config
from runtime.context import compile_model_messages
from runtime.privacy import InferenceExecutionScope
from runtime.tooling import ToolRegistry

# Exact metadata-only template verified for the pinned revision during V3.
# No from_pretrained, Transformers import, tokenizer download or GPU operation.
PINNED_TEMPLATE = """{%- for message in messages %}
{%- if message.role == "system" %}
<|im_start|>system
{{ message.content }}<|im_end|>
{%- elif message.role == "user" %}
<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == "assistant" %}
<|im_start|>assistant
{{ message.content }}<|im_end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- if enable_thinking is defined and enable_thinking is true %}
<think>
{%- else %}
<think>

</think>

{%- endif %}
{%- endif %}
"""
V4_CASES = DEFAULT_CASES.with_name("aevon_epistemic_regression_v4.jsonl")
CAPTURE_IDS = ("v4_memory_conflict", "v4_missing_opening")


def render_prompt(messages):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    return ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        PINNED_TEMPLATE,
    ).render(messages=messages, add_generation_prompt=True, enable_thinking=False)


def capture_prompts():
    config = load_production_runtime_config()
    captures, measures = {}, {}
    for case in load_cases(V4_CASES):
        if case.id not in CAPTURE_IDS:
            continue
        source = generation_messages(case)
        canonical = compile_model_messages(
            source, runtime_system_prompt=config.runtime_system_prompt,
            tool_instructions=ToolRegistry([CalculatorTool()]).instructions(),
            execution_scope=InferenceExecutionScope.LOCAL,
        )
        memory = source[0].content if case.memory else None
        for layout in LAYOUTS:
            messages = layout_messages(canonical, layout)
            prompt = render_prompt(messages)
            name = f"{case.id}_{layout}"
            captures[name] = prompt
            # Character distances, NOT token counts. Current-user frame begins
            # after any immediately preceding trusted notice.
            current = prompt.rindex("<|im_start|>user\n")
            section = memory or OMISSION_SECTION
            section_end = prompt.index(section) + len(section)
            measures[name] = {
                "roles": [item["role"] for item in messages],
                "prompt_chars": len(prompt),
                "section_end_to_current_user_chars": current - section_end,
                "section": "memory" if memory else "omission_notice",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
    return captures, {
        "model": config.model["name"], "revision": config.model["revision"],
        "template_sha256": hashlib.sha256(PINNED_TEMPLATE.encode("utf-8")).hexdigest(),
        "distance_unit": "Unicode characters, not tokens",
        "captures": measures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    captures, manifest = capture_prompts()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, prompt in captures.items():
        (args.output_dir / f"{name}.txt").write_bytes(prompt.encode("utf-8"))
    (args.output_dir / "comparison.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Wrote {len(captures)} synthetic prompt captures to {args.output_dir}")


if __name__ == "__main__":
    main()
