from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.data import load_sft_dataset


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    save_cfg = cfg["save"]

    # Import here so dataset/config validation can run on machines without CUDA/Unsloth.
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    dataset = load_sft_dataset(train_cfg["dataset"])
    max_length = int(train_cfg.get("max_length", 4096))

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_cfg["name"],
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        max_seq_length=max_length,
        full_finetuning=False,
        use_gradient_checkpointing=model_cfg.get("gradient_checkpointing", "unsloth"),
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(lora_cfg.get("finetune_vision_layers", False)),
        finetune_language_layers=bool(lora_cfg.get("finetune_language_layers", True)),
        finetune_attention_modules=bool(lora_cfg.get("finetune_attention_modules", True)),
        finetune_mlp_modules=bool(lora_cfg.get("finetune_mlp_modules", True)),
        r=int(lora_cfg.get("rank", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=lora_cfg.get("bias", "none"),
        random_state=int(train_cfg.get("seed", 3407)),
        use_rslora=bool(lora_cfg.get("use_rslora", False)),
        loftq_config=None,
    )

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=train_cfg["output_dir"],
            per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
            learning_rate=float(train_cfg["learning_rate"]),
            num_train_epochs=float(train_cfg["num_train_epochs"]),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
            logging_steps=int(train_cfg.get("logging_steps", 1)),
            save_steps=int(train_cfg.get("save_steps", 50)),
            optim=train_cfg.get("optim", "adamw_8bit"),
            seed=int(train_cfg.get("seed", 3407)),
            report_to=train_cfg.get("report_to", "none"),
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_length,
        ),
    )

    stats = trainer.train()
    print(stats)

    adapter_dir = Path(save_cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to: {adapter_dir}")

    if bool(save_cfg.get("save_merged_16bit", False)):
        merged_dir = adapter_dir.parent / "merged-16bit"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"Saved merged 16-bit model to: {merged_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AmitAI BF16 LoRA adapter")
    parser.add_argument(
        "--config",
        default="configs/qlora_sft.yaml",
        help="Path to YAML training config",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
