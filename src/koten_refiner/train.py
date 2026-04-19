from __future__ import annotations

from pathlib import Path

import orjson
import yaml


def load_yaml_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def train_with_unsloth(dataset_path: Path, output_dir: Path, config: dict) -> None:
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"].get("load_in_4bit", False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["train"]["seed"],
    )

    def format_row(row: dict) -> dict:
        text = f"{row['prompt']}\n\n入力:\n{row['input_text']}\n\n出力:\n{row['target_text']}"
        return {"text": text}

    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=config["train"]["per_device_batch_size"],
            gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
            learning_rate=config["train"]["learning_rate"],
            num_train_epochs=config["train"]["epochs"],
            logging_steps=config["train"]["logging_steps"],
            save_strategy="epoch",
            bf16=True,
            seed=config["train"]["seed"],
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "train_config.json").write_bytes(orjson.dumps(config, option=orjson.OPT_INDENT_2))
