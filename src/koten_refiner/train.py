from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError
import os
from pathlib import Path
import re
import warnings

import orjson
import yaml


def load_yaml_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def append_eos_token(text: str, eos_token: str | None) -> str:
    if not eos_token or text.endswith(eos_token):
        return text
    return f"{text}{eos_token}"


def format_sft_training_text(
    prompt: str,
    input_text: str,
    target_text: str,
    eos_token: str | None,
) -> str:
    body = f"{prompt}\n\n入力:\n{input_text}\n\n出力:\n{target_text}"
    return append_eos_token(body, eos_token)


def _wandb_config(config: dict) -> dict[str, object]:
    raw = config.get("wandb")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("wandb config must be a mapping")
    return raw


def wandb_enabled(config: dict) -> bool:
    enabled = _wandb_config(config).get("enabled", False)
    return bool(enabled)


def build_report_to(config: dict) -> list[str]:
    if wandb_enabled(config):
        return ["wandb"]
    return []


def _normalize_run_name_label(label: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", label.strip()).strip("_")
    return normalized or "train"


def build_default_run_name(config: dict, output_dir: Path, now: datetime | None = None) -> str:
    current = now or datetime.now()
    wandb_config = _wandb_config(config)
    label = wandb_config.get("group")
    if not isinstance(label, str) or not label:
        label = output_dir.name
    safe_label = _normalize_run_name_label(label)
    return f"{current.strftime('%Y%m%d_%H%M%S')}_{safe_label}"


def resolve_run_name(config: dict, output_dir: Path) -> str | None:
    if not wandb_enabled(config):
        return None
    run_name = _wandb_config(config).get("run_name")
    if isinstance(run_name, str) and run_name:
        return run_name
    return build_default_run_name(config, output_dir)


def build_wandb_env(config: dict, output_dir: Path) -> dict[str, str]:
    if not wandb_enabled(config):
        return {}

    wandb_config = _wandb_config(config)
    env = {"WANDB_DIR": str(output_dir / "wandb")}
    key_map = {
        "project": "WANDB_PROJECT",
        "entity": "WANDB_ENTITY",
        "group": "WANDB_RUN_GROUP",
        "job_type": "WANDB_JOB_TYPE",
        "mode": "WANDB_MODE",
    }
    for config_key, env_key in key_map.items():
        value = wandb_config.get(config_key)
        if isinstance(value, str) and value:
            env[env_key] = value

    tags = wandb_config.get("tags")
    if isinstance(tags, list):
        string_tags = [tag for tag in tags if isinstance(tag, str) and tag]
        if string_tags:
            env["WANDB_TAGS"] = ",".join(string_tags)
    return env


def apply_wandb_environment(config: dict, output_dir: Path) -> None:
    for key, value in build_wandb_env(config, output_dir).items():
        os.environ[key] = value


def _make_model_card_generation_best_effort(trainer: object) -> None:
    original_create_model_card = getattr(trainer, "create_model_card", None)
    if original_create_model_card is None:
        return

    warned = False

    def safe_create_model_card(*args, **kwargs):
        nonlocal warned
        try:
            return original_create_model_card(*args, **kwargs)
        except PackageNotFoundError as exc:
            if not warned:
                warned = True
                missing = getattr(exc, "name", None) or "required package metadata"
                warnings.warn(
                    f"Skipping model card generation because metadata for {missing!r} is unavailable.",
                    stacklevel=2,
                )
            return None

    trainer.create_model_card = safe_create_model_card


def resolve_attention_implementation(config: dict) -> str | None:
    model_config = config.get("model", {})
    configured = model_config.get("attn_implementation")
    if isinstance(configured, str) and configured:
        return configured

    model_name = str(model_config.get("name", "")).lower()
    if "gpt-oss" not in model_name:
        return None

    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    major, _minor = torch.cuda.get_device_capability(0)
    if major < 9:
        return "eager"
    return None


def train_with_unsloth(dataset_path: Path, output_dir: Path, config: dict) -> None:
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    apply_wandb_environment(config, output_dir)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"].get("load_in_4bit", False)
    attn_implementation = resolve_attention_implementation(config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
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
        text = format_sft_training_text(
            row["prompt"],
            row["input_text"],
            row["target_text"],
            tokenizer.eos_token,
        )
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
            report_to=build_report_to(config),
            run_name=resolve_run_name(config, output_dir),
        ),
    )
    _make_model_card_generation_best_effort(trainer)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "train_config.json").write_bytes(orjson.dumps(config, option=orjson.OPT_INDENT_2))
