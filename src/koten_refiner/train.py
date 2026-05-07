from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError
import os
from pathlib import Path
import re
import warnings

import orjson
import yaml

from koten_refiner.alignment import CLOSE_TAG, OPEN_TAG


ERROR_TAG_SPECIAL_TOKENS = (OPEN_TAG, CLOSE_TAG)


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


def format_sft_prompt_text(prompt: str, input_text: str) -> str:
    return f"{prompt}\n\n入力:\n{input_text}\n\n出力:\n"


def format_sft_completion_text(target_text: str, eos_token: str | None) -> str:
    return append_eos_token(target_text, eos_token)


def format_chat_user_content(prompt: str, input_text: str) -> str:
    return f"{prompt}\n\n入力:\n{input_text}"


def chat_text_part(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def build_sft_training_row(
    row: dict,
    eos_token: str | None,
    *,
    completion_only_loss: bool = True,
    use_chat_template: bool = False,
    chat_content_parts: bool = True,
) -> dict:
    if use_chat_template:
        user_content = format_chat_user_content(row["prompt"], row["input_text"])
        prompt_content = chat_text_part(user_content) if chat_content_parts else user_content
        completion_content = chat_text_part(row["target_text"]) if chat_content_parts else row["target_text"]
        return {
            "prompt": [{"role": "user", "content": prompt_content}],
            "completion": [{"role": "assistant", "content": completion_content}],
        }
    if completion_only_loss:
        return {
            "prompt": format_sft_prompt_text(row["prompt"], row["input_text"]),
            "completion": format_sft_completion_text(row["target_text"], eos_token),
        }
    return {
        "text": format_sft_training_text(
            row["prompt"],
            row["input_text"],
            row["target_text"],
            eos_token,
        )
    }


def _train_flag(config: dict, key: str, default: bool) -> bool:
    return bool(config.get("train", {}).get(key, default))


def text_tokenizer(processing_class):
    return getattr(processing_class, "tokenizer", processing_class)


def add_error_tag_special_tokens(processing_class, model=None) -> list[int]:
    tokenizer = text_tokenizer(processing_class)
    existing = set(getattr(tokenizer, "additional_special_tokens", None) or [])
    missing = [token for token in ERROR_TAG_SPECIAL_TOKENS if token not in existing]
    if missing:
        try:
            num_added = tokenizer.add_special_tokens(
                {"additional_special_tokens": missing},
                replace_additional_special_tokens=False,
            )
        except TypeError:
            num_added = tokenizer.add_special_tokens({"additional_special_tokens": missing})
        if num_added and model is not None:
            model.resize_token_embeddings(len(tokenizer))
    return [int(tokenizer.convert_tokens_to_ids(token)) for token in ERROR_TAG_SPECIAL_TOKENS]


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


def _allow_meta_nonzero_best_effort() -> None:
    try:
        import torch.fx.experimental._config as torch_fx_config
    except ImportError:
        return
    torch_fx_config.meta_nonzero_assume_all_nonzero = True


def _skip_unsloth_untrained_token_check_for_model(model) -> None:
    model_name = str(getattr(getattr(model, "config", None), "_name_or_path", "") or "")
    if not model_name:
        return
    current = os.environ.get("UNSLOTH_IGNORED_TOKENIZER_NAMES", "")
    names = [name for name in current.split("\n") if name]
    if model_name not in names:
        names.append(model_name)
    os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = "\n".join(names)


def train_with_unsloth(dataset_path: Path, output_dir: Path, config: dict) -> None:
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")
    _allow_meta_nonzero_best_effort()
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
    special_token_ids: list[int] = []
    if _train_flag(config, "add_error_tag_tokens", True):
        special_token_ids = add_error_tag_special_tokens(tokenizer, model)
    peft_kwargs = {}
    if special_token_ids and _train_flag(config, "train_error_tag_embeddings", True):
        peft_kwargs["trainable_token_indices"] = special_token_ids
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["train"]["seed"],
        **peft_kwargs,
    )
    if special_token_ids:
        _skip_unsloth_untrained_token_check_for_model(model)

    completion_only_loss = _train_flag(config, "completion_only_loss", True)
    use_chat_template = _train_flag(config, "use_chat_template", False)
    chat_content_parts = _train_flag(config, "chat_content_parts", True)

    def format_row(row: dict) -> dict:
        return build_sft_training_row(
            row,
            text_tokenizer(tokenizer).eos_token,
            completion_only_loss=completion_only_loss,
            use_chat_template=use_chat_template,
            chat_content_parts=chat_content_parts,
        )

    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    dataset_text_field = None if completion_only_loss else "text"
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field=dataset_text_field,
            max_length=max_seq_length,
            completion_only_loss=completion_only_loss,
            assistant_only_loss=False,
            packing=False,
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
