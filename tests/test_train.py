import os
import re
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import warnings

import pytest

from koten_refiner.train import (
    _make_model_card_generation_best_effort,
    apply_wandb_environment,
    append_eos_token,
    build_default_run_name,
    build_report_to,
    build_wandb_env,
    format_sft_training_text,
    load_yaml_config,
    resolve_run_name,
    wandb_enabled,
)


class DummyTrainer:
    def __init__(self) -> None:
        self.calls = 0

    def create_model_card(self, *args, **kwargs):
        self.calls += 1
        raise PackageNotFoundError("trl")


def test_make_model_card_generation_best_effort_skips_missing_metadata_once():
    trainer = DummyTrainer()
    _make_model_card_generation_best_effort(trainer)

    with pytest.warns(UserWarning, match="Skipping model card generation"):
        assert trainer.create_model_card(model_name="demo") is None

    assert trainer.calls == 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert trainer.create_model_card(model_name="demo") is None

    assert trainer.calls == 2
    assert not record


def test_append_eos_token_adds_suffix_once():
    assert append_eos_token("abc", "<eos>") == "abc<eos>"
    assert append_eos_token("abc<eos>", "<eos>") == "abc<eos>"


def test_format_sft_training_text_appends_eos_to_target_block():
    text = format_sft_training_text("prompt", "input", "target", "<eos>")
    assert text == "prompt\n\n入力:\ninput\n\n出力:\ntarget<eos>"


def test_wandb_helpers_stay_disabled_by_default(tmp_path):
    config = {"train": {"seed": 42}}
    assert wandb_enabled(config) is False
    assert build_report_to(config) == []
    assert resolve_run_name(config, tmp_path) is None
    assert build_wandb_env(config, tmp_path) == {}


def test_wandb_helpers_build_report_name_and_env(tmp_path):
    config = {
        "wandb": {
            "enabled": True,
            "project": "koten-text-refiner",
            "entity": "team-a",
            "group": "detector",
            "job_type": "train",
            "mode": "offline",
            "run_name": "detector-fold-0",
            "tags": ["detector", "fold0"],
        }
    }
    assert wandb_enabled(config) is True
    assert build_report_to(config) == ["wandb"]
    assert resolve_run_name(config, tmp_path) == "detector-fold-0"
    assert build_wandb_env(config, tmp_path) == {
        "WANDB_DIR": str(tmp_path / "wandb"),
        "WANDB_PROJECT": "koten-text-refiner",
        "WANDB_ENTITY": "team-a",
        "WANDB_RUN_GROUP": "detector",
        "WANDB_JOB_TYPE": "train",
        "WANDB_MODE": "offline",
        "WANDB_TAGS": "detector,fold0",
    }


def test_build_default_run_name_uses_timestamp_and_group_label(tmp_path):
    config = {"wandb": {"enabled": True, "group": "detector"}}
    run_name = build_default_run_name(config, tmp_path, now=datetime(2026, 4, 20, 12, 34, 56))
    assert run_name == "20260420_123456_detector"


def test_build_default_run_name_normalizes_group_label(tmp_path):
    config = {"wandb": {"enabled": True, "group": "detector-smoke"}}
    run_name = build_default_run_name(config, tmp_path, now=datetime(2026, 4, 20, 1, 2, 3))
    assert run_name == "20260420_010203_detector_smoke"


def test_resolve_run_name_uses_timestamp_default_when_missing(tmp_path):
    config = {"wandb": {"enabled": True, "group": "detector"}}
    run_name = resolve_run_name(config, tmp_path)
    assert re.fullmatch(r"\d{8}_\d{6}_detector", run_name) is not None


def test_apply_wandb_environment_sets_expected_variables(tmp_path, monkeypatch):
    for key in (
        "WANDB_DIR",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_RUN_GROUP",
        "WANDB_JOB_TYPE",
        "WANDB_MODE",
        "WANDB_TAGS",
    ):
        monkeypatch.delenv(key, raising=False)

    config = {
        "wandb": {
            "enabled": True,
            "project": "koten-text-refiner",
            "mode": "offline",
        }
    }
    apply_wandb_environment(config, tmp_path)

    assert os.environ["WANDB_DIR"] == str(tmp_path / "wandb")
    assert os.environ["WANDB_PROJECT"] == "koten-text-refiner"
    assert os.environ["WANDB_MODE"] == "offline"


def test_gemma4_31b_qlora_config_loads_with_expected_defaults():
    config = load_yaml_config(Path("configs/gemma4_31b_it_qlora.yaml"))

    assert config["model"] == {
        "name": "gemma-4-31B-it",
        "max_seq_length": 2048,
        "load_in_4bit": True,
    }
    assert config["lora"]["r"] == 16
    assert config["lora"]["alpha"] == 32
    assert config["lora"]["dropout"] == 0.05
    assert config["train"]["per_device_batch_size"] == 1
    assert config["train"]["gradient_accumulation_steps"] == 16
    assert config["wandb"]["group"] == "corrector-gemma4-31b-qlora"
