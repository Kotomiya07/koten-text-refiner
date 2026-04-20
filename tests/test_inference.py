from __future__ import annotations

import pytest
import torch

from koten_refiner.inference import (
    DETECTOR_MAX_NEW_TOKENS,
    apply_edit_only_prediction,
    build_generation_config,
    generate_texts,
    has_only_error_markup,
    normalize_detector_prediction,
    resolve_max_new_tokens,
)


def test_apply_edit_only_prediction_keeps_untouched_text():
    tagged = '今日は<error id="1">天期</error>が<error id="2">良</error>い'
    pred = "1\t天気\n2\t<KEEP>"
    restored = apply_edit_only_prediction(tagged, pred)
    assert restored == "今日は天気が良い"


def test_apply_edit_only_prediction_handles_multiple_spans_and_missing_updates():
    tagged = 'A<error id="1">B</error>C<error id="2">D</error>E<error id="3">F</error>'
    pred = "1\tX\n2\t<KEEP>"
    restored = apply_edit_only_prediction(tagged, pred)
    assert restored == "AXCDEF"


def test_resolve_max_new_tokens_uses_detector_safe_default():
    assert resolve_max_new_tokens("detector", None) == DETECTOR_MAX_NEW_TOKENS


def test_build_generation_config_adds_detector_repetition_controls():
    config = build_generation_config("detector", pad_token_id=0, eos_token_id=1)
    assert config["max_new_tokens"] == DETECTOR_MAX_NEW_TOKENS
    assert config["repetition_penalty"] == 1.1
    assert config["no_repeat_ngram_size"] == 6


def test_build_generation_config_keeps_generic_defaults_for_corrector():
    config = build_generation_config("corrector", pad_token_id=0, eos_token_id=1)
    assert config["max_new_tokens"] == 512
    assert "repetition_penalty" not in config
    assert "no_repeat_ngram_size" not in config


def test_has_only_error_markup_rejects_unknown_tags():
    assert not has_only_error_markup("A<color>B</color>C")


def test_normalize_detector_prediction_falls_back_on_unknown_tags():
    assert normalize_detector_prediction("A<color>B</color>C", "ABC") == "ABC"


def test_normalize_detector_prediction_accepts_balanced_error_tags():
    assert normalize_detector_prediction("A<error>B</error>C", "ABC") == "A<error>B</error>C"


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, texts, return_tensors="pt", padding=False):
        assert return_tensors == "pt"
        assert padding is True
        assert texts == [
            "prompt-a\n\n入力:\ninput-a\n\n出力:\n",
            "prompt-b\n\n入力:\ninput-b\n\n出力:\n",
        ]
        return FakeBatch({"input_ids": torch.tensor([[10, 11, 12], [20, 21, 22]])})

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "/".join(str(int(token)) for token in token_ids)


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 123
        assert kwargs["no_repeat_ngram_size"] == 6
        assert kwargs["repetition_penalty"] == 1.1
        return torch.tensor(
            [
                [10, 11, 12, 101, 102],
                [20, 21, 22, 201, 202],
            ]
        )


def test_generate_texts_decodes_batch_outputs():
    generated = generate_texts(
        FakeModel(),
        FakeTokenizer(),
        ["prompt-a", "prompt-b"],
        ["input-a", "input-b"],
        task="detector",
        max_new_tokens=123,
    )
    assert generated == ["101/102", "201/202"]


def test_generate_texts_rejects_mismatched_batch_lengths():
    with pytest.raises(ValueError, match="same length"):
        generate_texts(FakeModel(), FakeTokenizer(), ["prompt-a"], ["input-a", "input-b"], task="detector")
