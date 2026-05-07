from __future__ import annotations

import pytest
import torch

import koten_refiner.inference as inference
from koten_refiner.inference import (
    DETECTOR_MAX_NEW_TOKENS,
    _build_quantization_config,
    apply_edit_only_prediction,
    build_generation_config,
    clean_decoded_generation,
    detector_prediction_diagnostics,
    generate_texts,
    has_only_error_markup,
    normalize_error_tag_variants,
    normalize_detector_prediction,
    parse_detector_span_prediction,
    render_generation_prompt,
    restore_detector_span_prediction,
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
    assert config["max_new_tokens"] == 2048
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


def test_restore_detector_span_prediction_projects_json_spans_to_error_tags():
    restored = restore_detector_span_prediction("説明\n[[1,3],[2,4]]", "ABCDE")
    assert restored == "A<error>BCD</error>E"


def test_restore_detector_span_prediction_accepts_legacy_dict_spans():
    restored = restore_detector_span_prediction('[{"start":1,"end":3}]', "ABCDE")
    assert restored == "A<error>BC</error>DE"


def test_parse_detector_span_prediction_rejects_invalid_or_empty_json():
    assert parse_detector_span_prediction("not json", "ABCDE") == []
    assert parse_detector_span_prediction('[{"start": 4, "end": 4}]', "ABCDE") == []


def test_render_generation_prompt_uses_chat_template_when_configured():
    class ChatTokenizer:
        koten_use_chat_template = True

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
            assert messages == [{"role": "user", "content": [{"type": "text", "text": "prompt\n\n入力:\ninput"}]}]
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            return "CHAT-PROMPT"

    assert render_generation_prompt(ChatTokenizer(), "prompt", "input") == "CHAT-PROMPT"


def test_render_generation_prompt_falls_back_when_enable_thinking_is_unsupported():
    class ChatTokenizer:
        koten_use_chat_template = True
        calls = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            self.calls += 1
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword")
            return "CHAT-PROMPT"

    tokenizer = ChatTokenizer()
    assert render_generation_prompt(tokenizer, "prompt", "input") == "CHAT-PROMPT"
    assert tokenizer.calls == 2


def test_detector_prediction_diagnostics_counts_failure_modes():
    rows = [
        {
            "prediction_text": "A<error>B</error>C",
            "raw_ocr_text": "ABC",
        },
        {
            "prediction_text": "ユーザーの指示に従います <think>考え</think>",
            "raw_ocr_text": "ABCDE",
        },
        {
            "prediction_text": "短",
            "raw_ocr_text": "ABCDE",
        },
    ]
    diagnostics = detector_prediction_diagnostics(rows)

    assert diagnostics["total"] == 3
    assert diagnostics["valid_error_markup"] == 1
    assert diagnostics["invalid_error_markup"] == 1
    assert diagnostics["no_error_tags"] == 2
    assert diagnostics["contains_thinking_marker"] == 1
    assert diagnostics["contains_explanatory_text"] == 1
    assert diagnostics["too_short"] == 1
    assert diagnostics["normalized_to_raw_ocr"] == 2
    assert diagnostics["valid_error_markup_rate"] == 1 / 3


def test_clean_decoded_generation_keeps_error_special_tokens():
    class DummyTokenizer:
        eos_token = "<|im_end|>"
        pad_token = "<|pad|>"
        bos_token = "<|im_start|>"

    text = "<|im_start|><error>誤</error><|im_end|>"
    assert clean_decoded_generation(DummyTokenizer(), text) == "<error>誤</error>"


def test_normalize_error_tag_variants_canonicalizes_common_llm_typos():
    text = '<Error>誤</エラー><error link="">字</errors><err>仮</ err><error>脱</>'
    assert normalize_error_tag_variants(text) == (
        "<error>誤</error><error>字</error><error>仮</error><error>脱</error>"
    )


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    eos_token = "<eos>"
    pad_token = "<pad>"
    bos_token = "<bos>"

    def __call__(self, text=None, return_tensors="pt", padding=False):
        assert return_tensors == "pt"
        assert padding is True
        assert text == [
            "prompt-a\n\n入力:\ninput-a\n\n出力:\n",
            "prompt-b\n\n入力:\ninput-b\n\n出力:\n",
        ]
        return FakeBatch({"input_ids": torch.tensor([[10, 11, 12], [20, 21, 22]])})

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
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


def test_build_quantization_config_can_enable_cpu_offload():
    config = _build_quantization_config(cpu_offload=True)
    assert config.load_in_4bit is True
    assert config.llm_int8_enable_fp32_cpu_offload is True


def test_load_generation_model_retries_with_cpu_offload(monkeypatch, tmp_path):
    calls: list[object] = []

    class DummyTokenizer:
        pad_token = None
        eos_token = "</s>"
        padding_side = "right"

    class DummyModel:
        def eval(self):
            return self

    def fake_tokenizer_from_pretrained(model_dir, trust_remote_code=True):
        assert model_dir == str(tmp_path)
        assert trust_remote_code is True
        return DummyTokenizer()

    def fake_model_from_pretrained(model_dir, **kwargs):
        calls.append(kwargs["quantization_config"])
        if len(calls) == 1:
            raise ValueError("Some modules are dispatched on the CPU or the disk")
        return DummyModel()

    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(inference.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    model, tokenizer = inference.load_generation_model(tmp_path)

    assert isinstance(model, DummyModel)
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "left"
    assert len(calls) == 2
    assert calls[0].llm_int8_enable_fp32_cpu_offload is False
    assert calls[1].llm_int8_enable_fp32_cpu_offload is True


def test_load_generation_model_uses_unsloth_for_adapter_dirs(monkeypatch, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "train_config.json").write_bytes(
        b'{"model":{"max_seq_length":1024,"load_in_4bit":false}}'
    )

    captured: dict[str, object] = {}

    class DummyTokenizer:
        pad_token = None
        eos_token = "</s>"
        padding_side = "right"

    class DummyFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            captured["kwargs"] = kwargs
            return object(), DummyTokenizer()

        @staticmethod
        def for_inference(model):
            captured["model"] = model

    import sys
    import types

    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=DummyFastLanguageModel))

    model, tokenizer = inference.load_generation_model(tmp_path)

    assert captured["kwargs"] == {
        "model_name": str(tmp_path),
        "max_seq_length": 1024,
        "dtype": None,
        "load_in_4bit": False,
        "attn_implementation": None,
    }
    assert captured["model"] is model
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "left"
