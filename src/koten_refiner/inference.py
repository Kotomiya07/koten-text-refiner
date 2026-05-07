from __future__ import annotations

import os
from pathlib import Path
import re
from typing import TypeAlias

import orjson
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from koten_refiner.alignment import CLOSE_TAG, OPEN_TAG, ErrorSpan, tag_error_spans
from koten_refiner.models import TaskName


GenerationValue: TypeAlias = int | float | bool | None | list[int]

DEFAULT_MAX_NEW_TOKENS = 512
DETECTOR_MAX_NEW_TOKENS = 2048
DETECTOR_NO_REPEAT_NGRAM_SIZE = 6
DETECTOR_REPETITION_PENALTY = 1.1


def normalize_error_tag_variants(text: str) -> str:
    text = re.sub(r"<\s*/\s*(?:errors?|err|エラー)\s*>", CLOSE_TAG, text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*>", CLOSE_TAG, text)
    text = re.sub(r"<\s*(?:error|err|エラー)(?:\s+[^<>]*)?\s*>", OPEN_TAG, text, flags=re.IGNORECASE)
    return text


def _build_quantization_config(*, cpu_offload: bool = False) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=cpu_offload,
    )


def _resolve_attention_implementation(model_name: str) -> str | None:
    if "gpt-oss" not in model_name.lower():
        return None
    if not torch.cuda.is_available():
        return None
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 9:
        return "eager"
    return None


def _finalize_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _attach_training_runtime_options(tokenizer, train_config: dict | None):
    train_section = (train_config or {}).get("train", {})
    tokenizer.koten_use_chat_template = bool(train_section.get("use_chat_template", False))
    tokenizer.koten_chat_content_parts = bool(train_section.get("chat_content_parts", True))
    return tokenizer


def _load_train_config(model_dir: Path) -> dict | None:
    train_config_path = model_dir / "train_config.json"
    if not train_config_path.exists():
        return None
    return orjson.loads(train_config_path.read_bytes())


def _load_unsloth_peft_model(model_dir: Path):
    # GPT-OSS inference on this stack trips Unsloth's compiled path for long inputs.
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    from unsloth import FastLanguageModel

    max_seq_length = 2048
    load_in_4bit = True
    model_name = str(model_dir)
    train_config = _load_train_config(model_dir)
    if train_config is not None:
        model_config = train_config.get("model", {})
        max_seq_length = int(model_config.get("max_seq_length", max_seq_length))
        load_in_4bit = bool(model_config.get("load_in_4bit", load_in_4bit))
        model_name = str(model_config.get("name", model_name))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        attn_implementation=_resolve_attention_implementation(model_name),
    )
    FastLanguageModel.for_inference(model)
    return model, _attach_training_runtime_options(_finalize_tokenizer(tokenizer), train_config)


def load_generation_model(model_dir: Path, load_in_4bit: bool = True):
    if (model_dir / "adapter_config.json").exists():
        return _load_unsloth_peft_model(model_dir)

    train_config = _load_train_config(model_dir)
    tokenizer = _attach_training_runtime_options(_finalize_tokenizer(
        AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ), train_config)
    quantization_config = None
    if load_in_4bit and torch.cuda.is_available():
        quantization_config = _build_quantization_config()

    load_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
        "trust_remote_code": True,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
        "attn_implementation": _resolve_attention_implementation(str(model_dir)),
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            **load_kwargs,
        )
    except ValueError as exc:
        if quantization_config is None or "Some modules are dispatched on the CPU or the disk" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            **{
                **load_kwargs,
                "quantization_config": _build_quantization_config(cpu_offload=True),
            },
        )
    model.eval()
    return model, tokenizer


def render_sft_prompt(prompt: str, input_text: str) -> str:
    return f"{prompt}\n\n入力:\n{input_text}\n\n出力:\n"


def render_generation_prompt(tokenizer, prompt: str, input_text: str) -> str:
    if getattr(tokenizer, "koten_use_chat_template", False):
        content = f"{prompt}\n\n入力:\n{input_text}"
        if getattr(tokenizer, "koten_chat_content_parts", True):
            content = [{"type": "text", "text": content}]
        messages = [{"role": "user", "content": content}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return render_sft_prompt(prompt, input_text)


def resolve_max_new_tokens(task: TaskName, max_new_tokens: int | None) -> int:
    if max_new_tokens is not None:
        return max_new_tokens
    if task == "detector":
        return DETECTOR_MAX_NEW_TOKENS
    return DEFAULT_MAX_NEW_TOKENS


def build_generation_config(
    task: TaskName,
    pad_token_id: int | None,
    eos_token_id: int | None,
    max_new_tokens: int | None = None,
) -> dict[str, GenerationValue]:
    config: dict[str, GenerationValue] = {
        "max_new_tokens": resolve_max_new_tokens(task, max_new_tokens),
        "do_sample": False,
        "temperature": None,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if task == "detector":
        config["repetition_penalty"] = DETECTOR_REPETITION_PENALTY
        config["no_repeat_ngram_size"] = DETECTOR_NO_REPEAT_NGRAM_SIZE
    return config


def clean_decoded_generation(tokenizer, text: str) -> str:
    text = re.sub(r"^\s*(?:<\|channel\|>\s*\w+\s*)?<\|message\|>\s*", "", text)
    text = re.sub(r"^\s*<\|channel\|>\s*\w+\s*", "", text)
    text = normalize_error_tag_variants(text)
    removable_tokens = {
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "bos_token", None),
        "<|end|>",
    }
    protected_tokens = {OPEN_TAG, CLOSE_TAG}
    for token in sorted((token for token in removable_tokens if token and token not in protected_tokens), key=len, reverse=True):
        while text.endswith(token):
            text = text[: -len(token)].rstrip()
        while text.startswith(token):
            text = text[len(token) :].lstrip()
    return text.strip()


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    input_text: str,
    task: TaskName,
    max_new_tokens: int | None = None,
) -> str:
    generated = generate_texts(
        model,
        tokenizer,
        [prompt],
        [input_text],
        task=task,
        max_new_tokens=max_new_tokens,
    )
    return generated[0]


@torch.inference_mode()
def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    input_texts: list[str],
    task: TaskName,
    max_new_tokens: int | None = None,
) -> list[str]:
    if len(prompts) != len(input_texts):
        raise ValueError("prompts and input_texts must have the same length")
    if not prompts:
        return []

    rendered = [
        render_generation_prompt(tokenizer, prompt, input_text)
        for prompt, input_text in zip(prompts, input_texts, strict=True)
    ]
    inputs = tokenizer(text=rendered, return_tensors="pt", padding=True).to(model.device)
    generation_config = build_generation_config(
        task,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    end_token_id = None
    try:
        end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    except AttributeError:
        end_token_id = None
    if isinstance(end_token_id, int) and end_token_id >= 0 and tokenizer.eos_token_id is not None:
        generation_config["eos_token_id"] = sorted({int(tokenizer.eos_token_id), end_token_id})
    output = model.generate(
        **inputs,
        **generation_config,
    )
    prompt_width = inputs["input_ids"].shape[1]
    return [
        clean_decoded_generation(
            tokenizer,
            tokenizer.decode(row[prompt_width:], skip_special_tokens=False),
        )
        for row in output
    ]


def parse_edit_only_lines(text: str) -> dict[int, str]:
    updates: dict[int, str] = {}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        idx_text, value = line.split("\t", 1)
        try:
            idx = int(idx_text.strip())
        except ValueError:
            continue
        updates[idx] = value.strip()
    return updates


def apply_edit_only_prediction(tagged_input: str, prediction_text: str) -> str:
    updates = parse_edit_only_lines(prediction_text)
    output: list[str] = []
    cursor = 0
    span_idx = 0
    while True:
        start = tagged_input.find('<error id="', cursor)
        if start == -1:
            output.append(tagged_input[cursor:])
            break
        output.append(tagged_input[cursor:start])
        id_start = tagged_input.find('"', start + len('<error id=')) + 1
        id_end = tagged_input.find('"', id_start)
        span_idx = int(tagged_input[id_start:id_end])
        content_start = tagged_input.find(">", id_end) + 1
        content_end = tagged_input.find("</error>", content_start)
        original = tagged_input[content_start:content_end]
        replacement = updates.get(span_idx, "<KEEP>")
        output.append(original if replacement == "<KEEP>" else replacement)
        cursor = content_end + len("</error>")
    return "".join(output)


def has_only_error_markup(text: str) -> bool:
    depth = 0
    idx = 0
    while idx < len(text):
        if text.startswith(OPEN_TAG, idx):
            if depth != 0:
                return False
            depth = 1
            idx += len(OPEN_TAG)
            continue
        if text.startswith(CLOSE_TAG, idx):
            if depth != 1:
                return False
            depth = 0
            idx += len(CLOSE_TAG)
            continue
        if text[idx] == "<":
            return False
        idx += 1
    return depth == 0


def normalize_detector_prediction(prediction_text: str, fallback_raw_text: str) -> str:
    if OPEN_TAG not in prediction_text or CLOSE_TAG not in prediction_text:
        return fallback_raw_text
    if not has_only_error_markup(prediction_text):
        return fallback_raw_text
    return prediction_text


def _extract_json_array(text: str) -> str | None:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def parse_detector_span_prediction(prediction_text: str, raw_ocr_text: str) -> list[ErrorSpan]:
    json_text = _extract_json_array(prediction_text.strip())
    if json_text is None:
        return []
    try:
        parsed = orjson.loads(json_text)
    except orjson.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    text_length = len(raw_ocr_text)
    spans: list[ErrorSpan] = []
    for item in parsed:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, list | tuple) and len(item) >= 2:
            start = item[0]
            end = item[1]
        else:
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        start = max(0, min(start, text_length))
        end = max(0, min(end, text_length))
        if start < end:
            spans.append(ErrorSpan(start, end))
    if not spans:
        return []

    spans = sorted(spans, key=lambda span: (span.start, span.end))
    merged = [spans[0]]
    for span in spans[1:]:
        previous = merged[-1]
        if span.start <= previous.end:
            merged[-1] = ErrorSpan(previous.start, max(previous.end, span.end))
        else:
            merged.append(span)
    return merged


def restore_detector_span_prediction(prediction_text: str, raw_ocr_text: str) -> str:
    return tag_error_spans(raw_ocr_text, parse_detector_span_prediction(prediction_text, raw_ocr_text))


def detector_prediction_diagnostics(rows: list[dict]) -> dict[str, int | float]:
    total = len(rows)
    counts = {
        "total": total,
        "has_open_tag": 0,
        "has_close_tag": 0,
        "has_error_pair": 0,
        "valid_error_markup": 0,
        "invalid_error_markup": 0,
        "no_error_tags": 0,
        "contains_unknown_markup": 0,
        "contains_thinking_marker": 0,
        "contains_explanatory_text": 0,
        "too_short": 0,
        "too_long": 0,
        "normalized_to_raw_ocr": 0,
    }
    explanatory_markers = ("ユーザー", "指示", "説明", "補足", "求め", "まず", "考え", "The ")
    for row in rows:
        prediction = str(row.get("prediction_text", ""))
        raw_ocr = str(row.get("raw_ocr_text", ""))
        has_open = OPEN_TAG in prediction
        has_close = CLOSE_TAG in prediction
        has_pair = has_open and has_close
        valid_markup = has_pair and has_only_error_markup(prediction)
        if has_open:
            counts["has_open_tag"] += 1
        if has_close:
            counts["has_close_tag"] += 1
        if has_pair:
            counts["has_error_pair"] += 1
        if valid_markup:
            counts["valid_error_markup"] += 1
        elif has_open or has_close or "<" in prediction or ">" in prediction:
            counts["invalid_error_markup"] += 1
        if not has_open and not has_close:
            counts["no_error_tags"] += 1
        if "<" in prediction and not valid_markup:
            counts["contains_unknown_markup"] += 1
        if "<think>" in prediction or "</think>" in prediction:
            counts["contains_thinking_marker"] += 1
        if any(marker in prediction for marker in explanatory_markers):
            counts["contains_explanatory_text"] += 1
        if raw_ocr:
            if len(prediction) < len(raw_ocr) * 0.5:
                counts["too_short"] += 1
            if len(prediction) > len(raw_ocr) * 2.0:
                counts["too_long"] += 1
        if normalize_detector_prediction(prediction, raw_ocr) == raw_ocr:
            counts["normalized_to_raw_ocr"] += 1
    if total == 0:
        counts["valid_error_markup_rate"] = 0.0
        counts["normalized_to_raw_ocr_rate"] = 0.0
        counts["explanatory_text_rate"] = 0.0
    else:
        counts["valid_error_markup_rate"] = counts["valid_error_markup"] / total
        counts["normalized_to_raw_ocr_rate"] = counts["normalized_to_raw_ocr"] / total
        counts["explanatory_text_rate"] = counts["contains_explanatory_text"] / total
    return counts


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")
