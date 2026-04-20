from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import orjson
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from koten_refiner.alignment import CLOSE_TAG, OPEN_TAG
from koten_refiner.models import TaskName


GenerationValue: TypeAlias = int | float | bool | None

DEFAULT_MAX_NEW_TOKENS = 512
DETECTOR_MAX_NEW_TOKENS = 256
DETECTOR_NO_REPEAT_NGRAM_SIZE = 6
DETECTOR_REPETITION_PENALTY = 1.1


def load_generation_model(model_dir: Path, load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization_config = None
    if load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def render_sft_prompt(prompt: str, input_text: str) -> str:
    return f"{prompt}\n\n入力:\n{input_text}\n\n出力:\n"


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

    rendered = [render_sft_prompt(prompt, input_text) for prompt, input_text in zip(prompts, input_texts, strict=True)]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
    generation_config = build_generation_config(
        task,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    output = model.generate(
        **inputs,
        **generation_config,
    )
    prompt_width = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
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


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")
