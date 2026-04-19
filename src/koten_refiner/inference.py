from __future__ import annotations

from pathlib import Path

import orjson
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from koten_refiner.alignment import CLOSE_TAG, OPEN_TAG


def load_generation_model(model_dir: Path, load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, input_text: str, max_new_tokens: int = 512) -> str:
    rendered = render_sft_prompt(prompt, input_text)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated.strip()


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


def normalize_detector_prediction(prediction_text: str, fallback_raw_text: str) -> str:
    if OPEN_TAG not in prediction_text or CLOSE_TAG not in prediction_text:
        return fallback_raw_text
    return prediction_text


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")
