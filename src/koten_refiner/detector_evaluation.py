from __future__ import annotations

from pathlib import Path
from typing import Protocol

import orjson
import sentencepiece as spm


OPEN_TAG = "<error>"
CLOSE_TAG = "</error>"
DEFAULT_DETECTOR_TOKENIZER = "rinna/japanese-roberta-base"
DEFAULT_DETECTOR_CHAR_CHUNK_SIZE = 256


class OffsetTokenizer(Protocol):
    def offsets(self, text: str) -> list[tuple[int, int]]:
        ...


class SentencePieceOffsetTokenizer:
    def __init__(self, model_path: str) -> None:
        self.processor = spm.SentencePieceProcessor(model_file=model_path)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        proto = self.processor.encode(text, out_type="immutable_proto")
        return [(piece.begin, piece.end) for piece in proto.pieces if piece.begin != piece.end]


def char_labels_from_tagged(tagged_text: str) -> tuple[str, list[int]]:
    plain_chars: list[str] = []
    labels: list[int] = []
    idx = 0
    while idx < len(tagged_text):
        if tagged_text.startswith(OPEN_TAG, idx):
            open_start = idx
            idx += len(OPEN_TAG)
            close = tagged_text.find(CLOSE_TAG, idx)
            if close == -1:
                # Fail closed on malformed output so evaluation can continue.
                remainder = tagged_text[open_start:]
                plain_chars.extend(remainder)
                labels.extend([0] * len(remainder))
                break
            segment = tagged_text[idx:close]
            plain_chars.extend(segment)
            labels.extend([1] * len(segment))
            idx = close + len(CLOSE_TAG)
            continue
        plain_chars.append(tagged_text[idx])
        labels.append(0)
        idx += 1
    return "".join(plain_chars), labels


def _offsets_for_plain_text(
    plain_text: str,
    tokenizer: OffsetTokenizer,
    char_chunk_size: int = DEFAULT_DETECTOR_CHAR_CHUNK_SIZE,
) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    step = max(1, char_chunk_size)
    for start_idx in range(0, len(plain_text), step):
        chunk_text = plain_text[start_idx : start_idx + step]
        for start, end in tokenizer.offsets(chunk_text):
            offsets.append((start_idx + start, start_idx + end))
    return offsets


def _project_char_labels_to_offsets(
    char_labels: list[int],
    offsets: list[tuple[int, int]],
) -> list[int]:
    labels: list[int] = []
    for start, end in offsets:
        labels.append(1 if any(char_labels[idx] for idx in range(start, end)) else 0)
    return labels


def load_detector_tokenizer(model_name: str = DEFAULT_DETECTOR_TOKENIZER):
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=model_name, filename="spiece.model")
    return SentencePieceOffsetTokenizer(model_path)


def _confusion_counts(true_labels: list[int], pred_labels: list[int]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for y_true, y_pred in zip(true_labels, pred_labels, strict=True):
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _metrics_from_confusion(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            yield orjson.loads(line)


def _evaluate_detector_row(
    row: dict,
    tokenizer: OffsetTokenizer | None = None,
    char_chunk_size: int = DEFAULT_DETECTOR_CHAR_CHUNK_SIZE,
) -> tuple[int, int, int, int]:
    true_text, true_char_labels = char_labels_from_tagged(row["target_text"])
    pred_text, pred_char_labels = char_labels_from_tagged(row["prediction_text"])
    if tokenizer is None:
        true_labels = true_char_labels
        pred_labels = pred_char_labels if pred_text == true_text else [0] * len(true_labels)
    else:
        offsets = _offsets_for_plain_text(true_text, tokenizer, char_chunk_size=char_chunk_size)
        true_labels = _project_char_labels_to_offsets(true_char_labels, offsets)
        pred_labels = (
            _project_char_labels_to_offsets(pred_char_labels, offsets)
            if pred_text == true_text
            else [0] * len(true_labels)
        )
    return _confusion_counts(true_labels, pred_labels)


def compute_detector_metrics(
    rows: list[dict],
    tokenizer: OffsetTokenizer | None = None,
    char_chunk_size: int = DEFAULT_DETECTOR_CHAR_CHUNK_SIZE,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        row_tp, row_fp, row_tn, row_fn = _evaluate_detector_row(
            row,
            tokenizer=tokenizer,
            char_chunk_size=char_chunk_size,
        )
        tp += row_tp
        fp += row_fp
        tn += row_tn
        fn += row_fn
    return _metrics_from_confusion(tp, fp, tn, fn)


def compute_detector_metrics_from_path(
    path: Path,
    tokenizer: OffsetTokenizer | None = None,
    char_chunk_size: int = DEFAULT_DETECTOR_CHAR_CHUNK_SIZE,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for row in iter_jsonl(path):
        row_tp, row_fp, row_tn, row_fn = _evaluate_detector_row(
            row,
            tokenizer=tokenizer,
            char_chunk_size=char_chunk_size,
        )
        tp += row_tp
        fp += row_fp
        tn += row_tn
        fn += row_fn
    return _metrics_from_confusion(tp, fp, tn, fn)


def write_json(path: Path, payload: dict) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
