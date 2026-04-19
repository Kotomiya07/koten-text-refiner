from __future__ import annotations

from pathlib import Path

from koten_refiner.detector_evaluation import (
    DEFAULT_DETECTOR_TOKENIZER,
    char_labels_from_tagged,
    compute_detector_metrics,
    compute_detector_metrics_from_path,
)
from koten_refiner.evaluation import (
    summarize_metric_files,
    write_summary_csv,
)


class MockTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def offsets(self, text: str) -> list[tuple[int, int]]:
        self.calls.append(text)
        offsets = []
        cursor = 0
        chunk_sizes = (2, 1, 2, max(0, len(text) - 5))
        for size in chunk_sizes:
            if size <= 0:
                continue
            end = min(len(text), cursor + size)
            offsets.append((cursor, end))
            cursor = end
            if cursor >= len(text):
                break
        return offsets


def test_char_labels_from_tagged_marks_only_error_region():
    plain, labels = char_labels_from_tagged("今日は<error>天期</error>です")
    assert plain == "今日は天期です"
    assert labels == [0, 0, 0, 1, 1, 0, 0]


def test_char_labels_from_tagged_handles_unclosed_error_tag_safely():
    plain, labels = char_labels_from_tagged("<error>未完了")
    assert plain == "<error>未完了"
    assert labels == [0] * len(plain)


def test_subtoken_labels_from_tagged_project_error_to_tokens():
    rows = [
        {
            "target_text": "今日は<error>天期</error>です",
            "prediction_text": "今日は<error>天期</error>です",
        }
    ]
    metrics = compute_detector_metrics(rows, tokenizer=MockTokenizer())
    plain, labels = char_labels_from_tagged("今日は<error>天期</error>です")
    assert plain == "今日は天期です"
    assert labels == [0, 0, 0, 1, 1, 0, 0]
    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_compute_detector_metrics_matches_known_small_example():
    rows = [
        {
            "target_text": "A<error>BC</error>D",
            "prediction_text": "A<error>BC</error>D",
        },
        {
            "target_text": "W<error>X</error>YZ",
            "prediction_text": "WXYZ",
        },
    ]
    metrics = compute_detector_metrics(rows)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 2 / 3
    assert round(metrics["f1"], 6) == round(0.8, 6)
    assert round(metrics["accuracy"], 6) == round(7 / 8, 6)


def test_compute_detector_metrics_uses_subtoken_projection_when_tokenizer_is_given():
    rows = [
        {
            "target_text": "今日は<error>天期</error>です",
            "prediction_text": "今日は天期です",
        }
    ]
    metrics = compute_detector_metrics(rows, tokenizer=MockTokenizer())
    assert metrics == {
        "accuracy": 0.75,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_compute_detector_metrics_char_fallback_is_available():
    rows = [
        {
            "target_text": "今日は<error>天期</error>です",
            "prediction_text": "今日は<error>天期</error>です",
        }
    ]
    metrics = compute_detector_metrics(rows, tokenizer=None)
    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_compute_detector_metrics_from_path_supports_subtokens(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"target_text":"今日は<error>天期</error>です","prediction_text":"今日は<error>天期</error>です"}\n',
        encoding="utf-8",
    )
    metrics = compute_detector_metrics_from_path(path, tokenizer=MockTokenizer())
    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_compute_detector_metrics_batches_subtoken_tokenization():
    rows = [
        {
            "target_text": "今日は<error>天期</error>です",
            "prediction_text": "今日は<error>天期</error>です",
        },
        {
            "target_text": "今日は<error>天期</error>です",
            "prediction_text": "今日は天期です",
        },
    ]
    metrics = compute_detector_metrics(rows, tokenizer=MockTokenizer())
    assert metrics == {
        "accuracy": 0.875,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6666666666666666,
    }


def test_subtoken_labels_chunk_long_text_to_limit_memory():
    tokenizer = MockTokenizer()
    rows = [
        {
            "target_text": "<error>" + ("あ" * 300) + "</error>",
            "prediction_text": "<error>" + ("あ" * 300) + "</error>",
        }
    ]
    metrics = compute_detector_metrics(rows, tokenizer=tokenizer, char_chunk_size=128)
    assert metrics["accuracy"] == 1.0
    assert len(tokenizer.calls) == 3


def test_summarize_metric_files_and_write_summary_csv(tmp_path: Path):
    path_a = tmp_path / "fold_0_metrics.json"
    path_b = tmp_path / "fold_1_metrics.json"
    path_a.write_text('{"bleu": 10, "crr": 0.8, "wrr": 0.5}', encoding="utf-8")
    path_b.write_text('{"bleu": 20, "crr": 0.6, "wrr": 0.7}', encoding="utf-8")
    summary = summarize_metric_files([path_a, path_b])
    assert summary == {"bleu": 15.0, "crr": 0.7, "wrr": 0.6}

    csv_path = tmp_path / "summary.csv"
    write_summary_csv(csv_path, summary)
    assert csv_path.read_text(encoding="utf-8") == "metric,value\nbleu,15.0\ncrr,0.7\nwrr,0.6\n"


def test_default_detector_tokenizer_model_is_documented_constant():
    assert DEFAULT_DETECTOR_TOKENIZER == "rinna/japanese-roberta-base"
