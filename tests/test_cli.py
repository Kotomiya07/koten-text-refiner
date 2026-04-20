from __future__ import annotations

from pathlib import Path

from koten_refiner import cli


def test_progress_bar_creates_sample_tqdm(monkeypatch):
    captured: dict[str, object] = {}

    def fake_tqdm(**kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cli, "tqdm", fake_tqdm)

    cli._progress_bar(2, task="detector", fold=3, split="test")

    assert captured["kwargs"] == {
        "total": 2,
        "desc": "detector fold=3 split=test",
        "unit": "sample",
    }


def test_batch_rows_splits_rows_by_batch_size():
    rows = [{"record_id": "a"}, {"record_id": "b"}, {"record_id": "c"}]
    assert list(cli._batch_rows(rows, 2)) == [
        [{"record_id": "a"}, {"record_id": "b"}],
        [{"record_id": "c"}],
    ]


def test_predict_fold_uses_batch_generation(monkeypatch, tmp_path: Path):
    import koten_refiner.inference as inference

    rows = [
        {"record_id": "a", "prompt": "p1", "input_text": "i1", "raw_ocr_text": "r1"},
        {"record_id": "b", "prompt": "p2", "input_text": "i2", "raw_ocr_text": "r2"},
        {"record_id": "c", "prompt": "p3", "input_text": "i3", "raw_ocr_text": "r3"},
    ]
    calls: list[list[str]] = []
    saved: dict[str, object] = {}

    class DummyProgress:
        def __init__(self) -> None:
            self.updated: list[int] = []
            self.closed = False

        def update(self, count: int) -> None:
            self.updated.append(count)

        def close(self) -> None:
            self.closed = True

    progress = DummyProgress()

    monkeypatch.setattr(cli, "_filter_task_rows", lambda processed_dir, task, fold, split: rows)
    monkeypatch.setattr(cli, "_progress_bar", lambda total, task, fold, split: progress)
    monkeypatch.setattr(inference, "load_generation_model", lambda model_dir: ("model", "tokenizer"))

    def fake_generate_texts(model, tokenizer, prompts, input_texts, task, max_new_tokens=None):
        calls.append(list(input_texts))
        return [f"pred-{text}" for text in input_texts]

    monkeypatch.setattr(inference, "generate_texts", fake_generate_texts)
    monkeypatch.setattr(inference, "normalize_detector_prediction", lambda prediction_text, fallback_raw_text: prediction_text)
    monkeypatch.setattr(inference, "write_predictions", lambda path, batch_rows: saved.setdefault("rows", batch_rows))

    cli.predict_fold(
        task="detector",
        model_dir=tmp_path,
        processed_dir=tmp_path,
        fold=0,
        split="test",
        input_override=None,
        output_path=tmp_path / "predictions.jsonl",
        max_new_tokens=None,
        batch_size=2,
        max_samples=None,
    )

    assert calls == [["i1", "i2"], ["i3"]]
    assert progress.updated == [2, 1]
    assert progress.closed is True
    assert [row["prediction_text"] for row in saved["rows"]] == ["pred-i1", "pred-i2", "pred-i3"]
