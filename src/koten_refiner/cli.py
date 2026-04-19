from __future__ import annotations

from pathlib import Path

import orjson
import typer

from koten_refiner.dataset_builder import build_experiment_records, build_fold_map, discover_page_records, write_jsonl
from koten_refiner.models import PageRecord, record_id_for

app = typer.Typer(help="Classical Japanese OCR error correction experiments")


def _default_dataset_dir() -> Path:
    return Path("datasets")


def _default_processed_dir() -> Path:
    return Path("data/processed")


@app.command("prepare-data")
def prepare_data(
    dataset_dir: Path = typer.Option(_default_dataset_dir(), exists=True, file_okay=False),
    output_dir: Path = typer.Option(_default_processed_dir(), file_okay=False),
    seed: int = typer.Option(42),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_page_records(dataset_dir)
    fold_map = build_fold_map(records, seed=seed)
    experiment_records = build_experiment_records(records, fold_map)

    write_jsonl(output_dir / "pages.jsonl", (record.model_dump() for record in records))
    write_jsonl(output_dir / "folds.jsonl", ({"record_id": rid, **info} for rid, info in fold_map.items()))
    write_jsonl(output_dir / "experiments.jsonl", (record.model_dump() for record in experiment_records))

    for task in ("detector", "corrector", "one_stage", "edit_only"):
        task_rows = [row.model_dump() for row in experiment_records if row.task == task]
        write_jsonl(output_dir / f"{task}.jsonl", task_rows)
    typer.echo(f"Prepared {len(records)} pages and {len(experiment_records)} task rows in {output_dir}")


def _filter_task_rows(processed_dir: Path, task: str, fold: int, split: str) -> list[dict]:
    rows: list[dict] = []
    with (processed_dir / f"{task}.jsonl").open("rb") as handle:
        for line in handle:
            row = orjson.loads(line)
            if row["fold"] == fold and row["split"] == split:
                rows.append(row)
    return rows


def _limit_rows(rows: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None or max_samples <= 0:
        return rows
    return rows[:max_samples]


def _rows_by_record_id(rows: list[dict]) -> dict[str, dict]:
    return {row["record_id"]: row for row in rows}


@app.command("train-detector")
def train_detector(
    processed_dir: Path = typer.Option(_default_processed_dir(), exists=True, file_okay=False),
    config_path: Path = typer.Option(Path("configs/detector.yaml"), exists=True, dir_okay=False),
    fold: int = typer.Option(0),
    output_dir: Path = typer.Option(Path("results/detector"), file_okay=False),
    max_samples: int | None = typer.Option(None),
) -> None:
    from koten_refiner.train import load_yaml_config, train_with_unsloth

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _limit_rows(_filter_task_rows(processed_dir, "detector", fold, "train"), max_samples)
    train_path = output_dir / f"detector_fold{fold}_train.jsonl"
    write_jsonl(train_path, rows)
    config = load_yaml_config(config_path)
    train_with_unsloth(train_path, output_dir / f"fold_{fold}", config)


@app.command("train-corrector")
def train_corrector(
    processed_dir: Path = typer.Option(_default_processed_dir(), exists=True, file_okay=False),
    config_path: Path = typer.Option(Path("configs/corrector.yaml"), exists=True, dir_okay=False),
    fold: int = typer.Option(0),
    task: str = typer.Option("corrector"),
    output_dir: Path = typer.Option(Path("results/corrector"), file_okay=False),
    max_samples: int | None = typer.Option(None),
) -> None:
    from koten_refiner.train import load_yaml_config, train_with_unsloth

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _limit_rows(_filter_task_rows(processed_dir, task, fold, "train"), max_samples)
    train_path = output_dir / f"{task}_fold{fold}_train.jsonl"
    write_jsonl(train_path, rows)
    config = load_yaml_config(config_path)
    train_with_unsloth(train_path, output_dir / f"{task}_fold_{fold}", config)


@app.command("eval-cv")
def eval_cv(processed_dir: Path = typer.Option(_default_processed_dir(), exists=True, file_okay=False)) -> None:
    counts: dict[str, int] = {}
    with (processed_dir / "experiments.jsonl").open("rb") as handle:
        for line in handle:
            row = orjson.loads(line)
            key = f"{row['task']}:{row['split']}:fold{row['fold']}"
            counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        typer.echo(f"{key}\t{counts[key]}")


@app.command("prepare-corrector-test")
def prepare_corrector_test(
    processed_dir: Path = typer.Option(_default_processed_dir(), exists=True, file_okay=False),
    detector_predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    fold: int = typer.Option(0),
    task: str = typer.Option("corrector"),
    output_path: Path = typer.Option(Path("data/processed/corrector_test_predicted.jsonl"), dir_okay=False),
) -> None:
    from koten_refiner.evaluation import load_jsonl
    from koten_refiner.inference import normalize_detector_prediction

    detector_rows = load_jsonl(detector_predictions)
    detector_by_id = _rows_by_record_id(detector_rows)
    base_rows = _filter_task_rows(processed_dir, task, fold, "test")
    patched: list[dict] = []
    for row in base_rows:
        detector_row = detector_by_id.get(row["record_id"])
        if detector_row is None:
            continue
        tagged_prediction = normalize_detector_prediction(detector_row["prediction_text"], row["raw_ocr_text"])
        row = dict(row)
        row["tagged_ocr_text"] = tagged_prediction
        row["input_text"] = f"{tagged_prediction}<sep>{row['raw_ocr_text']}"
        row["metadata"] = {**row.get("metadata", {}), "tag_source": "detector_prediction"}
        patched.append(row)
    write_jsonl(output_path, patched)
    typer.echo(f"Wrote {len(patched)} rows to {output_path}")


@app.command("predict-fold")
def predict_fold(
    task: str = typer.Option(...),
    model_dir: Path = typer.Option(..., exists=True, file_okay=False),
    processed_dir: Path = typer.Option(_default_processed_dir(), exists=True, file_okay=False),
    fold: int = typer.Option(0),
    split: str = typer.Option("test"),
    input_override: Path | None = typer.Option(None, exists=True, dir_okay=False),
    output_path: Path = typer.Option(Path("results/predictions.jsonl"), dir_okay=False),
    max_new_tokens: int = typer.Option(512),
    max_samples: int | None = typer.Option(None),
) -> None:
    from koten_refiner.evaluation import load_jsonl
    from koten_refiner.inference import (
        apply_edit_only_prediction,
        generate_text,
        load_generation_model,
        normalize_detector_prediction,
        write_predictions,
    )

    if input_override is not None:
        rows = load_jsonl(input_override)
    else:
        rows = _filter_task_rows(processed_dir, task, fold, split)
    rows = _limit_rows(rows, max_samples)
    model, tokenizer = load_generation_model(model_dir)
    predictions: list[dict] = []
    for row in rows:
        prediction_text = generate_text(model, tokenizer, row["prompt"], row["input_text"], max_new_tokens=max_new_tokens)
        restored_text = prediction_text
        if task == "edit_only":
            restored_text = apply_edit_only_prediction(row["input_text"], prediction_text)
        elif task == "detector":
            restored_text = normalize_detector_prediction(prediction_text, row["raw_ocr_text"])
        predictions.append(
            {
                **row,
                "prediction_text": prediction_text,
                "restored_text": restored_text,
                "model_dir": str(model_dir),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output_path, predictions)
    typer.echo(f"Wrote {len(predictions)} predictions to {output_path}")


@app.command("evaluate-predictions")
def evaluate_predictions(
    task: str = typer.Option(...),
    predictions_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_path: Path = typer.Option(Path("results/metrics.json"), dir_okay=False),
    detector_eval_unit: str = typer.Option("char"),
    detector_tokenizer_model: str = typer.Option("rinna/japanese-roberta-base"),
    detector_char_chunk_size: int = typer.Option(256),
) -> None:
    if task == "detector":
        from koten_refiner.detector_evaluation import (
            compute_detector_metrics_from_path,
            load_detector_tokenizer,
            write_json,
        )

        if detector_eval_unit == "char":
            metrics = compute_detector_metrics_from_path(predictions_path)
        elif detector_eval_unit == "subtoken":
            tokenizer = load_detector_tokenizer(detector_tokenizer_model)
            metrics = compute_detector_metrics_from_path(
                predictions_path,
                tokenizer=tokenizer,
                char_chunk_size=detector_char_chunk_size,
            )
        else:
            raise typer.BadParameter("detector-eval-unit must be 'char' or 'subtoken'")
    else:
        from koten_refiner.evaluation import compute_generation_metrics_from_path, write_json

        metrics = compute_generation_metrics_from_path(predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, metrics)
    typer.echo(orjson.dumps(metrics, option=orjson.OPT_INDENT_2).decode())


@app.command("summarize-metrics")
def summarize_metrics(
    metrics_paths: list[Path] = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Option(Path("results/summary.json"), dir_okay=False),
) -> None:
    from koten_refiner.evaluation import summarize_metric_files, write_json, write_summary_csv

    summary = summarize_metric_files(metrics_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary)
    write_summary_csv(output_path.with_suffix(".csv"), summary)
    typer.echo(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())


@app.command("run-improvement")
def run_improvement(
    processed_dir: Path = typer.Option(_default_processed_dir(), exists=True, file_okay=False),
    config_path: Path = typer.Option(Path("configs/edit_only.yaml"), exists=True, dir_okay=False),
    fold: int = typer.Option(0),
    output_dir: Path = typer.Option(Path("results/improvement"), file_okay=False),
    max_samples: int | None = typer.Option(None),
) -> None:
    from koten_refiner.train import load_yaml_config, train_with_unsloth

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _limit_rows(_filter_task_rows(processed_dir, "edit_only", fold, "train"), max_samples)
    train_path = output_dir / f"edit_only_fold{fold}_train.jsonl"
    write_jsonl(train_path, rows)
    config = load_yaml_config(config_path)
    train_with_unsloth(train_path, output_dir / f"edit_only_fold_{fold}", config)


@app.command("export-fold")
def export_fold(
    task: str = typer.Option(...),
    processed_dir: Path = typer.Option(_default_processed_dir(), exists=True, file_okay=False),
    fold: int = typer.Option(0),
    split: str = typer.Option("train"),
    output_path: Path = typer.Option(..., dir_okay=False),
    max_samples: int | None = typer.Option(None),
) -> None:
    rows = _limit_rows(_filter_task_rows(processed_dir, task, fold, split), max_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    typer.echo(f"Wrote {len(rows)} rows to {output_path}")


@app.command("inspect-record")
def inspect_record(
    processed_dir: Path = typer.Option(_default_processed_dir(), exists=True, file_okay=False),
    work_id: str = typer.Argument(...),
    page_number: int = typer.Argument(...),
) -> None:
    target = record_id_for(work_id, page_number)
    with (processed_dir / "pages.jsonl").open("rb") as handle:
        for line in handle:
            row = PageRecord.model_validate(orjson.loads(line))
            if record_id_for(row.work_id, row.page_number) == target:
                typer.echo(orjson.dumps(row.model_dump(), option=orjson.OPT_INDENT_2).decode())
                return
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
