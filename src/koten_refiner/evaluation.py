from __future__ import annotations

from pathlib import Path

import orjson

from koten_refiner.metrics import compute_text_metrics


def compute_generation_metrics(rows: list[dict]) -> dict[str, float]:
    metrics = [compute_text_metrics(row["restored_text"], row["reference_text"]) for row in rows]
    if not metrics:
        return {"bleu": 0.0, "crr": 0.0, "wrr": 0.0}
    return {
        key: sum(item[key] for item in metrics) / len(metrics)
        for key in ("bleu", "crr", "wrr")
    }


def load_jsonl(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        return [orjson.loads(line) for line in handle]


def write_json(path: Path, payload: dict) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def summarize_metric_files(paths: list[Path]) -> dict[str, float]:
    metrics = [orjson.loads(path.read_bytes()) for path in paths]
    if not metrics:
        return {}
    keys = sorted(metrics[0].keys())
    return {
        key: sum(float(item[key]) for item in metrics) / len(metrics)
        for key in keys
    }


def write_summary_csv(path: Path, payload: dict[str, float]) -> None:
    lines = ["metric,value"]
    for key, value in payload.items():
        lines.append(f"{key},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            yield orjson.loads(line)


def compute_generation_metrics_from_path(path: Path) -> dict[str, float]:
    total = {"bleu": 0.0, "crr": 0.0, "wrr": 0.0}
    count = 0
    for row in iter_jsonl(path):
        metrics = compute_text_metrics(row["restored_text"], row["reference_text"])
        for key in total:
            total[key] += metrics[key]
        count += 1
    if count == 0:
        return total
    return {key: value / count for key, value in total.items()}
