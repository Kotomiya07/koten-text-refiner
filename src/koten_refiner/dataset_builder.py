from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import KFold

from koten_refiner.alignment import edit_only_tagged_text, error_spans_from_alignment, tag_error_spans
from koten_refiner.metrics import normalize_text
from koten_refiner.models import ExperimentRecord, PageRecord, record_id_for
from koten_refiner.prompts import CORRECTOR_PROMPT, DETECTOR_PROMPT, EDIT_ONLY_PROMPT, ONE_STAGE_PROMPT


def _json_load(path: Path) -> dict[str, Any]:
    return orjson.loads(path.read_bytes())


def _human_extension(source: str) -> str:
    return "txt" if source == "humanA" else "text"


def ndl_page_key_from_path(path: Path) -> tuple[str, int]:
    work_id, page = path.stem.split("-", 1)
    return work_id, int(page)


def human_page_key_from_path(path: Path) -> tuple[str, int]:
    work_id, page, _suffix = path.stem.split("_", 2)
    return work_id, int(page)


def reconstruct_ndl_ocr_text(ocr_obj: dict[str, Any]) -> str:
    return normalize_text("\n".join(item[-1] for item in ocr_obj.get("contents", [])))


def reconstruct_human_json_text(human_obj: dict[str, Any]) -> str:
    return normalize_text("\n".join(rect.get("str", "") for rect in human_obj.get("rects", [])))


def reconstruct_human_json_text_geometric(human_obj: dict[str, Any]) -> str:
    rects = sorted(
        human_obj.get("rects", []),
        key=lambda rect: (-int(rect.get("x", 0)), int(rect.get("y", 0))),
    )
    return normalize_text("\n".join(rect.get("str", "") for rect in rects))


def discover_page_records(dataset_dir: Path) -> list[PageRecord]:
    ndl_root = dataset_dir / "ndl"
    human_a_root = dataset_dir / "humanA"
    human_b_root = dataset_dir / "humanB"

    records: list[PageRecord] = []
    for ndl_work_dir in sorted(p for p in ndl_root.iterdir() if p.is_dir()):
        work_id = ndl_work_dir.name
        if (human_a_root / work_id).exists():
            human_source = "humanA"
        elif (human_b_root / work_id).exists():
            human_source = "humanB"
        else:
            continue

        human_work_dir = dataset_dir / human_source / work_id
        ext = _human_extension(human_source)
        for ocr_path in sorted((ndl_work_dir / "json").glob("*.json")):
            _ocr_work_id, page_number = ndl_page_key_from_path(ocr_path)
            human_text_path = human_work_dir / "text" / f"{work_id}_{page_number:04d}_qc.{ext}"
            human_json_path = human_work_dir / "json" / f"{work_id}_{page_number:04d}_qc.json"
            if not human_text_path.exists() or not human_json_path.exists():
                continue
            ocr_obj = _json_load(ocr_path)
            ocr_text = reconstruct_ndl_ocr_text(ocr_obj)
            reference_text = normalize_text(human_text_path.read_text())
            if not ocr_text or not reference_text:
                continue
            records.append(
                PageRecord(
                    work_id=work_id,
                    page_number=page_number,
                    human_source=human_source,
                    ocr_path=str(ocr_path),
                    human_text_path=str(human_text_path),
                    human_json_path=str(human_json_path),
                    ocr_text=ocr_text,
                    reference_text=reference_text,
                )
            )
    return records


def build_fold_map(records: list[PageRecord], seed: int = 42, n_splits: int = 5) -> dict[str, dict[str, int | str]]:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    index_to_id = [record_id_for(record.work_id, record.page_number) for record in records]
    fold_map: dict[str, dict[str, int | str]] = {}
    for fold, (train_indices, test_indices) in enumerate(kfold.split(records)):
        for split, indices in (("train", train_indices), ("test", test_indices)):
            for idx in indices:
                fold_map[f"{index_to_id[idx]}|fold={fold}"] = {
                    "record_id": index_to_id[idx],
                    "fold": fold,
                    "split": split,
                }
    return fold_map


def build_experiment_records(records: list[PageRecord], fold_map: dict[str, dict[str, int | str]]) -> list[ExperimentRecord]:
    output: list[ExperimentRecord] = []
    split_infos_by_record: dict[str, list[dict[str, int | str]]] = {}
    for info in fold_map.values():
        split_infos_by_record.setdefault(str(info["record_id"]), []).append(info)
    for record in records:
        rid = record_id_for(record.work_id, record.page_number)
        spans = error_spans_from_alignment(record.ocr_text, record.reference_text)
        tagged = tag_error_spans(record.ocr_text, spans)
        edit_only = edit_only_tagged_text(record.ocr_text, spans)
        for split_info in split_infos_by_record[rid]:
            common = {
                "record_id": rid,
                "work_id": record.work_id,
                "page_number": record.page_number,
                "human_source": record.human_source,
                "fold": int(split_info["fold"]),
                "split": str(split_info["split"]),
                "raw_ocr_text": record.ocr_text,
                "tagged_ocr_text": tagged,
                "reference_text": record.reference_text,
            }
            output.append(
                ExperimentRecord(
                    **common,
                    task="detector",
                    prompt=DETECTOR_PROMPT,
                    input_text=record.ocr_text,
                    target_text=tagged,
                    metadata={"num_error_spans": len(spans)},
                )
            )
            output.append(
                ExperimentRecord(
                    **common,
                    task="corrector",
                    prompt=CORRECTOR_PROMPT,
                    input_text=f"{tagged}<sep>{record.ocr_text}",
                    target_text=record.reference_text,
                    metadata={"num_error_spans": len(spans), "tag_source": "oracle"},
                )
            )
            output.append(
                ExperimentRecord(
                    **common,
                    task="one_stage",
                    prompt=ONE_STAGE_PROMPT,
                    input_text=record.ocr_text,
                    target_text=record.reference_text,
                    metadata={},
                )
            )
            output.append(
                ExperimentRecord(
                    **common,
                    task="edit_only",
                    prompt=EDIT_ONLY_PROMPT,
                    input_text=edit_only,
                    target_text=_edit_only_target(record.ocr_text, record.reference_text, spans),
                    metadata={"num_error_spans": len(spans)},
                )
            )
    return output


def _edit_only_target(ocr_text: str, reference_text: str, spans: list) -> str:
    if not spans:
        return ""
    opcodes = Levenshtein.opcodes(ocr_text, reference_text)
    lines: list[str] = []
    for idx, span in enumerate(spans, start=1):
        replacements: list[str] = []
        for tag, src_start, src_end, dest_start, dest_end in opcodes:
            if tag == "equal":
                continue
            overlaps = not (src_end <= span.start or src_start >= span.end)
            is_insert_at_start = src_start == src_end and span.start <= src_start <= span.end
            if overlaps or is_insert_at_start:
                replacements.append(reference_text[dest_start:dest_end])
        replacement = "".join(part for part in replacements if part)
        if not replacement:
            lines.append(f"{idx}\t<KEEP>")
        else:
            lines.append(f"{idx}\t{replacement}")
    return "\n".join(lines)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")
