from __future__ import annotations

from pathlib import Path

from koten_refiner.dataset_builder import (
    build_experiment_records,
    build_fold_map,
    human_page_key_from_path,
    ndl_page_key_from_path,
    reconstruct_human_json_text,
    reconstruct_human_json_text_geometric,
    reconstruct_ndl_ocr_text,
)
from koten_refiner.models import PageRecord, record_id_for


def make_page_record(
    work_id: str = "200003425",
    page_number: int = 1,
    ocr_text: str = "今日は天期です",
    reference_text: str = "今日は天気です",
) -> PageRecord:
    return PageRecord(
        work_id=work_id,
        page_number=page_number,
        human_source="humanA",
        ocr_path="datasets/ndl/200003425/json/200003425-00001.json",
        human_text_path="datasets/humanA/200003425/text/200003425_0001_qc.txt",
        human_json_path="datasets/humanA/200003425/json/200003425_0001_qc.json",
        ocr_text=ocr_text,
        reference_text=reference_text,
    )


def test_page_key_normalization_matches_between_sources():
    ndl = Path("datasets/ndl/200003425/json/200003425-00001.json")
    human = Path("datasets/humanA/200003425/text/200003425_0001_qc.txt")
    assert ndl_page_key_from_path(ndl) == human_page_key_from_path(human)


def test_reconstruct_ndl_ocr_text_handles_empty_single_and_multiple_contents():
    assert reconstruct_ndl_ocr_text({"contents": []}) == ""
    assert reconstruct_ndl_ocr_text({"contents": [[0, 0, 1, 1, "単独"]]} ) == "単独"
    assert reconstruct_ndl_ocr_text({"contents": [[0, 0, 1, 1, "一行"], [0, 0, 1, 1, "二行"]]}) == "一行\n二行"


def test_saved_order_is_not_worse_than_geometric_sort_for_human_json():
    human_obj = {
        "rects": [
            {"x": 0, "y": 0, "str": "左列"},
            {"x": 100, "y": 0, "str": "右列"},
        ]
    }
    reference = "左列\n右列"
    saved = reconstruct_human_json_text(human_obj)
    geometric = reconstruct_human_json_text_geometric(human_obj)
    assert saved == reference
    assert geometric != reference


def test_build_experiment_records_generates_paper_format_corrector_input():
    record = make_page_record()
    rid = record_id_for(record.work_id, record.page_number)
    fold_map = {f"{rid}|fold=0": {"record_id": rid, "fold": 0, "split": "train"}}
    rows = build_experiment_records([record], fold_map)
    corrector_row = next(row for row in rows if row.task == "corrector")
    assert corrector_row.input_text == f"{corrector_row.tagged_ocr_text}<sep>{record.ocr_text}"
    assert corrector_row.metadata["tag_source"] == "oracle"


def test_build_fold_map_is_reproducible_with_same_seed():
    records = [make_page_record(page_number=idx, ocr_text=f"ocr{idx}", reference_text=f"ref{idx}") for idx in range(1, 11)]
    fold_map_a = build_fold_map(records, seed=42)
    fold_map_b = build_fold_map(records, seed=42)
    assert fold_map_a == fold_map_b
