from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


HumanSource = Literal["humanA", "humanB"]
TaskName = Literal["detector", "corrector", "one_stage", "edit_only"]


class PageRecord(BaseModel):
    work_id: str
    page_number: int
    human_source: HumanSource
    ocr_path: str
    human_text_path: str
    human_json_path: str
    ocr_text: str
    reference_text: str


class FoldAssignment(BaseModel):
    fold: int
    split: Literal["train", "test"]
    record_id: str


class ExperimentRecord(BaseModel):
    record_id: str
    work_id: str
    page_number: int
    human_source: HumanSource
    fold: int
    split: Literal["train", "test"]
    task: TaskName
    prompt: str
    input_text: str
    target_text: str
    raw_ocr_text: str
    tagged_ocr_text: str | None = None
    reference_text: str
    metadata: dict = Field(default_factory=dict)


def record_id_for(work_id: str, page_number: int) -> str:
    return f"{work_id}:{page_number:04d}"


def relative_to_cwd(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)

