from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


OPEN_TAG = "<error>"
CLOSE_TAG = "</error>"


@dataclass(frozen=True)
class ErrorSpan:
    start: int
    end: int


def _merge_adjacent(spans: list[ErrorSpan]) -> list[ErrorSpan]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda span: (span.start, span.end))
    merged = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        if span.start <= prev.end:
            merged[-1] = ErrorSpan(prev.start, max(prev.end, span.end))
            continue
        if span.start == prev.end:
            merged[-1] = ErrorSpan(prev.start, span.end)
            continue
        merged.append(span)
    return merged


def error_spans_from_alignment(ocr_text: str, reference_text: str) -> list[ErrorSpan]:
    raw_spans: list[ErrorSpan] = []
    for tag, src_start, src_end, _dest_start, _dest_end in Levenshtein.opcodes(ocr_text, reference_text):
        if tag == "equal":
            continue
        if src_start != src_end:
            raw_spans.append(ErrorSpan(src_start, src_end))
            continue

        if not ocr_text:
            continue
        if src_start > 0:
            raw_spans.append(ErrorSpan(src_start - 1, src_start))
        else:
            raw_spans.append(ErrorSpan(0, 1))
    return _merge_adjacent(raw_spans)


def tag_error_spans(ocr_text: str, spans: list[ErrorSpan]) -> str:
    if not spans:
        return ocr_text
    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(ocr_text[cursor:span.start])
        parts.append(OPEN_TAG)
        parts.append(ocr_text[span.start:span.end])
        parts.append(CLOSE_TAG)
        cursor = span.end
    parts.append(ocr_text[cursor:])
    return "".join(parts)


def edit_only_tagged_text(ocr_text: str, spans: list[ErrorSpan]) -> str:
    if not spans:
        return ocr_text
    parts: list[str] = []
    cursor = 0
    for idx, span in enumerate(spans, start=1):
        parts.append(ocr_text[cursor:span.start])
        parts.append(f'<error id="{idx}">')
        parts.append(ocr_text[span.start:span.end])
        parts.append("</error>")
        cursor = span.end
    parts.append(ocr_text[cursor:])
    return "".join(parts)

