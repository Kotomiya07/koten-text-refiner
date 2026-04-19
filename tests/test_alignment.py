from koten_refiner.alignment import error_spans_from_alignment, tag_error_spans


def test_insert_at_beginning_expands_forward():
    spans = error_spans_from_alignment("ABC", "XABC")
    assert spans[0].start == 0
    assert spans[0].end == 1


def test_tags_replace_span():
    ocr = "今日は天期です"
    ref = "今日は天気です"
    tagged = tag_error_spans(ocr, error_spans_from_alignment(ocr, ref))
    assert "<error>" in tagged
    assert "</error>" in tagged


def test_equal_text_has_no_error_spans():
    assert error_spans_from_alignment("同じ文", "同じ文") == []


def test_insert_in_middle_marks_previous_character():
    spans = error_spans_from_alignment("ABCD", "ABXCD")
    assert [(span.start, span.end) for span in spans] == [(1, 2)]


def test_adjacent_replace_and_delete_are_merged():
    spans = error_spans_from_alignment("ABCDE", "AXE")
    assert [(span.start, span.end) for span in spans] == [(1, 4)]
