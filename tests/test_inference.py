from koten_refiner.inference import apply_edit_only_prediction


def test_apply_edit_only_prediction_keeps_untouched_text():
    tagged = '今日は<error id="1">天期</error>が<error id="2">良</error>い'
    pred = "1\t天気\n2\t<KEEP>"
    restored = apply_edit_only_prediction(tagged, pred)
    assert restored == "今日は天気が良い"


def test_apply_edit_only_prediction_handles_multiple_spans_and_missing_updates():
    tagged = 'A<error id="1">B</error>C<error id="2">D</error>E<error id="3">F</error>'
    pred = "1\tX\n2\t<KEEP>"
    restored = apply_edit_only_prediction(tagged, pred)
    assert restored == "AXCDEF"
