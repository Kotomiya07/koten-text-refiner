from importlib.metadata import PackageNotFoundError
import warnings

import pytest

from koten_refiner.train import _make_model_card_generation_best_effort


class DummyTrainer:
    def __init__(self) -> None:
        self.calls = 0

    def create_model_card(self, *args, **kwargs):
        self.calls += 1
        raise PackageNotFoundError("trl")


def test_make_model_card_generation_best_effort_skips_missing_metadata_once():
    trainer = DummyTrainer()
    _make_model_card_generation_best_effort(trainer)

    with pytest.warns(UserWarning, match="Skipping model card generation"):
        assert trainer.create_model_card(model_name="demo") is None

    assert trainer.calls == 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert trainer.create_model_card(model_name="demo") is None

    assert trainer.calls == 2
    assert not record
