import datetime as dt
from unittest.mock import patch

import pandas as pd

from app.train_ensemble import train
from app.sync_data import _history_seasons


def test_all_sync_defaults_to_eight_seasons_but_explicit_value_wins():
    assert _history_seasons(None, all_sources=True) == 8
    assert _history_seasons(None, all_sources=False) == 1
    assert _history_seasons(2, all_sources=True) == 2


def test_small_single_season_dataset_does_not_replace_production_bundle(tmp_path):
    model_path = tmp_path / "model_bundle.joblib"
    model_path.write_bytes(b"existing production model")
    frame = pd.DataFrame({
        "played": [True] * 518,
        "season": [2026] * 518,
        "date": [dt.date(2026, 8, 1)] * 518,
        "match_id": list(range(518)),
        "league": ["Premier League"] * 518,
    })

    with patch("app.train_ensemble.build_feature_frame", return_value=frame):
        result = train(None, model_out=str(model_path))

    assert result is None
    assert model_path.read_bytes() == b"existing production model"