"""Shared fixture builders for database synchronization tests."""

import pandas as pd

from app.data_sources.footballdata_uk import CANONICAL_COLUMNS


def fixture_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in CANONICAL_COLUMNS:
        if column not in frame:
            frame[column] = float("nan")
    return frame[CANONICAL_COLUMNS]