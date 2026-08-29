"""
Tests for app.data_split — the most important property being tested is
that splits are chronological, not random: every date in train must be
earlier than every date in val, which must be earlier than every date in test.
"""

import datetime as dt

import pandas as pd
import pytest

from app.data_split import time_based_split, time_based_split_grouped


def _make_df(n_rows: int, shuffle: bool = False) -> pd.DataFrame:
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n_rows)]
    df = pd.DataFrame({"date": dates, "value": range(n_rows)})
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def test_split_sizes_approximately_match_fractions():
    df = _make_df(1000)
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150


def test_split_is_chronological_train_before_val_before_test():
    df = _make_df(100)
    train, val, test = time_based_split(df, train_frac=0.6, val_frac=0.2)

    assert train["date"].max() <= val["date"].min()
    assert val["date"].max() <= test["date"].min()


def test_split_works_even_when_input_is_shuffled():
    """The function must sort internally — a shuffled input must still produce a chronological split."""
    df = _make_df(100, shuffle=True)
    train, val, test = time_based_split(df, train_frac=0.6, val_frac=0.2)

    assert train["date"].max() <= val["date"].min()
    assert val["date"].max() <= test["date"].min()


def test_split_no_rows_lost_or_duplicated():
    df = _make_df(137)  # deliberately not evenly divisible
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)

    assert len(train) + len(val) + len(test) == 137
    combined_values = set(train["value"]) | set(val["value"]) | set(test["value"])
    assert combined_values == set(range(137))


def test_split_rejects_invalid_fractions():
    df = _make_df(10)
    with pytest.raises(ValueError):
        time_based_split(df, train_frac=0.8, val_frac=0.3)  # sums to > 1

    with pytest.raises(ValueError):
        time_based_split(df, train_frac=1.5, val_frac=0.1)  # out of (0,1) range


# ---------------------------------------------------------------------------
# Grouped split (the fix for a group being entirely excluded from training)
# ---------------------------------------------------------------------------
def _make_grouped_df(n_rows_a, n_rows_b, group_a_start, group_b_start):
    """Group A spans a long early period; group B is clustered at the tail — mirrors the real EPL/KPL scenario."""
    dates_a = [group_a_start + dt.timedelta(days=i) for i in range(n_rows_a)]
    dates_b = [group_b_start + dt.timedelta(days=i) for i in range(n_rows_b)]
    df = pd.DataFrame({
        "date": dates_a + dates_b,
        "league": ["A"] * n_rows_a + ["B"] * n_rows_b,
        "value": list(range(n_rows_a + n_rows_b)),
    })
    return df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle to prove sorting isn't assumed


def test_grouped_split_reproduces_the_real_bug_with_plain_split():
    """Sanity check: confirms plain time_based_split DOES exclude group B
    entirely when it's clustered at the tail — proving the grouped version
    is solving a real, reproducible problem, not a hypothetical one."""
    df = _make_grouped_df(
        n_rows_a=700, n_rows_b=100,
        group_a_start=dt.date(2015, 1, 1), group_b_start=dt.date(2024, 1, 1),
    )
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    assert "B" not in train["league"].values  # confirms the bug is real and reproducible


def test_grouped_split_includes_every_group_in_every_split():
    df = _make_grouped_df(
        n_rows_a=700, n_rows_b=100,
        group_a_start=dt.date(2015, 1, 1), group_b_start=dt.date(2024, 1, 1),
    )
    train, val, test = time_based_split_grouped(df, group_col="league", train_frac=0.7, val_frac=0.15)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        assert "A" in split_df["league"].values, f"group A missing from {split_name}"
        assert "B" in split_df["league"].values, f"group B missing from {split_name}"


def test_grouped_split_is_chronological_within_each_group():
    df = _make_grouped_df(
        n_rows_a=700, n_rows_b=100,
        group_a_start=dt.date(2015, 1, 1), group_b_start=dt.date(2024, 1, 1),
    )
    train, val, test = time_based_split_grouped(df, group_col="league", train_frac=0.7, val_frac=0.15)

    for group in ["A", "B"]:
        train_dates = train[train["league"] == group]["date"]
        val_dates = val[val["league"] == group]["date"]
        test_dates = test[test["league"] == group]["date"]
        assert train_dates.max() <= val_dates.min()
        assert val_dates.max() <= test_dates.min()


def test_grouped_split_no_rows_lost_or_duplicated():
    df = _make_grouped_df(
        n_rows_a=700, n_rows_b=100,
        group_a_start=dt.date(2015, 1, 1), group_b_start=dt.date(2024, 1, 1),
    )
    train, val, test = time_based_split_grouped(df, group_col="league", train_frac=0.7, val_frac=0.15)

    assert len(train) + len(val) + len(test) == len(df)
    combined_values = set(train["value"]) | set(val["value"]) | set(test["value"])
    assert combined_values == set(range(800))


def test_grouped_split_tiny_group_goes_entirely_to_train():
    """A group with too few rows to split three ways (e.g. a brand-new
    league with 2 matches) should go entirely to train rather than being
    dropped or creating a broken empty split."""
    df = _make_grouped_df(n_rows_a=700, n_rows_b=2, group_a_start=dt.date(2015, 1, 1), group_b_start=dt.date(2024, 1, 1))
    train, val, test = time_based_split_grouped(df, group_col="league", train_frac=0.7, val_frac=0.15)

    assert (train["league"] == "B").sum() == 2
    assert (val["league"] == "B").sum() == 0
    assert (test["league"] == "B").sum() == 0
