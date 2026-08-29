"""
Time-based train/validation/test split.

Deliberately NOT a random split. A random split lets the model train on
matches from March 2024 and get validated on matches from January 2024 —
information from the "future" (relative to the validation point) leaks in
via engineered features like rolling form and Elo, making validation
scores look better than real-world performance ever will be.

Splitting by date instead means: train on everything up to some date,
validate on the next chunk, test on the most recent chunk — exactly
mimicking how the model will actually be used (predicting matches that
haven't happened yet, using only what's known so far).
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame (must have a 'date' column) into chronological
    train/val/test sets by row position after sorting by date — NOT by
    calendar date directly, since match density varies. The remaining
    fraction after train_frac + val_frac becomes the test set.

    Ties on the exact same date are kept together on whichever side the
    row count naturally falls — acceptable here since date-level ties
    (multiple matches on one matchday) don't meaningfully leak information
    about each other in the way that ordering across days would.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must each be in (0, 1) and sum to less than 1")

    sorted_df = df.sort_values("date").reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]

    return train_df, val_df, test_df


def time_based_split_grouped(
    df: pd.DataFrame, group_col: str = "league", train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Like time_based_split, but applies the chronological split SEPARATELY
    within each group (e.g. each league), then concatenates the results.

    Why this exists: a single global chronological cut can accidentally
    exclude an entire group from training if that group's matches happen
    to cluster in a particular time window relative to the other groups.
    Concretely: if league A spans 2015-2023 and league B only has matches
    in 2024-2026, a global 70/15/15 cut by row position can put 100% of
    league B into the test set and 0% into training — the model never
    sees a single league B example to learn from, and any categorical
    feature distinguishing the groups gets zero importance because it had
    no variance in training. Splitting per-group first prevents this: each
    group gets its own fair 70/15/15 share of ITS OWN history.

    Within each group, ordering is still strictly chronological (no
    leakage) — this only changes which groups get represented in each
    split, not the temporal integrity within a group.
    """
    train_parts, val_parts, test_parts = [], [], []

    for group_value, group_df in df.groupby(group_col):
        if len(group_df) < 3:
            # Too few rows to meaningfully split three ways — put them all
            # in train rather than silently dropping them or creating an
            # empty-but-present val/test slice for this group.
            train_parts.append(group_df.sort_values("date"))
            continue
        g_train, g_val, g_test = time_based_split(group_df, train_frac=train_frac, val_frac=val_frac)
        train_parts.append(g_train)
        val_parts.append(g_val)
        test_parts.append(g_test)

    def _combine(parts):
        if not parts:
            return df.iloc[0:0]  # empty DataFrame with the same columns
        return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)

    return _combine(train_parts), _combine(val_parts), _combine(test_parts)


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    def date_range(d: pd.DataFrame) -> str:
        if len(d) == 0:
            return "empty"
        return f"{d['date'].min()} to {d['date'].max()}"

    print(f"  Train: {len(train_df):5d} matches ({date_range(train_df)})")
    print(f"  Val:   {len(val_df):5d} matches ({date_range(val_df)})")
    print(f"  Test:  {len(test_df):5d} matches ({date_range(test_df)})")
