"""
Baseline predictors — the "floor" any real model must clearly beat to be
worth using. Two baselines:

1. elo_baseline_probabilities() — turns an Elo rating gap into H/D/A
   probabilities using a heuristic draw-probability split. This is NOT a
   rigorous derivation (Elo's expected_score doesn't cleanly decompose into
   three outcomes) — it's a reasonable, simple baseline, not a claim of
   statistical correctness.
2. naive_baseline_probabilities() — the empirical outcome distribution from
   the training set (e.g. "historically 45% of matches are home wins"),
   predicted identically for every match regardless of the two teams. If
   your trained model can't beat this, something is wrong with the model
   or the features, not with your ambitions.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from app.elo import expected_score

OUTCOME_LABELS = ["A", "D", "H"]  # alphabetical order, used consistently for encoding throughout this module


def elo_baseline_probabilities(
    home_elo: float, away_elo: float, home_advantage: float = 70.0, draw_prob: float = 0.25
) -> Dict[str, float]:
    """
    Splits the non-draw probability mass between home/away according to
    Elo's expected_score, with a fixed draw probability (pass the
    training set's empirical draw rate for a more grounded value — see
    estimate_draw_probability below).
    """
    expected_home = expected_score(home_elo + home_advantage, away_elo)
    non_draw = 1.0 - draw_prob
    return {
        "H": non_draw * expected_home,
        "D": draw_prob,
        "A": non_draw * (1.0 - expected_home),
    }


def estimate_draw_probability(train_df: pd.DataFrame) -> float:
    """Empirical draw rate from the training set — use this instead of a hardcoded guess."""
    return float((train_df["result"] == "D").mean())


def naive_baseline_probabilities(train_df: pd.DataFrame) -> Dict[str, float]:
    """
    The training set's empirical outcome distribution (e.g. {'H': 0.45, 'D': 0.25, 'A': 0.30}),
    to be predicted identically for every match in validation/test.
    """
    counts = train_df["result"].value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0)) for label in OUTCOME_LABELS}
