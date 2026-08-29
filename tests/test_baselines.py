"""
Tests for app.baselines.
"""

import pandas as pd
import pytest

from app.baselines import (
    OUTCOME_LABELS,
    elo_baseline_probabilities,
    estimate_draw_probability,
    naive_baseline_probabilities,
)


def test_elo_baseline_probabilities_sum_to_one():
    probs = elo_baseline_probabilities(1600, 1500)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_elo_baseline_favors_stronger_home_team():
    probs = elo_baseline_probabilities(1700, 1400)
    assert probs["H"] > probs["A"]


def test_elo_baseline_favors_stronger_away_team():
    probs = elo_baseline_probabilities(1400, 1700, home_advantage=0)
    assert probs["A"] > probs["H"]


def test_elo_baseline_equal_ratings_with_home_advantage_favors_home():
    probs = elo_baseline_probabilities(1500, 1500, home_advantage=70)
    assert probs["H"] > probs["A"]


def test_elo_baseline_draw_prob_is_respected():
    probs = elo_baseline_probabilities(1500, 1500, draw_prob=0.3)
    assert probs["D"] == pytest.approx(0.3)


def test_elo_baseline_all_probabilities_non_negative():
    # Even a huge rating gap shouldn't push any probability below zero.
    probs = elo_baseline_probabilities(2200, 1000, draw_prob=0.25)
    assert all(p >= 0 for p in probs.values())


def test_estimate_draw_probability():
    df = pd.DataFrame({"result": ["H", "D", "A", "D", "H", "D", "D"]})
    assert estimate_draw_probability(df) == pytest.approx(4 / 7)


def test_naive_baseline_matches_empirical_distribution():
    df = pd.DataFrame({"result": ["H", "H", "H", "D", "A"]})
    probs = naive_baseline_probabilities(df)
    assert probs["H"] == pytest.approx(0.6)
    assert probs["D"] == pytest.approx(0.2)
    assert probs["A"] == pytest.approx(0.2)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_naive_baseline_handles_missing_outcome_class():
    # No draws at all in this training set — should return 0.0 for 'D', not KeyError.
    df = pd.DataFrame({"result": ["H", "H", "A"]})
    probs = naive_baseline_probabilities(df)
    assert probs["D"] == 0.0
    assert set(probs.keys()) == set(OUTCOME_LABELS)
