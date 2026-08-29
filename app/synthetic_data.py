"""
Generates a synthetic feature table shaped exactly like the real output of
app.build_feature_table, for testing app.train_model's pipeline logic
without needing hundreds of real historical matches.

The outcome is deliberately generated with a real (noisy) dependence on
elo_diff and home advantage, so a correctly-implemented training pipeline
should be able to beat the naive baseline on this data — that's the
property tests in test_train_model.py check for. This is NOT meant to
resemble real football data closely; it exists purely to exercise the code.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def generate_synthetic_feature_table(n_matches: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    team_names = [f"Team {i}" for i in range(20)]
    start_date = dt.date(2018, 8, 1)

    rows = []
    for i in range(n_matches):
        home_team, away_team = rng.choice(team_names, size=2, replace=False)
        home_elo = rng.normal(1500, 120)
        away_elo = rng.normal(1500, 120)
        elo_diff = home_elo - away_elo + 70  # +70 home advantage baked into the generative process

        # True win probability follows a logistic function of elo_diff, so a model
        # that learns this relationship should clearly beat the naive baseline.
        p_home_raw = 1 / (1 + np.exp(-elo_diff / 200))
        p_draw = 0.25
        p_home = (1 - p_draw) * p_home_raw
        p_away = (1 - p_draw) * (1 - p_home_raw)

        result = rng.choice(["H", "D", "A"], p=[p_home, p_draw, p_away])
        home_score, away_score = _score_from_result(rng, result)

        rows.append({
            "match_id": i,
            "date": start_date + dt.timedelta(days=int(i * 2.5)),  # roughly spreads matches over ~4 seasons
            "home_team": home_team,
            "away_team": away_team,
            "competition": "Synthetic League",
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_rest_days": rng.integers(3, 14),
            "away_rest_days": rng.integers(3, 14),
            "home_win_rate_last5": rng.uniform(0, 1),
            "home_avg_goals_scored_last5": rng.uniform(0.5, 2.5),
            "home_avg_goals_conceded_last5": rng.uniform(0.5, 2.5),
            "away_win_rate_last5": rng.uniform(0, 1),
            "away_avg_goals_scored_last5": rng.uniform(0.5, 2.5),
            "away_avg_goals_conceded_last5": rng.uniform(0.5, 2.5),
            "h2h_home_win_rate": rng.uniform(0, 1),
            "h2h_avg_goal_diff": rng.normal(0, 1),
            "h2h_matches_played": rng.integers(0, 6),
            "travel_distance_km": rng.uniform(5, 500),
            "venue_name": f"{home_team} (default stadium)",
            "is_derby": bool(rng.random() < 0.1),
            "home_injury_count": rng.integers(0, 4),
            "home_injury_importance_sum": rng.uniform(0, 2),
            "away_injury_count": rng.integers(0, 4),
            "away_injury_importance_sum": rng.uniform(0, 2),
            "home_score": home_score,
            "away_score": away_score,
            "result": result,
        })

    return pd.DataFrame(rows)


def _score_from_result(rng: np.random.Generator, result: str):
    if result == "H":
        home = rng.integers(1, 4)
        away = rng.integers(0, home)
    elif result == "A":
        away = rng.integers(1, 4)
        home = rng.integers(0, away)
    else:
        goals = rng.integers(0, 3)
        home = away = goals
    return int(home), int(away)
