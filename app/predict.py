"""
Builds a single feature row for predicting an UPCOMING match — as opposed
to app.build_feature_table, which builds feature rows for historical
matches already in the database. The key structural difference: an
upcoming match has no match_id yet, so injury impact uses
get_current_injury_impact (a date-window lookup) instead of
get_injury_impact (which needs an existing match_id).

The returned dict's keys deliberately match the training feature columns
exactly (see app.build_feature_table.build_feature_row) so it can be fed
straight into a model trained on that feature table without any renaming.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy.orm import Session

from app.derbies import is_derby
from app.features import (
    get_current_elo_rating,
    get_current_injury_impact,
    get_head_to_head,
    get_rest_days,
    get_rolling_form,
    get_travel_distance_km,
)
from app.models import Team


def build_prediction_features(
    db: Session, home_team: Team, away_team: Team, match_date: Optional[dt.date] = None
) -> dict:
    """
    Returns a flat dict of feature values for a hypothetical match between
    home_team and away_team on match_date (defaults to today). None values
    for missing data (e.g. no stadium coordinates) are left as None —
    callers feeding this into a model should be aware XGBoost handles NaN
    natively, but other consumers may need to impute.
    """
    match_date = match_date or dt.date.today()

    home_form = get_rolling_form(db, home_team.id, match_date)
    away_form = get_rolling_form(db, away_team.id, match_date)
    h2h = get_head_to_head(db, home_team.id, away_team.id, match_date)
    home_injuries = get_current_injury_impact(db, home_team.id, match_date)
    away_injuries = get_current_injury_impact(db, away_team.id, match_date)

    return {
        "home_elo": get_current_elo_rating(db, home_team.id, match_date),
        "away_elo": get_current_elo_rating(db, away_team.id, match_date),
        "league": home_team.league or "Unknown",
        "home_rest_days": get_rest_days(db, home_team.id, match_date),
        "away_rest_days": get_rest_days(db, away_team.id, match_date),
        "home_win_rate_last5": home_form["win_rate"],
        "home_avg_goals_scored_last5": home_form["avg_goals_scored"],
        "home_avg_goals_conceded_last5": home_form["avg_goals_conceded"],
        "away_win_rate_last5": away_form["win_rate"],
        "away_avg_goals_scored_last5": away_form["avg_goals_scored"],
        "away_avg_goals_conceded_last5": away_form["avg_goals_conceded"],
        "h2h_home_win_rate": h2h["h2h_win_rate"],
        "h2h_avg_goal_diff": h2h["h2h_avg_goal_diff"],
        "h2h_matches_played": h2h["h2h_matches_played"],
        "travel_distance_km": get_travel_distance_km(home_team, away_team, match=None),
        "is_derby": is_derby(home_team.name, away_team.name),
        "home_injury_count": home_injuries["injury_count"],
        "home_injury_importance_sum": home_injuries["injury_importance_sum"],
        "away_injury_count": away_injuries["injury_count"],
        "away_injury_importance_sum": away_injuries["injury_importance_sum"],
    }
