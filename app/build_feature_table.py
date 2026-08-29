"""
Assembles every feature from app.features (plus Elo and derby flags) into
one row per scored match — this is the feature table the XGBoost model in
the next phase will train on.

Usage:
    python -m app.build_feature_table                       # prints a preview
    python -m app.build_feature_table --output features.csv # writes to CSV

Each row is one historical match, with:
  - Elo ratings for both teams (pre-match)
  - Rest days for both teams
  - Rolling form (win rate, goals for/against) for both teams
  - Head-to-head record between the two teams
  - Travel distance for the away team
  - Derby flag
  - Injury impact for both teams
  - The target column: match_result (H / D / A)

Rows where a feature couldn't be computed (e.g. missing stadium
coordinates, or a team's very first match with no rolling form yet) still
get a row, with sensible defaults/None — filter or impute those as you see
fit once you're in the modeling stage; that decision belongs there, not
baked silently into this script.
"""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.derbies import is_derby
from app.features import (
    get_elo_rating,
    get_head_to_head,
    get_injury_impact,
    get_rest_days,
    get_rolling_form,
    get_travel_distance_km,
)
from app.models import Match


def build_feature_row(db: Session, match: Match) -> dict:
    home, away = match.home_team, match.away_team

    home_form = get_rolling_form(db, home.id, match.date)
    away_form = get_rolling_form(db, away.id, match.date)
    h2h = get_head_to_head(db, home.id, away.id, match.date)
    home_injuries = get_injury_impact(db, home.id, match.id)
    away_injuries = get_injury_impact(db, away.id, match.id)

    if match.home_score > match.away_score:
        result = "H"
    elif match.home_score < match.away_score:
        result = "A"
    else:
        result = "D"

    return {
        "match_id": match.id,
        "date": match.date,
        "home_team": home.name,
        "away_team": away.name,
        "competition": match.competition,
        "league": home.league or "Unknown",  # a genuine model FEATURE (unlike competition/venue_name) — lets the
        # model distinguish e.g. a high-scoring local league from a defensive top-flight one
        # Elo
        "home_elo": get_elo_rating(db, home.id, match.date),
        "away_elo": get_elo_rating(db, away.id, match.date),
        # Rest / fixture congestion
        "home_rest_days": get_rest_days(db, home.id, match.date),
        "away_rest_days": get_rest_days(db, away.id, match.date),
        # Rolling form
        "home_win_rate_last5": home_form["win_rate"],
        "home_avg_goals_scored_last5": home_form["avg_goals_scored"],
        "home_avg_goals_conceded_last5": home_form["avg_goals_conceded"],
        "away_win_rate_last5": away_form["win_rate"],
        "away_avg_goals_scored_last5": away_form["avg_goals_scored"],
        "away_avg_goals_conceded_last5": away_form["avg_goals_conceded"],
        # Head-to-head (from home team's perspective)
        "h2h_home_win_rate": h2h["h2h_win_rate"],
        "h2h_avg_goal_diff": h2h["h2h_avg_goal_diff"],
        "h2h_matches_played": h2h["h2h_matches_played"],
        # Travel / context
        "travel_distance_km": get_travel_distance_km(home, away, match=match),
        "venue_name": match.venue_name or f"{home.name} (default stadium)",
        "is_derby": is_derby(home.name, away.name),
        # Injuries
        "home_injury_count": home_injuries["injury_count"],
        "home_injury_importance_sum": home_injuries["injury_importance_sum"],
        "away_injury_count": away_injuries["injury_count"],
        "away_injury_importance_sum": away_injuries["injury_importance_sum"],
        # Target
        "home_score": match.home_score,
        "away_score": match.away_score,
        "result": result,
    }


def build_feature_table(db: Session) -> pd.DataFrame:
    matches = (
        db.query(Match)
        .filter(Match.home_score.isnot(None), Match.away_score.isnot(None))
        .order_by(Match.date.asc())
        .all()
    )
    rows = [build_feature_row(db, m) for m in matches]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full feature table from the database.")
    parser.add_argument("--output", default=None, help="Path to write the feature table as CSV")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        df = build_feature_table(db)
    finally:
        db.close()

    print(f"Built feature table: {len(df)} rows, {len(df.columns)} columns")
    print(df.head())

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
