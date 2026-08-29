"""
Computes Elo ratings from the `matches` table, processed strictly in
chronological order, and writes one row per team per match into
`elo_history`.

Stores the PRE-MATCH rating for each team on each match date — i.e. the
rating used to make that match's prediction, not the rating that resulted
from it. This is deliberate: when you later build features like
"home_elo_pre" / "away_elo_pre" for the XGBoost model, you want the rating
as it stood *before* kickoff, since that's the only information actually
available at prediction time.

Usage:
    python -m app.elo                  # build/update ratings for all matches
    python -m app.elo --reset          # wipe elo_history and rebuild from scratch
    python -m app.elo --k 32           # use a different K-factor
    python -m app.elo --home-advantage 100
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, Optional, Tuple

from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import EloHistory, Match

BASE_RATING = 1500.0
SEASON_GAP_DAYS = 60  # a gap this long between a team's matches is treated as an off-season
SEASON_REGRESSION_WEIGHT = 0.75  # fraction of old rating kept; rest pulled toward BASE_RATING


def expected_score(elo_a: float, elo_b: float) -> float:
    """Win-probability-like expected score for A, given both ratings (no home bonus applied here)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_home: float, elo_away: float, result_home: float, k: float, home_advantage: float) -> Tuple[float, float]:
    """
    result_home: 1.0 = home win, 0.5 = draw, 0.0 = home loss.
    home_advantage is added ONLY for computing expected score, never stored
    back into the team's actual rating.
    """
    exp_home = expected_score(elo_home + home_advantage, elo_away)
    exp_away = 1.0 - exp_home
    result_away = 1.0 - result_home

    new_home = elo_home + k * (result_home - exp_home)
    new_away = elo_away + k * (result_away - exp_away)
    return new_home, new_away


def regress_toward_mean(rating: float) -> float:
    return SEASON_REGRESSION_WEIGHT * rating + (1 - SEASON_REGRESSION_WEIGHT) * BASE_RATING


def build_elo_history(db: Session, k: float, home_advantage: float) -> int:
    """
    Processes every scored match in chronological order, updates an
    in-memory rating dict, and upserts a pre-match EloHistory row per team
    per match. Returns the number of EloHistory rows written.
    """
    matches = (
        db.query(Match)
        .filter(Match.home_score.isnot(None), Match.away_score.isnot(None))
        .order_by(Match.date.asc(), Match.id.asc())
        .all()
    )

    ratings: Dict[int, float] = {}
    last_played: Dict[int, dt.date] = {}
    rows_written = 0

    for match in matches:
        home_id, away_id = match.home_team_id, match.away_team_id

        # Initialize new teams at the base rating.
        ratings.setdefault(home_id, BASE_RATING)
        ratings.setdefault(away_id, BASE_RATING)

        # Season regression: if either team hasn't played in a long time
        # (i.e. this looks like a new season), pull their rating back
        # toward the mean before using it for this match.
        for team_id in (home_id, away_id):
            prev_date = last_played.get(team_id)
            if prev_date is not None and (match.date - prev_date).days > SEASON_GAP_DAYS:
                ratings[team_id] = regress_toward_mean(ratings[team_id])

        home_elo_pre = ratings[home_id]
        away_elo_pre = ratings[away_id]

        # Write PRE-match ratings — the info available before kickoff.
        _upsert_elo_row(db, home_id, match.date, home_elo_pre)
        _upsert_elo_row(db, away_id, match.date, away_elo_pre)
        rows_written += 2

        # Determine result and update ratings for future matches.
        if match.home_score > match.away_score:
            result_home = 1.0
        elif match.home_score < match.away_score:
            result_home = 0.0
        else:
            result_home = 0.5

        new_home, new_away = update_elo(home_elo_pre, away_elo_pre, result_home, k, home_advantage)
        ratings[home_id] = new_home
        ratings[away_id] = new_away
        last_played[home_id] = match.date
        last_played[away_id] = match.date

    db.commit()
    return rows_written


def _upsert_elo_row(db: Session, team_id: int, date: dt.date, rating: float) -> None:
    """
    Inserts an EloHistory row, or updates it in place if one already exists
    for this team+date (the unique constraint from the schema) — this makes
    the script safe to re-run without manually clearing elo_history first,
    as long as you're not also duplicating matches (ingest_csv already
    guards against that separately).
    """
    existing = db.query(EloHistory).filter_by(team_id=team_id, date=date).one_or_none()
    if existing is not None:
        existing.rating = rating
    else:
        db.add(EloHistory(team_id=team_id, date=date, rating=rating))
    db.flush()


def reset_elo_history(db: Session) -> None:
    deleted = db.query(EloHistory).delete()
    db.commit()
    print(f"Cleared {deleted} existing elo_history rows.")


def print_current_standings(db: Session, top_n: int = 10) -> None:
    """Prints the most recent rating per team — a quick sanity check after building."""
    from sqlalchemy import func

    from app.models import Team

    latest_dates = (
        db.query(EloHistory.team_id, func.max(EloHistory.date).label("latest_date"))
        .group_by(EloHistory.team_id)
        .subquery()
    )
    rows = (
        db.query(Team.name, EloHistory.rating, EloHistory.date)
        .join(EloHistory, EloHistory.team_id == Team.id)
        .join(
            latest_dates,
            (EloHistory.team_id == latest_dates.c.team_id) & (EloHistory.date == latest_dates.c.latest_date),
        )
        .order_by(EloHistory.rating.desc())
        .limit(top_n)
        .all()
    )
    print(f"\n--- Current Elo standings (top {top_n}) ---")
    for name, rating, date in rows:
        print(f"  {rating:7.1f}  {name}  (as of {date})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Elo rating history from the matches table.")
    parser.add_argument("--k", type=float, default=20.0, help="K-factor controlling update size (default 20)")
    parser.add_argument(
        "--home-advantage", type=float, default=70.0, help="Rating bonus applied to home team's expected score (default 70)"
    )
    parser.add_argument("--reset", action="store_true", help="Clear elo_history before rebuilding")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if args.reset:
            reset_elo_history(db)

        rows_written = build_elo_history(db, k=args.k, home_advantage=args.home_advantage)
        print(f"Elo build complete. {rows_written} elo_history rows written/updated.")
        print_current_standings(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
