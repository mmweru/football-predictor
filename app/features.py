"""
Feature engineering functions — each one computes a single feature for one
team going into one match, using ONLY information available before that
match's kickoff (i.e. strictly earlier match dates). This "before" cutoff
is the most important property of every function here: get it wrong and
you leak future information into training, which makes your model look
great in validation and fail in production.

Every function takes a Session (db) and returns a plain Python value, so
they compose cleanly into build_feature_table() at the bottom, which
assembles the full per-match feature row for the modeling stage.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.derbies import is_derby
from app.distance import haversine_km
from app.models import EloHistory, Injury, Match, Player, Team


# ---------------------------------------------------------------------------
# Rest days / fixture congestion
# ---------------------------------------------------------------------------
def get_rest_days(db: Session, team_id: int, before_date: dt.date) -> Optional[int]:
    """
    Days since this team's most recent PREVIOUS match (strictly before
    before_date). Returns None if this is the team's first match on record
    (no prior match to compare against) — callers should decide how to
    treat that (e.g. impute a neutral default like 7).
    """
    last_match = (
        db.query(Match)
        .filter(
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            Match.date < before_date,
            Match.home_score.isnot(None),
        )
        .order_by(Match.date.desc())
        .first()
    )
    if last_match is None:
        return None
    return (before_date - last_match.date).days


# ---------------------------------------------------------------------------
# Rolling form
# ---------------------------------------------------------------------------
def get_rolling_form(db: Session, team_id: int, before_date: dt.date, num_matches: int = 5) -> Dict[str, float]:
    """
    Win rate and average goals scored/conceded over the team's last
    `num_matches` matches strictly before before_date. Returns neutral
    defaults (0.5 win rate isn't neutral for goals, so we use None-safe
    zeros there) if there's no history yet — callers should treat a
    returned `matches_played: 0` as "insufficient history, consider
    imputing league-average or skipping this row."
    """
    recent_matches: List[Match] = (
        db.query(Match)
        .filter(
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            Match.date < before_date,
            Match.home_score.isnot(None),
        )
        .order_by(Match.date.desc())
        .limit(num_matches)
        .all()
    )

    if not recent_matches:
        return {"win_rate": 0.5, "avg_goals_scored": 0.0, "avg_goals_conceded": 0.0, "matches_played": 0}

    wins = 0
    goals_scored_total = 0
    goals_conceded_total = 0

    for m in recent_matches:
        is_home = m.home_team_id == team_id
        scored = m.home_score if is_home else m.away_score
        conceded = m.away_score if is_home else m.home_score
        goals_scored_total += scored
        goals_conceded_total += conceded
        if scored > conceded:
            wins += 1

    n = len(recent_matches)
    return {
        "win_rate": wins / n,
        "avg_goals_scored": goals_scored_total / n,
        "avg_goals_conceded": goals_conceded_total / n,
        "matches_played": n,
    }


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------
def get_head_to_head(
    db: Session, team_a_id: int, team_b_id: int, before_date: dt.date, num_matches: int = 5
) -> Dict[str, float]:
    """
    Team A's win rate and average goal difference in the last N meetings
    between these two specific teams (in either home/away configuration),
    strictly before before_date.
    """
    recent_meetings: List[Match] = (
        db.query(Match)
        .filter(
            or_(
                (Match.home_team_id == team_a_id) & (Match.away_team_id == team_b_id),
                (Match.home_team_id == team_b_id) & (Match.away_team_id == team_a_id),
            ),
            Match.date < before_date,
            Match.home_score.isnot(None),
        )
        .order_by(Match.date.desc())
        .limit(num_matches)
        .all()
    )

    if not recent_meetings:
        return {"h2h_win_rate": 0.5, "h2h_avg_goal_diff": 0.0, "h2h_matches_played": 0}

    a_wins = 0
    goal_diff_total = 0

    for m in recent_meetings:
        a_is_home = m.home_team_id == team_a_id
        a_goals = m.home_score if a_is_home else m.away_score
        b_goals = m.away_score if a_is_home else m.home_score
        goal_diff_total += a_goals - b_goals
        if a_goals > b_goals:
            a_wins += 1

    n = len(recent_meetings)
    return {
        "h2h_win_rate": a_wins / n,
        "h2h_avg_goal_diff": goal_diff_total / n,
        "h2h_matches_played": n,
    }


# ---------------------------------------------------------------------------
# Travel distance
# ---------------------------------------------------------------------------
def get_travel_distance_km(
    home_team: Team, away_team: Team, match: Optional[Match] = None
) -> Optional[float]:
    """
    Distance the AWAY team is travelling, from their home stadium to the
    match venue.

    Venue resolution order:
      1. match.venue_lat / match.venue_lon, if set (a specific override —
         e.g. a local match not played at the home team's usual ground,
         set via app.set_match_venue)
      2. home_team.stadium_lat / stadium_lon (the default assumption)

    Returns None if neither a venue override nor home stadium coordinates
    are available, or if the away team has no stadium coordinates.
    """
    if match is not None and match.venue_lat is not None and match.venue_lon is not None:
        dest_lat, dest_lon = match.venue_lat, match.venue_lon
    else:
        dest_lat, dest_lon = home_team.stadium_lat, home_team.stadium_lon

    if None in (dest_lat, dest_lon, away_team.stadium_lat, away_team.stadium_lon):
        return None
    return haversine_km(away_team.stadium_lat, away_team.stadium_lon, dest_lat, dest_lon)


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------
def get_injury_impact(db: Session, team_id: int, match_id: int) -> Dict[str, float]:
    """
    For the given match, counts how many of this team's players are marked
    injured for it (Injury.match_missed_id == match_id), and sums their
    importance_weight as a rough "how much this hurts" score.

    Only usable for HISTORICAL matches that already exist as a row with an
    id — not usable for predicting an upcoming match that hasn't been
    ingested yet. For that case, use get_current_injury_impact instead.
    """
    injuries = (
        db.query(Injury)
        .join(Player, Player.id == Injury.player_id)
        .filter(Player.team_id == team_id, Injury.match_missed_id == match_id)
        .all()
    )
    count = len(injuries)
    total_importance = sum(i.player.importance_weight for i in injuries)
    return {"injury_count": count, "injury_importance_sum": total_importance}


def get_current_injury_impact(db: Session, team_id: int, as_of_date: dt.date) -> Dict[str, float]:
    """
    Counts players from this team who are injured AS OF a given date,
    based on their injury window (date_out <= as_of_date <= date_back),
    independent of any specific match_missed_id. This is the version used
    for LIVE predictions of upcoming matches — a future match has no
    match_id in the database yet, so get_injury_impact's approach doesn't
    apply. An injury with no date_back (long-term/unresolved) counts as
    "still out" for any as_of_date on or after date_out.
    """
    injuries = (
        db.query(Injury)
        .join(Player, Player.id == Injury.player_id)
        .filter(
            Player.team_id == team_id,
            Injury.date_out <= as_of_date,
            or_(Injury.date_back.is_(None), Injury.date_back >= as_of_date),
        )
        .all()
    )
    count = len(injuries)
    total_importance = sum(i.player.importance_weight for i in injuries)
    return {"injury_count": count, "injury_importance_sum": total_importance}


# ---------------------------------------------------------------------------
# Elo lookup (reads what app.elo already computed)
# ---------------------------------------------------------------------------
def get_elo_rating(db: Session, team_id: int, match_date: dt.date) -> float:
    """
    Returns the pre-match Elo rating stored for this team on this exact
    match date (written by app.elo's build_elo_history). Falls back to
    1500 if not found — shouldn't normally happen if app.elo has been run
    on this data, but keeps this function safe to call standalone.

    Used for HISTORICAL training rows, where match_date always corresponds
    to an actual match that has an elo_history entry. For predicting an
    UPCOMING match (no elo_history entry on that future date yet), use
    get_current_elo_rating instead.
    """
    row = db.query(EloHistory).filter_by(team_id=team_id, date=match_date).one_or_none()
    return row.rating if row is not None else 1500.0


def get_current_elo_rating(db: Session, team_id: int, as_of_date: dt.date) -> float:
    """
    Returns the team's most recent Elo rating on or before as_of_date —
    i.e. "what is this team's Elo right now, as of this date" rather than
    an exact-date lookup. This is the version to use for LIVE predictions
    of upcoming matches, since a future date has no elo_history row of its
    own yet. Falls back to 1500 if the team has no Elo history at all.
    """
    row = (
        db.query(EloHistory)
        .filter(EloHistory.team_id == team_id, EloHistory.date <= as_of_date)
        .order_by(EloHistory.date.desc())
        .first()
    )
    return row.rating if row is not None else 1500.0
