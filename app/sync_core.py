"""
Database-side half of data syncing: takes clean DataFrames (from any source)
and upserts them. Kept separate from the network code so it is fully testable
offline.

Two jobs that fix your original problems:

1. upsert_matches()      — inserts new matches AND updates existing ones, so a
                           fixture ingested last week as "upcoming" gets its
                           real score + stats once played ("subsequent games
                           played"), and rescheduled fixtures move instead of
                           duplicating.
2. refresh_membership()  — re-derives every team's league FROM THE CURRENT
                           SEASON and flips Team.is_active. Your old
                           ingest_csv only set `league` when a team was first
                           created, so relegated clubs stayed tagged
                           "Premier League" forever (35 clubs, not 20).
"""

from __future__ import annotations

import datetime as dt
import difflib
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.leagues import BY_NAME, LEAGUES, canonical_league
from app.models import Match, Prediction, Team
from app.team_aliases import normalize_team_name

STAT_FIELDS = [
    "home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
    "home_fouls", "away_fouls", "home_yellow", "away_yellow", "home_red", "away_red",
]
ODDS_FIELDS = ["odds_home", "odds_draw", "odds_away"]


def _clean(v):
    """pandas NaN/NaT/numpy scalars -> plain python / None for SQLAlchemy."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        v = v.item()
    return v


class TeamResolver:
    """Caches name -> Team so a 10k-row ingest doesn't issue 20k lookups."""

    def __init__(self, db: Session):
        self.db = db
        self._cache: Dict[str, Team] = {t.name: t for t in db.query(Team).all()}

    def get(self, raw_name: str, league: Optional[str]) -> Team:
        name = normalize_team_name(raw_name)
        team = self._cache.get(name)
        if team is None:
            team = Team(name=name, league=league, is_active=True)
            self.db.add(team)
            self.db.flush()
            self._cache[name] = team
        return team


def upsert_matches(db: Session, df: pd.DataFrame, *, resolver: Optional[TeamResolver] = None) -> Dict[str, int]:
    """
    df columns: see app.data_sources.footballdata_uk.CANONICAL_COLUMNS. Returns counts
    {inserted, updated, unchanged}. Matching order for an incoming row:
      1. exact (date, home, away)
      2. same (home, away, season) among UNPLAYED matches  -> it was rescheduled: move the date
      3. otherwise insert
    """
    resolver = resolver or TeamResolver(db)
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    if df.empty:
        return counts

    existing = db.query(Match).filter(Match.date >= min(df["date"]) - dt.timedelta(days=400)).all()
    by_exact: Dict[Tuple, Match] = {(m.date, m.home_team_id, m.away_team_id): m for m in existing}
    unplayed_by_pair: Dict[Tuple, List[Match]] = {}
    for m in existing:
        if m.home_score is None:
            unplayed_by_pair.setdefault((m.home_team_id, m.away_team_id, m.season), []).append(m)

    for row in df.itertuples(index=False):
        home = resolver.get(row.home_team, row.league)
        away = resolver.get(row.away_team, row.league)
        if home.id == away.id:
            continue
        season = int(row.season)
        key = (row.date, home.id, away.id)
        match = by_exact.get(key)
        if match is None:
            pair = unplayed_by_pair.get((home.id, away.id, season), [])
            if pair and pd.isna(row.home_score):
                match = pair[0]            # rescheduled fixture: same pair + season, still unplayed
                del by_exact[(match.date, match.home_team_id, match.away_team_id)]
                match.date = row.date
                by_exact[key] = match
        is_new = match is None
        if is_new:
            match = Match(date=row.date, home_team_id=home.id, away_team_id=away.id)
            db.add(match)
            by_exact[key] = match

        values = {
            "competition": row.league, "season": season,
            "home_score": _clean(int(row.home_score)) if pd.notna(row.home_score) else None,
            "away_score": _clean(int(row.away_score)) if pd.notna(row.away_score) else None,
        }
        for f in STAT_FIELDS + ODDS_FIELDS:
            v = _clean(getattr(row, f, None))
            values[f] = int(v) if (v is not None and f in STAT_FIELDS) else v

        changed = is_new
        for k, v in values.items():
            if v is None and not is_new:
                continue                      # never overwrite known data with blanks
            if getattr(match, k) != v:
                setattr(match, k, v)
                changed = True
        counts["inserted" if is_new else ("updated" if changed else "unchanged")] += 1
        if changed or is_new:
            db.flush()
    db.commit()
    return counts


def purge_stale_fixtures(db: Session, upcoming: pd.DataFrame, today: dt.date) -> int:
    """
    Removes UNPLAYED matches inside the window covered by `upcoming` that the source no longer lists
    (cancelled / replaced). Played matches are never touched.
    """
    if upcoming.empty:
        return 0
    lo, hi = min(upcoming["date"]), max(upcoming["date"])
    resolver = TeamResolver(db)
    keep = set()
    for r in upcoming.itertuples(index=False):
        keep.add((r.date, resolver.get(r.home_team, r.league).id, resolver.get(r.away_team, r.league).id))
    stale = (
        db.query(Match)
        .filter(Match.home_score.is_(None), Match.date >= max(lo, today), Match.date <= hi)
        .all()
    )
    removed = 0
    for m in stale:
        if not canonical_league(m.competition):
            continue                                  # leave anything that isn't one of our 4 leagues alone
        if (m.date, m.home_team_id, m.away_team_id) not in keep:
            db.query(Prediction).filter_by(match_id=m.id).delete()
            db.delete(m)
            removed += 1
    db.commit()
    return removed


# ---------------------------------------------------------------------------
# Team membership: which league is each club in THIS season?
# ---------------------------------------------------------------------------
def refresh_membership(db: Session, active_season: int, official_rosters: Optional[Dict[str, set[str]]] = None) -> Dict[str, object]:
    """
    For each team: look at its matches in the most recent season that has any
    (<= active_season) within our 4 leagues. It is ACTIVE (visible in the UI)
    only if that season is `active_season`; its `league` is set to the league it
    played in that season. Teams with no matches in a supported league (e.g. the
    Kenyan clubs) are marked inactive, never deleted.
    """
    report: Dict[str, object] = {"active_season": active_season, "per_league": {}, "warnings": []}
    # latest (season, competition) per team, across home and away matches, supported leagues only
    rows = (
        db.query(Match.home_team_id, Match.away_team_id, Match.season, Match.competition)
        .filter(Match.season.isnot(None))
        .all()
    )
    latest: Dict[int, Tuple[int, str]] = {}
    for home_id, away_id, season, comp in rows:
        league = canonical_league(comp)
        if league is None or season is None or season > active_season:
            continue
        for tid in (home_id, away_id):
            if tid not in latest or season > latest[tid][0]:
                latest[tid] = (season, league)

    per_league: Dict[str, int] = {l.name: 0 for l in LEAGUES}
    official_rosters = official_rosters or {}
    official_assignment = {
        normalize_team_name(name): league
        for league, names in official_rosters.items()
        for name in names
    }
    for team in db.query(Team).all():
        official_league = official_assignment.get(normalize_team_name(team.name))
        if official_league is not None:
            team.league = official_league
            team.is_active = True
            per_league[official_league] += 1
            continue
        info = latest.get(team.id)
        if info is None:
            team.is_active = False
            continue
        season, league = info
        team.league = league
        team.is_active = season == active_season and league not in official_rosters
        if team.is_active:
            per_league[league] += 1
    db.commit()

    report["per_league"] = per_league
    for lg in LEAGUES:
        n = per_league[lg.name]
        if n == 0:
            report["warnings"].append(f"{lg.name}: no clubs found for {active_season}/{(active_season + 1) % 100:02d} — has this season's data been synced?")
        elif n != lg.expected_teams:
            report["warnings"].append(
                f"{lg.name}: {n} active clubs but a season has {lg.expected_teams}. "
                f"{'Some clubs have not played/been scheduled yet' if n < lg.expected_teams else 'A club name is probably spelled two ways — see duplicate report'}."
            )
    return report


def possible_duplicate_teams(db: Session, cutoff: float = 0.86) -> List[Tuple[str, str, float]]:
    """Pairs of teams (one active, one not or both active) whose names are suspiciously alike — add them to team_aliases.py."""
    teams = db.query(Team).filter(Team.league.isnot(None)).all()
    names = [t.name for t in teams if canonical_league(t.league)]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if ratio >= cutoff and not ({a, b} <= {"Sheffield United", "Sheffield Wednesday"}):
                out.append((a, b, round(ratio, 2)))
    return out
