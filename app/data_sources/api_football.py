"""API-Football v3 client for historical fixtures and season player statistics."""

from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from app.data_sources import http
from app.data_sources.footballdata_uk import CANONICAL_COLUMNS
from app.leagues import LEAGUES, League

BASE = "https://v3.football.api-sports.io"
LEAGUE_IDS = {
    "Premier League": 39,
    "EFL Championship": 40,
    "EFL League One": 41,
    "EFL League Two": 42,
}
FINISHED_STATUSES = {"FT", "AET", "PEN"}
MAX_PLAYER_PAGES = 3


class SeasonUnavailableError(http.DataSourceError):
    """The API plan does not include the requested season."""


class RateLimitError(http.DataSourceError):
    """The API reports that its request quota is currently exhausted."""


def _as_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_fixtures(league: League, season: int, payload_rows: Iterable[dict]) -> pd.DataFrame:
    """Convert API-Football fixture records to the sync pipeline's canonical frame."""
    rows = []
    for item in payload_rows:
        fixture = item.get("fixture") or {}
        teams = item.get("teams") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        if not fixture.get("date") or not home.get("name") or not away.get("name"):
            continue
        kickoff = dt.datetime.fromisoformat(fixture["date"].replace("Z", "+00:00"))
        finished = (fixture.get("status") or {}).get("short") in FINISHED_STATUSES
        goals = item.get("goals") or {}
        rows.append({
            "date": kickoff.date(), "league": league.name, "season": season,
            "home_team": home["name"], "away_team": away["name"],
            "home_score": goals.get("home") if finished else None,
            "away_score": goals.get("away") if finished else None,
        })
    frame = pd.DataFrame(rows)
    for column in CANONICAL_COLUMNS:
        if column not in frame:
            frame[column] = float("nan")
    return frame[CANONICAL_COLUMNS]


def parse_player_stats(payload_rows: Iterable[dict], league: League, season: int) -> List[dict]:
    """Normalize API-Football player/club season totals without mixing seasons."""
    out = []
    for item in payload_rows:
        player = item.get("player") or {}
        for stats in item.get("statistics") or []:
            team = stats.get("team") or {}
            if not player.get("id") or not player.get("name") or not team.get("name"):
                continue
            games = stats.get("games") or {}
            goals = stats.get("goals") or {}
            shots = stats.get("shots") or {}
            cards = stats.get("cards") or {}
            out.append({
                "api_player_id": _as_int(player.get("id")), "player_name": player["name"],
                "team_name": team["name"], "league": league.name, "season": season,
                "position": games.get("position"), "appearances": _as_int(games.get("appearances")),
                "lineups": _as_int(games.get("lineups")), "minutes": _as_int(games.get("minutes")),
                "goals": _as_int(goals.get("total")), "assists": _as_int(goals.get("assists")),
                "shots": _as_int(shots.get("total")), "shots_on_target": _as_int(shots.get("on")),
                "yellow_cards": _as_int(cards.get("yellow")), "red_cards": _as_int(cards.get("red")),
                "rating": _as_float(games.get("rating")),
            })
    return out


class APIFootball:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None, min_request_interval_s: float = 6.1):
        if not api_key:
            raise http.DataSourceError("FOOTBALL_API_KEY is not set")
        self.session = session or requests.Session()
        self.headers = {"x-apisports-key": api_key}
        self.limiter = http.RateLimiter(min_interval_s=min_request_interval_s)

    def _get(self, path: str, **params) -> dict:
        payload = http.get(
            self.session, f"{BASE}{path}", headers=self.headers, params=params or None, limiter=self.limiter,
        ).json()
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if errors:
            detail = str(errors)
            lowered = detail.casefold()
            if "too many requests" in lowered or "ratelimit" in lowered or "rate limit" in lowered:
                raise RateLimitError(f"API-Football rate limit reached: {detail}")
            if "do not have access to this season" in lowered:
                raise SeasonUnavailableError(f"API-Football plan cannot access this season: {detail}")
            raise http.DataSourceError(f"API-Football {path} returned errors: {errors}")
        return payload

    def fixtures(self, league: League, season: int) -> pd.DataFrame:
        league_id = LEAGUE_IDS[league.name]
        payload = self._get("/fixtures", league=league_id, season=season)
        return parse_fixtures(league, season, payload.get("response", []))

    def player_stats(self, league: League, season: int) -> List[dict]:
        league_id = LEAGUE_IDS[league.name]
        page = 1
        total_pages = 1
        players = []
        while page <= min(total_pages, MAX_PLAYER_PAGES):
            payload = self._get("/players", league=league_id, season=season, page=page)
            players.extend(parse_player_stats(payload.get("response", []), league, season))
            paging = payload.get("paging") or {}
            total_pages = max(total_pages, _as_int(paging.get("total")) or page)
            page += 1
        if total_pages > MAX_PLAYER_PAGES:
            print(f"  [limited] API-Football {league.name} {season}: fetched {MAX_PLAYER_PAGES} of {total_pages} player pages allowed by this plan")
        return players


def sync_player_stats(db, rows: List[dict]) -> Dict[str, int]:
    """Upsert historical player totals into PlayerSeasonStats, never current Player availability."""
    from app.models import PlayerSeasonStats
    from app.sync_core import TeamResolver

    resolver = TeamResolver(db)
    existing = {
        (row.api_player_id, row.team_id, row.season, row.league): row
        for row in db.query(PlayerSeasonStats).all()
    }
    counts = {"inserted": 0, "updated": 0}
    for data in rows:
        league = data["league"]
        team = resolver.get(data["team_name"], league)
        key = (data["api_player_id"], team.id, data["season"], league)
        row = existing.get(key)
        is_new = row is None
        if is_new:
            row = PlayerSeasonStats(api_player_id=data["api_player_id"], team_id=team.id,
                                    season=data["season"], league=league)
            db.add(row)
            existing[key] = row
        for field in (
            "player_name", "position", "appearances", "lineups", "minutes", "goals", "assists",
            "shots", "shots_on_target", "yellow_cards", "red_cards", "rating",
        ):
            setattr(row, field, data.get(field))
        counts["inserted" if is_new else "updated"] += 1
    db.commit()
    return counts


def sync_api_football(db, api: APIFootball, seasons: Iterable[int]) -> Dict[str, int]:
    """Sync fixtures and historical player totals for the configured EPL/EFL seasons."""
    from app.sync_core import upsert_matches

    totals = {"matches_inserted": 0, "player_rows_inserted": 0, "player_rows_updated": 0}
    for season in seasons:
        season_unavailable = False
        for league in LEAGUES:
            try:
                fixture_frame = api.fixtures(league, season)
                counts = upsert_matches(db, fixture_frame)
                totals["matches_inserted"] += counts["inserted"]
            except RateLimitError as error:
                print(f"  [stop] API-Football: {error}")
                return totals
            except SeasonUnavailableError as error:
                print(f"  [skip] API-Football {season}: {error}")
                season_unavailable = True
                break
            except (http.DataSourceError, ValueError) as error:
                print(f"  [skip] API-Football {league.name} {season} fixtures: {error}")
                continue
            try:
                player_counts = sync_player_stats(db, api.player_stats(league, season))
                totals["player_rows_inserted"] += player_counts["inserted"]
                totals["player_rows_updated"] += player_counts["updated"]
            except RateLimitError as error:
                print(f"  [stop] API-Football: {error}")
                return totals
            except (http.DataSourceError, ValueError) as error:
                print(f"  [skip] API-Football {league.name} {season} player stats: {error}")
        if season_unavailable:
            continue
    return totals