"""
football-data.org v4 — FREE TIER covers the Premier League (PL) and Championship (ELC) ONLY; League One / League Two are
not on the free plan (they come from football-data.co.uk instead). Free limits: 10 requests/minute, scores delayed.
Register for a token at https://www.football-data.org/client/register and put it in .env as FOOTBALL_DATA_ORG_TOKEN.

Used here for the FULL season schedule of PL + ELC (football-data.co.uk's fixtures.csv only lists the next round or two),
and for the official list of clubs in each competition this season.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import requests

from app.data_sources import http
from app.data_sources.footballdata_uk import CANONICAL_COLUMNS
from app.leagues import LEAGUES, League

BASE = "https://api.football-data.org/v4"


class FootballDataOrg:
    def __init__(self, token: str, session: Optional[requests.Session] = None):
        if not token:
            raise http.DataSourceError("FOOTBALL_DATA_ORG_TOKEN is not set")
        self.token = token
        self.session = session or requests.Session()
        self.limiter = http.RateLimiter(6.5)   # 10 requests/min

    def _get(self, path: str, **params) -> dict:
        return http.get(self.session, f"{BASE}{path}", headers={"X-Auth-Token": self.token}, params=params or None,
                        limiter=self.limiter).json()

    def teams(self, league: League, season: int) -> List[dict]:
        return self._get(f"/competitions/{league.fdorg_code}/teams", season=season).get("teams", [])

    def matches(self, league: League, season: int) -> pd.DataFrame:
        """Whole-season schedule + results as our canonical frame."""
        rows = []
        for m in self._get(f"/competitions/{league.fdorg_code}/matches", season=season).get("matches", []):
            if m.get("stage") not in (None, "REGULAR_SEASON"):
                continue
            ft = (m.get("score") or {}).get("fullTime") or {}
            finished = m.get("status") == "FINISHED"
            rows.append({
                "date": dt.datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).date(),
                "league": league.name, "season": season,
                "home_team": m["homeTeam"]["name"], "away_team": m["awayTeam"]["name"],
                "home_score": ft.get("home") if finished else None, "away_score": ft.get("away") if finished else None,
            })
        df = pd.DataFrame(rows)
        for c in CANONICAL_COLUMNS:
            if c not in df:
                df[c] = float("nan")
        return df[CANONICAL_COLUMNS]

    def free_leagues(self) -> List[League]:
        return [l for l in LEAGUES if l.fdorg_code]
