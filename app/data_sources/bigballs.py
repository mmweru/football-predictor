"""
BigBalls Sports Data API (https://bigballsdata.com) — free key: 250 requests/day.

WHAT IT CAN DO FOR YOU (checked against their docs):
  * EPL ONLY among our 4 leagues — Championship / League One / League Two are not in its league list.
  * /v1/injuries?league=epl  -> past ABSENCES (who missed which played match, and why). Free tier. NOT a forecast of
    who is injured for the next game (that is what the FPL source gives us).
  * /v1/standings?league=EPL, /v1/matches?league=epl, /v1/teams/:id/elo  -> free.
  * xG leaderboards, injuries "designations" and odds are paid plans.

Response shapes below are parsed defensively (missing keys never crash a sync). Run
    python -m app.data_sources.check
once to print the real shape of each response and confirm the parser matches.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import requests

from app.data_sources import http

BASE = "https://api.bigballsdata.com"


class BigBalls:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None):
        if not api_key:
            raise http.DataSourceError("BIGBALLS_API_KEY is not set")
        self.headers = {"Authorization": f"Bearer {api_key}", "x-api-key": api_key}
        self.session = session or requests.Session()

    def _get(self, path: str, **params) -> dict:
        return http.get(self.session, f"{BASE}{path}", headers=self.headers, params=params or None).json()

    def epl_standings(self) -> List[dict]:
        data = self._get("/v1/standings", league="EPL").get("data", [])
        standings = data.get("standings", []) if isinstance(data, dict) else data
        if not isinstance(standings, list):
            return []
        rows = []
        for table in standings:
            if isinstance(table, dict) and isinstance(table.get("rows"), list):
                rows.extend(table["rows"])
            elif isinstance(table, dict):
                rows.append(table)
        return rows

    def epl_absences(self, today: Optional[dt.date] = None) -> List[dict]:
        """Normalise the current EPL injury feed into player/team/date records."""
        today = today or dt.date.today()
        payload = self._get("/v1/injuries", sport="football", limit=10)
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        injuries = data.get("injuries", []) if isinstance(data, dict) else data
        if isinstance(injuries, dict):
            rows = injuries.get("value", injuries.get("response", injuries.get("items", [])))
        else:
            rows = injuries
        if not isinstance(rows, list):
            return []

        team_names = {
            str(row["team_id"]): row["team_name"]
            for row in self.epl_standings()
            if row.get("team_id") is not None and row.get("team_name")
        }
        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            team = r.get("team") or r.get("team_name") or r.get("current_team_name")
            if isinstance(team, dict):
                team = team.get("name") or team.get("team_name")
            team_id = r.get("current_team_id") or r.get("team_id")
            team = team or team_names.get(str(team_id)) if team_id is not None else team
            player = r.get("player_name") or r.get("full_name") or r.get("display_name") or r.get("name")
            if isinstance(r.get("player"), dict):
                player = player or r["player"].get("name")
            if player and (team or team_id is not None):
                out.append({
                    "player": player,
                    "team": team,
                    "team_id": str(team_id) if team_id is not None else None,
                    "date": r.get("date") or r.get("injury_date") or r.get("match_date") or today,
                    "reason": r.get("reason") or r.get("type"),
                    "status": r.get("status") or "i",
                })
        return out


def sync_epl_injuries(db, today: Optional[dt.date] = None, client: Optional[BigBalls] = None) -> dict:
    """Apply current BigBalls injury rows only when a player maps unambiguously to an existing player."""
    from app.config import settings
    from app.models import Injury, Player

    today = today or dt.date.today()
    if client is None:
        if not settings.bigballs_api_key:
            raise http.DataSourceError("BIGBALLS_API_KEY is not set")
        client = BigBalls(settings.bigballs_api_key)
    players_by_name = {}
    for player in db.query(Player).all():
        key = " ".join(player.name.casefold().split())
        players_by_name.setdefault(key, []).append(player)

    result = {"rows": 0, "opened": 0, "matched": 0, "unmatched": 0}
    for row in client.epl_absences(today=today):
        result["rows"] += 1
        key = " ".join(row["player"].casefold().split())
        matches = players_by_name.get(key, [])
        if len(matches) != 1:
            result["unmatched"] += 1
            continue

        player = matches[0]
        player.status = "i"
        player.chance_of_playing = 0
        if row.get("reason"):
            player.news = row["reason"][:300]
        existing = db.query(Injury).filter_by(
            player_id=player.id, source="bigballs", date_back=None
        ).first()
        if existing is None:
            db.add(Injury(
                player_id=player.id,
                date_out=today,
                reason=(row.get("reason") or "Current BigBalls injury listing")[:200],
                source="bigballs",
            ))
            result["opened"] += 1
        result["matched"] += 1

    db.commit()
    return result
