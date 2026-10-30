"""
Fantasy Premier League API — free, no key, Premier League ONLY.
  https://fantasy.premierleague.com/api/bootstrap-static/
Gives, for every PL player: availability status (a/d/i/s/u), chance of playing next round, injury news text,
minutes, goals, assists, expected goals (xG) and expected assists (xA). This is our free source for
injuries (#1) and player stats (#7) for the EPL. It is an unofficial endpoint — be polite (one call per sync).
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

import requests
from sqlalchemy.orm import Session

from app.data_sources import http
from app.injuries import UNAVAILABLE_STATUSES, compute_importance
from app.models import Injury, Player, Team
from app.team_aliases import normalize_team_name

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch_bootstrap(session: Optional[requests.Session] = None) -> dict:
    return http.get(session or requests.Session(), URL, retries=2).json()


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def sync_fpl(db: Session, data: Optional[dict] = None, today: Optional[dt.date] = None) -> Dict[str, int]:
    """Updates players, their importance, status, and an injury HISTORY (open/close Injury rows) from one bootstrap payload."""
    today = today or dt.date.today()
    data = data or fetch_bootstrap()
    teams_by_name = {t.name: t for t in db.query(Team).all()}
    fpl_team: Dict[int, Team] = {}
    for t in data.get("teams", []):
        team = teams_by_name.get(normalize_team_name(t["name"])) or teams_by_name.get(normalize_team_name(t.get("short_name", "")))
        if team is not None:
            team.fpl_id = t["id"]
            fpl_team[t["id"]] = team
    stats = {"players": 0, "unmatched_teams": len(data.get("teams", [])) - len(fpl_team), "injuries_opened": 0, "injuries_closed": 0}

    by_team: Dict[int, list] = {}
    for e in data.get("elements", []):
        if e["team"] in fpl_team:
            by_team.setdefault(e["team"], []).append(e)

    existing = {p.fpl_id: p for p in db.query(Player).filter(Player.fpl_id.isnot(None)).all()}
    for fid, elems in by_team.items():
        team = fpl_team[fid]
        total_xgxa = sum(_f(e.get("expected_goals")) + _f(e.get("expected_assists")) for e in elems)
        max_min = max((e.get("minutes") or 0) for e in elems) or 1
        for e in elems:
            player = existing.get(e["id"])
            name = f'{e.get("first_name", "")} {e.get("second_name", "")}'.strip() or e.get("web_name", "?")
            if player is None:
                player = Player(name=name, team_id=team.id, fpl_id=e["id"])
                db.add(player)
                db.flush()
                existing[e["id"]] = player
            player.team_id = team.id
            player.position = POSITIONS.get(e.get("element_type"))
            player.minutes, player.goals, player.assists = e.get("minutes"), e.get("goals_scored"), e.get("assists")
            player.xg, player.xa = _f(e.get("expected_goals")), _f(e.get("expected_assists"))
            player.status = e.get("status")
            player.chance_of_playing = e.get("chance_of_playing_next_round")
            player.news = (e.get("news") or None) and e["news"][:300]
            player.importance_weight = compute_importance(player.minutes or 0, player.xg, player.xa, total_xgxa, max_min)
            stats["players"] += 1

            open_inj = db.query(Injury).filter(Injury.player_id == player.id, Injury.source == "fpl", Injury.date_back.is_(None)).first()
            out_now = player.status in UNAVAILABLE_STATUSES or (player.status == "d" and (player.chance_of_playing or 0) < 50)
            if out_now and open_inj is None:
                db.add(Injury(player_id=player.id, date_out=today, reason=player.news, source="fpl"))
                stats["injuries_opened"] += 1
            elif not out_now and open_inj is not None:
                open_inj.date_back = today
                stats["injuries_closed"] += 1
    db.commit()
    return stats
