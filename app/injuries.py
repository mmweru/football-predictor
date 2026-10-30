"""
Injuries + player importance.

HOW INJURIES ENTER A PREDICTION
  Historical injury records only exist for the Premier League and only recently, so injuries are NOT a trained model
  feature by default (a feature that is empty for 95% of training rows can't be learned). Instead they act at
  prediction time as an Elo penalty:  each unavailable player lowers his team's rating by
  importance x ELO_PER_IMPORTANCE points (capped), BEFORE the model sees the match. ELO_PER_IMPORTANCE is an
  ASSUMPTION (default 18, i.e. a star striker ~ -18 Elo, a whole spine missing ~ -60..-80); tune it once you have
  accumulated injury history (every `sync_data` run stores a snapshot, so after a season you can train with
  `--use-injuries` and measure it properly).
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List, Tuple

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models import Injury, Player

ELO_PER_IMPORTANCE = 18.0
MAX_ELO_PENALTY = 90.0
UNAVAILABLE_STATUSES = {"i", "s", "u"}


def compute_importance(minutes: float, xg: float, xa: float, team_total_xgxa: float, team_max_minutes: float) -> float:
    """0..1. A regular starter who creates ~20% of the team's attacking output = 1.0; a regular non-scorer ~0.4; a bench player ~0."""
    share = ((xg or 0) + (xa or 0)) / team_total_xgxa if team_total_xgxa > 0 else 0.0
    attack = min(1.0, share / 0.20)
    regular = min(1.0, (minutes or 0) / (0.85 * team_max_minutes)) if team_max_minutes > 0 else 0.0
    return round(regular * (0.4 + 0.6 * attack), 3)


def availability_factor(status: str, chance: int | None) -> float:
    """How 'out' a player is: 1 = definitely out, 0 = available. Doubtful players count in proportion to the doubt."""
    if status in UNAVAILABLE_STATUSES:
        return 1.0
    if status == "d":
        return 1.0 - (chance if chance is not None else 50) / 100.0
    return 0.0


def team_injury_report(db: Session, team_id: int) -> Dict[str, object]:
    """Current availability picture for a team: Elo penalty + the players behind it (for the UI)."""
    players = db.query(Player).filter(Player.team_id == team_id).all()
    out: List[dict] = []
    penalty = 0.0
    for p in players:
        f = availability_factor(p.status or "a", p.chance_of_playing)
        if f <= 0:
            continue
        impact = f * (p.importance_weight or 0.0)
        penalty += impact * ELO_PER_IMPORTANCE
        out.append({"player": p.name, "position": p.position, "status": p.status, "chance_of_playing": p.chance_of_playing,
                    "news": p.news, "importance": p.importance_weight, "impact": round(impact, 3)})
    out.sort(key=lambda r: -r["impact"])
    return {"count": len(out), "importance_sum": round(sum(r["impact"] for r in out), 3),
            "elo_penalty": round(min(penalty, MAX_ELO_PENALTY), 1), "players": out[:8], "has_data": bool(players)}


def make_injury_lookup(db: Session) -> Callable[[int, dt.date], Tuple[float, float]]:
    """For TRAINING with --use-injuries: f(team_id, date) -> (count, importance) from stored Injury rows; NaN if that team has none."""
    rows = db.query(Injury, Player).join(Player, Player.id == Injury.player_id).all()
    by_team: Dict[int, List[Tuple[dt.date, dt.date | None, float]]] = {}
    for inj, pl in rows:
        by_team.setdefault(pl.team_id, []).append((inj.date_out, inj.date_back, pl.importance_weight or 0.0))

    def lookup(team_id: int, d: dt.date) -> Tuple[float, float]:
        recs = by_team.get(team_id)
        if not recs:
            return (np.nan, np.nan)
        act = [imp for out, back, imp in recs if out <= d and (back is None or back >= d)]
        return (float(len(act)), float(sum(act)))

    return lookup


def add_manual_injury(db: Session, team_id: int, player_name: str, importance: float = 0.5,
                      reason: str | None = None, date_out: dt.date | None = None, date_back: dt.date | None = None) -> Injury:
    """Lets you record injuries for leagues with no free injury feed (Championship / League One / League Two)."""
    player = db.query(Player).filter_by(team_id=team_id, name=player_name).one_or_none()
    if player is None:
        player = Player(name=player_name, team_id=team_id, importance_weight=max(0.0, min(1.0, importance)))
        db.add(player)
        db.flush()
    player.status = "i"
    player.chance_of_playing = 0
    player.news = reason
    inj = Injury(player_id=player.id, date_out=date_out or dt.date.today(), date_back=date_back, reason=reason, source="manual")
    db.add(inj)
    db.commit()
    return inj
