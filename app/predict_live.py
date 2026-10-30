"""
Live prediction for ONE matchup (any two teams, any date) or for every stored upcoming fixture.

It reuses the exact FeatureEngine used in training. The engine is run once over all played matches (cached until the
data changes); a fixture is then just "compute features from the current state" — no leakage, no skew.

Output (per match): H/D/A probabilities, expected goals, scoreline/over-under/BTTS from the Poisson model, and a
`factors` list covering each of your metrics: injuries, home advantage, fatigue, game factors (derby / travel / weather),
head-to-head, team stats, match situation (attack vs defence) and player stats.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.feature_engine import FeatureEngine, load_matches_frame, team_info, MAX_REST_DAYS
from app.injuries import team_injury_report
from app.leagues import TIER_BY_NAME, canonical_league
from app.models import Match, Prediction, Team
from app.modeling import LABELS, scoreline_summary

_engine_cache: Dict[str, object] = {"key": None, "engine": None}


def _data_key(db: Session) -> tuple:
    played = db.query(func.count(Match.id), func.max(Match.date)).filter(Match.home_score.isnot(None)).one()
    return (played[0], str(played[1]))


def get_engine(db: Session) -> FeatureEngine:
    """Engine whose state = every played match so far. Rebuilt only when a new result arrives."""
    key = _data_key(db)
    if _engine_cache["key"] != key:
        frame = load_matches_frame(db)
        eng = FeatureEngine(team_info(db))
        if not frame.empty:
            eng.run(frame[frame["home_score"].notna()])        # state only; fixtures are featurised on demand below
        _engine_cache.update(key=key, engine=eng)
    return _engine_cache["engine"]


def matchup_features(db: Session, home: Team, away: Team, date: dt.date, overrides: Optional[dict] = None) -> dict:
    eng = get_engine(db)
    league = canonical_league(home.league) or "Premier League"
    tier = TIER_BY_NAME.get(league, 1)
    ov = overrides or {}
    row = SimpleNamespace(
        date=date, home_id=home.id, away_id=away.id, venue_lat=np.nan, venue_lon=np.nan,
        temp_c=ov.get("temp_c", np.nan), wind_kph=ov.get("wind_kph", np.nan), precip_mm=ov.get("precip_mm", np.nan),
        odds_home=ov.get("odds_home", np.nan), odds_draw=ov.get("odds_draw", np.nan), odds_away=ov.get("odds_away", np.nan),
    )
    from app.feature_engine import TeamState, ELO_BASE_BY_TIER, LeagueState
    hs = eng.state.get(home.id) or TeamState(elo=ELO_BASE_BY_TIER[tier], league_tier=tier)
    as_ = eng.state.get(away.id) or TeamState(elo=ELO_BASE_BY_TIER[tier], league_tier=tier)
    lg = eng.league_state.get(league) or LeagueState()
    return eng._features(row, hs, as_, lg, tier)


def apply_injury_adjustment(feats: dict, home_pen: float, away_pen: float) -> dict:
    f = dict(feats)
    f["home_elo"] -= home_pen
    f["away_elo"] -= away_pen
    f["elo_diff"] = f["home_elo"] - f["away_elo"]
    return f


def _factors(f: dict, home: str, away: str, hrep: dict, arep: dict, ex: dict) -> List[dict]:
    nz = lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))
    out: List[dict] = []
    add = lambda cat, detail: out.append({"category": cat, "detail": detail})

    # 1 injuries
    for name, rep in ((home, hrep), (away, arep)):
        if rep["has_data"] and rep["count"]:
            top = ", ".join(f'{p["player"]} ({p["status"]})' for p in rep["players"][:3])
            add("Injuries", f"{name}: {rep['count']} unavailable ({top}); rating adjusted by -{rep['elo_penalty']:.0f}.")
        elif not rep["has_data"]:
            add("Injuries", f"{name}: no injury data loaded (EPL: run sync_data --fpl; EFL: POST /api/injuries) — not factored in.")
    # 2 home advantage
    add("Home advantage", f"{home} take {f['home_team_hfa']:.2f} more points/game at home than away; league-wide home win rate {f['league_home_win_rate']*100:.0f}%.")
    # 3 fatigue
    for name, side in ((home, "home"), (away, "away")):
        rd, g7 = f[f"{side}_rest_days"], f[f"{side}_games_7d"]
        if nz(rd) and (rd <= 3 or g7 >= 2):
            add("Fatigue", f"{name} played {int(g7)} game(s) in the last 7 days; only {int(rd)} days rest.")
    if nz(f["rest_diff"]) and abs(f["rest_diff"]) >= 3:
        fresher = home if f["rest_diff"] > 0 else away
        add("Fatigue", f"{fresher} have {abs(f['rest_diff']):.0f} more days of rest.")
    # 4 game factors
    if f["is_derby"]:
        add("Game factors", "Derby / local rivalry — historically less predictable.")
    if nz(f["travel_km"]) and f["travel_km"] > 250:
        add("Game factors", f"{away} travel {f['travel_km']:.0f} km.")
    if f["bad_weather"] == 1.0:
        add("Game factors", f"Poor weather (wind {f['wind_kph']:.0f} km/h, rain {f['precip_mm']:.1f} mm) — tends to reduce quality and goals.")
    elif nz(f["temp_c"]):
        add("Game factors", f"Weather: {f['temp_c']:.0f}°C, wind {f['wind_kph']:.0f} km/h, rain {f['precip_mm']:.1f} mm.")
    # 5 head to head
    if f["h2h_n"] and f["h2h_n"] > 0:
        add("Head-to-head", f"Last {int(f['h2h_n'])} meetings: {home} took {f['h2h_home_score']*100:.0f}% of the points available (avg goal diff {f['h2h_goal_diff']:+.1f}).")
    # 8 team stats
    if nz(f["xgp_diff"]) and abs(f["xgp_diff"]) > 0.25:
        better = home if f["xgp_diff"] > 0 else away
        add("Team stats", f"{better} create better chances relative to what they concede (shot-quality gap {abs(f['xgp_diff']):.2f}).")
    # 9 match situation (offence / defence)
    add("Match situation", f"Expected goals {home} {ex['exp_home_goals']:.2f} – {ex['exp_away_goals']:.2f} {away} "
        f"(attack {f['home_attack']:.2f}/{f['away_attack']:.2f}, leakiness {f['home_defence']:.2f}/{f['away_defence']:.2f}; 1.00 = league average).")
    # rating + form
    d = f["elo_diff"]
    add("Team strength", f"{home if d > 0 else away} rated stronger by {abs(d):.0f} Elo points (after injury adjustment).")
    if nz(f["ppg_diff_s"]) and abs(f["ppg_diff_s"]) > 0.5:
        add("Form", f"{home if f['ppg_diff_s'] > 0 else away} in clearly better recent form ({f['home_ppg_s']:.1f} vs {f['away_ppg_s']:.1f} points/game).")
    # suspension proxy
    for name, side in ((home, "home"), (away, "away")):
        if f[f"{side}_red_last"] and f[f"{side}_red_last"] > 0:
            add("Injuries", f"{name} had a player sent off last match — likely suspension.")
    return out


def _star_players(db: Session, team_id: int, date: dt.date) -> List[dict]:
    target_season = date.year if date.month >= 7 else date.year - 1
    current_season = dt.date.today().year if dt.date.today().month >= 7 else dt.date.today().year - 1
    if target_season == current_season:
        from app.models import Player
        ps = db.query(Player).filter(Player.team_id == team_id, Player.xg.isnot(None)).all()
        ps.sort(key=lambda p: -((p.xg or 0) + (p.xa or 0)))
        if ps:
            return [{"player": p.name, "goals": p.goals, "assists": p.assists, "xg": round(p.xg or 0, 1), "xa": round(p.xa or 0, 1),
                     "status": p.status} for p in ps[:3]]

    from app.models import PlayerSeasonStats
    rows = db.query(PlayerSeasonStats).filter(
        PlayerSeasonStats.team_id == team_id, PlayerSeasonStats.season < target_season
    ).order_by(PlayerSeasonStats.season.desc()).all()
    if not rows:
        return []
    latest = rows[0].season
    rows = [p for p in rows if p.season == latest]
    rows.sort(key=lambda p: ((p.goals or 0) + (p.assists or 0), p.minutes or 0), reverse=True)
    return [{"player": p.player_name, "goals": p.goals or 0, "assists": p.assists or 0, "xg": None, "xa": None,
             "status": None, "appearances": p.appearances, "minutes": p.minutes, "season": p.season,
             "rating": p.rating} for p in rows[:3]]


def predict_matchup(db: Session, model, home: Team, away: Team, date: Optional[dt.date] = None, overrides: Optional[dict] = None,
                    use_market: bool = True, feature_overrides: Optional[dict] = None,
                    injury_penalty_override: Optional[tuple] = None) -> dict:
    """
    overrides          : weather / odds for this match (temp_c, wind_kph, precip_mm, odds_home/draw/away)
    feature_overrides  : force any engineered feature, e.g. {"home_elo": 1700, "is_derby": 1.0} (what the UI's "advanced" box sends)
    injury_penalty_override : (home_elo_penalty, away_elo_penalty) to replace the computed injury adjustment
    """
    date = date or dt.date.today()
    feats = matchup_features(db, home, away, date, overrides)
    for k, v in (feature_overrides or {}).items():
        if k in feats and v is not None:
            feats[k] = float(v)
    feats["elo_diff"] = feats["home_elo"] - feats["away_elo"]
    hrep, arep = team_injury_report(db, home.id), team_injury_report(db, away.id)
    hp, ap = injury_penalty_override if injury_penalty_override else (hrep["elo_penalty"], arep["elo_penalty"])
    adj = apply_injury_adjustment(feats, hp, ap)
    frame = pd.DataFrame([adj])
    proba = model.predict(frame, use_market=use_market)[0]
    ex = scoreline_summary(adj["exp_home_goals"], adj["exp_away_goals"])
    probs = {l: float(p) for l, p in zip(LABELS, proba)}
    market = None if np.isnan(adj["mkt_p_home"]) else {"H": adj["mkt_p_home"], "D": adj["mkt_p_draw"], "A": adj["mkt_p_away"]}
    return {
        "probabilities": probs, "predicted_outcome": max(probs, key=probs.get), "used_market_odds": market is not None and use_market,
        "market_probabilities": market, "expected_goals": ex, "factors": _factors(adj, home.name, away.name, hrep, arep, ex),
        "injuries": {"home": hrep, "away": arep},
        "key_players": {"home": _star_players(db, home.id, date), "away": _star_players(db, away.id, date)},
        "features": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in adj.items()},
    }
