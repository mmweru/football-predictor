"""
FastAPI backend.

  GET  /api/leagues              the 4 supported leagues (EPL, Championship, League One, League Two) + active club counts
  GET  /api/teams?league=        ACTIVE clubs only (this season) — so the Premier League lists exactly 20
  POST /api/teams                add a club (league must be one of the 4)
  POST /api/predict              full prediction for any two clubs (+ explanation of all nine metric groups)
  GET  /api/fixtures?league=     upcoming fixtures with their stored predictions
  GET  /api/accuracy             live track record of graded predictions
  POST /api/injuries             record an injury manually (Championship / League One / League Two have no free feed)
  GET  /api/model                which model is loaded, when it was trained, its held-out scores
  GET  /                         the frontend

Run:  uvicorn app.api:app --reload      (needs model_bundle.joblib from `python -m app.train_ensemble`)
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.database import SessionLocal
from app.grade_predictions import track_record
from app.injuries import ELO_PER_IMPORTANCE, MAX_ELO_PENALTY, add_manual_injury
from app.leagues import LEAGUE_NAMES, LEAGUES, canonical_league
from app.models import Match, Prediction, Team
from app.predict_live import predict_matchup
from app.schemas import TeamCreate

MODEL_PATH = os.environ.get("MODEL_PATH", "model_bundle.joblib")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Football Match Predictor")
_bundle: Dict[str, object] = {}


def _load_model():
    if not _bundle or _bundle.get("mtime") != _mtime():
        if not Path(MODEL_PATH).exists():
            raise HTTPException(503, f"No trained model at '{MODEL_PATH}'. Run `python -m app.train_ensemble` first.")
        saved = joblib.load(MODEL_PATH)
        _bundle.update(model=saved["model"], metadata=saved["metadata"], mtime=_mtime())
    return _bundle["model"], _bundle["metadata"]


def _mtime() -> Optional[float]:
    return os.path.getmtime(MODEL_PATH) if Path(MODEL_PATH).exists() else None


def _visible_teams(db, league: Optional[str] = None) -> List[Team]:
    """Active clubs in one of the 4 supported leagues. This ONE function is why nothing else (Kenyan clubs, relegated clubs) ever leaks into the UI."""
    wanted = canonical_league(league) if league else None
    if league and wanted is None:
        return []
    out = []
    for t in db.query(Team).filter(Team.is_active.is_(True)).order_by(Team.name).all():
        cl = canonical_league(t.league)
        if cl and (wanted is None or cl == wanted):
            out.append(t)
    return out


class TeamOut(BaseModel):
    id: int
    name: str
    league: Optional[str] = None


class LeagueOut(BaseModel):
    name: str
    badge: str
    tier: int
    teams: int
    expected_teams: int


class TeamCreateRequest(BaseModel):
    name: str
    league: str
    stadium_lat: Optional[float] = None
    stadium_lon: Optional[float] = None


class ManualOverrides(BaseModel):
    """Optional per-side overrides from the UI's advanced box. Anything left None keeps the computed value."""
    elo: Optional[float] = None
    rest_days: Optional[int] = None
    win_rate_last5: Optional[float] = None
    avg_goals_scored_last5: Optional[float] = None      # accepted for backwards compatibility; no longer a model input
    avg_goals_conceded_last5: Optional[float] = None    # accepted for backwards compatibility; no longer a model input
    injury_count: Optional[int] = None
    injury_importance_sum: Optional[float] = None


class PredictRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    match_date: Optional[dt.date] = None
    is_derby: Optional[bool] = None
    travel_distance_km: Optional[float] = None
    home_overrides: Optional[ManualOverrides] = None
    away_overrides: Optional[ManualOverrides] = None
    temp_c: Optional[float] = None
    wind_kph: Optional[float] = None
    precip_mm: Optional[float] = None
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None


class PredictionFactor(BaseModel):
    label: str
    detail: str


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    match_date: dt.date
    probabilities: dict
    predicted_outcome: str
    key_factors: List[PredictionFactor]
    expected_goals: Optional[dict] = None
    used_market_odds: bool = False
    injuries: Optional[dict] = None
    key_players: Optional[dict] = None


class InjuryRequest(BaseModel):
    team_id: int
    player_name: str
    importance: float = 0.5          # 0..1: 1.0 = star striker, 0.4 = regular starter, 0.1 = fringe
    reason: Optional[str] = None
    date_back: Optional[dt.date] = None


@app.get("/api/leagues", response_model=List[LeagueOut])
def list_leagues():
    db = SessionLocal()
    try:
        counts = {l.name: len(_visible_teams(db, l.name)) for l in LEAGUES}
        return [LeagueOut(name=l.name, badge=l.badge, tier=l.tier, teams=counts[l.name], expected_teams=l.expected_teams)
                for l in LEAGUES if counts[l.name] > 0]
    finally:
        db.close()


@app.get("/api/teams", response_model=List[TeamOut])
def list_teams(league: Optional[str] = None):
    db = SessionLocal()
    try:
        return [TeamOut(id=t.id, name=t.name, league=canonical_league(t.league)) for t in _visible_teams(db, league)]
    finally:
        db.close()


@app.post("/api/teams", response_model=TeamOut, status_code=201)
def create_team(req: TeamCreateRequest):
    league = canonical_league(req.league)
    if league is None:
        raise HTTPException(400, f"league must be one of {LEAGUE_NAMES}")
    db = SessionLocal()
    try:
        existing = db.query(Team).filter_by(name=req.name).one_or_none()
        if existing is not None:
            raise HTTPException(409, f"A team named {req.name!r} already exists (id={existing.id}).")
        team = Team(**TeamCreate(name=req.name, league=league, stadium_lat=req.stadium_lat, stadium_lon=req.stadium_lon).model_dump(), is_active=True)
        db.add(team)
        db.commit()
        return TeamOut(id=team.id, name=team.name, league=league)
    finally:
        db.close()


def _feature_overrides(req: PredictRequest) -> dict:
    f: dict = {}
    for side, ov in (("home", req.home_overrides), ("away", req.away_overrides)):
        if ov is None:
            continue
        if ov.elo is not None:
            f[f"{side}_elo"] = ov.elo
        if ov.rest_days is not None:
            f[f"{side}_rest_days"] = ov.rest_days
            f[f"{side}_short_rest"] = 1.0 if ov.rest_days <= 3 else 0.0
        if ov.win_rate_last5 is not None:
            f[f"{side}_win5"] = ov.win_rate_last5
    if req.is_derby is not None:
        f["is_derby"] = 1.0 if req.is_derby else 0.0
    if req.travel_distance_km is not None:
        f["travel_km"] = req.travel_distance_km
    return f


def _injury_override(req: PredictRequest):
    def pen(ov):
        if ov is None:
            return None
        if ov.injury_importance_sum is not None:
            return min(ov.injury_importance_sum * ELO_PER_IMPORTANCE, MAX_ELO_PENALTY)
        if ov.injury_count is not None:
            return min(ov.injury_count * 0.4 * ELO_PER_IMPORTANCE, MAX_ELO_PENALTY)
        return None
    h, a = pen(req.home_overrides), pen(req.away_overrides)
    return None if h is None and a is None else (h or 0.0, a or 0.0)


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.home_team_id == req.away_team_id:
        raise HTTPException(400, "A team cannot play itself.")
    model, _ = _load_model()
    date = req.match_date or dt.date.today()
    db = SessionLocal()
    try:
        home, away = db.get(Team, req.home_team_id), db.get(Team, req.away_team_id)
        if home is None or away is None:
            raise HTTPException(404, "One or both team IDs were not found.")
        ctx = {k: v for k, v in dict(temp_c=req.temp_c, wind_kph=req.wind_kph, precip_mm=req.precip_mm, odds_home=req.odds_home,
                                     odds_draw=req.odds_draw, odds_away=req.odds_away).items() if v is not None}
        r = predict_matchup(db, model, home, away, date, ctx, feature_overrides=_feature_overrides(req),
                            injury_penalty_override=_injury_override(req))
        ex = r["expected_goals"]
        factors = [PredictionFactor(label=f["category"], detail=f["detail"]) for f in r["factors"]]
        factors.append(PredictionFactor(
            label="Likely scorelines",
            detail=f"Most likely {', '.join(s['score'] + f' ({s['prob']*100:.0f}%)' for s in ex['most_likely_scores'])}; "
                   f"over 2.5 goals {ex['over_2_5']*100:.0f}%, both teams score {ex['btts']*100:.0f}%."))
        for name, side in ((home.name, "home"), (away.name, "away")):
            kp = r["key_players"][side]
            if kp:
                details = []
                for player in kp:
                    detail = f"{player['player']} {player['goals'] or 0}G/{player['assists'] or 0}A"
                    if player.get("xg") is not None:
                        detail += f" (xG {player['xg']})"
                    elif player.get("appearances") is not None:
                        detail += f" ({player['appearances']} apps, {player['minutes'] or 0} min; {player['season']}/{(player['season'] + 1) % 100:02d})"
                    details.append(detail)
                factors.append(PredictionFactor(label="Player stats", detail=f"{name} key players: " + ", ".join(details)))
        return PredictResponse(home_team=home.name, away_team=away.name, match_date=date, probabilities=r["probabilities"],
                               predicted_outcome=r["predicted_outcome"], key_factors=factors, expected_goals=ex,
                               used_market_odds=r["used_market_odds"], injuries=r["injuries"], key_players=r["key_players"])
    finally:
        db.close()


@app.get("/api/fixtures")
def upcoming_fixtures(league: Optional[str] = None, days: int = 14):
    wanted = canonical_league(league) if league else None
    db = SessionLocal()
    try:
        today = dt.date.today()
        teams = {t.id: t for t in db.query(Team).all()}
        q = db.query(Match).filter(Match.home_score.is_(None), Match.date >= today, Match.date <= today + dt.timedelta(days=days)).order_by(Match.date)
        out = []
        for m in q.all():
            cl = canonical_league(m.competition)
            h, a = teams[m.home_team_id], teams[m.away_team_id]
            if cl is None or (wanted and cl != wanted) or not (h.is_active and a.is_active):
                continue
            pred = db.query(Prediction).filter_by(match_id=m.id).order_by(Prediction.created_at.desc()).first()
            out.append({"match_id": m.id, "date": m.date.isoformat(), "league": cl, "home_team": h.name, "away_team": a.name,
                        "home_team_id": h.id, "away_team_id": a.id,
                        "prediction": None if pred is None else {"H": pred.p_home, "D": pred.p_draw, "A": pred.p_away,
                                                                 "predicted": pred.predicted, "exp_home_goals": pred.exp_home_goals,
                                                                 "exp_away_goals": pred.exp_away_goals}})
        return out
    finally:
        db.close()


@app.get("/api/accuracy")
def accuracy():
    db = SessionLocal()
    try:
        return track_record(db)
    finally:
        db.close()


@app.post("/api/injuries", status_code=201)
def post_injury(req: InjuryRequest):
    db = SessionLocal()
    try:
        if db.get(Team, req.team_id) is None:
            raise HTTPException(404, "Unknown team_id")
        inj = add_manual_injury(db, req.team_id, req.player_name, req.importance, req.reason, date_back=req.date_back)
        return {"injury_id": inj.id, "player": req.player_name}
    finally:
        db.close()


@app.get("/api/model")
def model_info():
    _, meta = _load_model()
    return meta


# Serve the frontend, if the directory exists (keeps the API usable standalone during development).
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    # sw.js MUST be served from the root path: a service worker only controls the directory it is served from.
    @app.get("/sw.js")
    def serve_service_worker():
        return FileResponse(str(FRONTEND_DIR / "sw.js"), media_type="application/javascript")

    @app.get("/manifest.json")
    def serve_manifest():
        return FileResponse(str(FRONTEND_DIR / "manifest.json"), media_type="application/manifest+json")

    for _name, _mt in (("icon-192.png", "image/png"), ("icon-512.png", "image/png"), ("icon-512-maskable.png", "image/png")):
        app.add_api_route(f"/{_name}", (lambda n=_name, m=_mt: FileResponse(str(FRONTEND_DIR / n), media_type=m)), methods=["GET"])
