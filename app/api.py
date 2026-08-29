"""
FastAPI backend for the match predictor frontend.

Serves:
  GET  /api/teams              -> list of teams for the dropdown selectors
  POST /api/predict             -> prediction for a hypothetical home/away matchup
  GET  /                        -> the frontend (frontend/index.html)

Run:
    uvicorn app.api:app --reload

Requires a trained model at MODEL_PATH (default: model.joblib, from
app.train_model --model-out). Prediction requests fail with a clear error
if the model file isn't found, rather than crashing at import time — so
you can still browse the frontend/teams list before training is done.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.database import SessionLocal
from app.models import Team
from app.predict import build_prediction_features
from app.schemas import TeamCreate
from app.train_model import CATEGORICAL_FEATURE_COLUMNS, OUTCOME_LABELS

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Football Match Predictor")

_model_cache = None
_feature_columns_cache = None
_category_dtypes_cache = None


def _load_model():
    """Lazily loads and caches the trained model. Raises a clear HTTPException if missing, rather than failing at import time."""
    global _model_cache, _feature_columns_cache, _category_dtypes_cache
    if _model_cache is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=f"No trained model found at '{MODEL_PATH}'. Run `python -m app.train_model --model-out {MODEL_PATH}` first.",
            )
        saved = joblib.load(MODEL_PATH)
        _model_cache = saved["model"]
        _feature_columns_cache = saved["feature_columns"]
        _category_dtypes_cache = saved.get("category_dtypes", {})  # older saved models may not have this key
    return _model_cache, _feature_columns_cache, _category_dtypes_cache


class TeamOut(BaseModel):
    id: int
    name: str
    league: Optional[str] = None


class TeamCreateRequest(BaseModel):
    name: str
    league: str  # e.g. "EPL" or "KPL" — required, since this is how the UI's league picker filters teams
    stadium_lat: Optional[float] = None
    stadium_lon: Optional[float] = None


class ManualOverrides(BaseModel):
    """
    Optional manual feature values for one side of a matchup — used when a
    team is brand new (just created via POST /api/teams) or otherwise has
    too little history for the computed features to mean anything. Any
    field left as None here falls back to the normally-computed value;
    only fields you actually provide override the computed ones.
    """
    elo: Optional[float] = None
    rest_days: Optional[int] = None
    win_rate_last5: Optional[float] = None
    avg_goals_scored_last5: Optional[float] = None
    avg_goals_conceded_last5: Optional[float] = None
    injury_count: Optional[int] = None
    injury_importance_sum: Optional[float] = None


class PredictRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    match_date: Optional[dt.date] = None  # defaults to today if omitted
    is_derby: Optional[bool] = None  # override the auto-detected derby flag, e.g. for a rivalry not in app/derbies.py
    travel_distance_km: Optional[float] = None  # override computed travel distance
    home_overrides: Optional[ManualOverrides] = None
    away_overrides: Optional[ManualOverrides] = None


class PredictionFactor(BaseModel):
    label: str
    detail: str


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    match_date: dt.date
    probabilities: dict  # {'H': .., 'D': .., 'A': ..}
    predicted_outcome: str
    key_factors: List[PredictionFactor]


@app.get("/api/leagues", response_model=List[str])
def list_leagues():
    db = SessionLocal()
    try:
        rows = db.query(Team.league).distinct().all()
        return sorted({r[0] for r in rows if r[0]})
    finally:
        db.close()


@app.get("/api/teams", response_model=List[TeamOut])
def list_teams(league: Optional[str] = None):
    db = SessionLocal()
    try:
        query = db.query(Team)
        if league:
            query = query.filter(Team.league == league)
        teams = query.order_by(Team.name).all()
        return [TeamOut(id=t.id, name=t.name, league=t.league) for t in teams]
    finally:
        db.close()


@app.post("/api/teams", response_model=TeamOut, status_code=201)
def create_team(req: TeamCreateRequest):
    """
    Creates a new team on the fly — for a completely new/unlisted team not
    already in the database. Use the returned id with home_overrides /
    away_overrides in POST /api/predict to supply manual feature values,
    since a brand-new team has no history for the normal features to
    compute from.
    """
    db = SessionLocal()
    try:
        existing = db.query(Team).filter_by(name=req.name).one_or_none()
        if existing is not None:
            raise HTTPException(status_code=409, detail=f"A team named {req.name!r} already exists (id={existing.id}).")

        payload = TeamCreate(name=req.name, league=req.league, stadium_lat=req.stadium_lat, stadium_lon=req.stadium_lon)
        team = Team(**payload.model_dump())
        db.add(team)
        db.commit()
        return TeamOut(id=team.id, name=team.name, league=team.league)
    finally:
        db.close()


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.home_team_id == req.away_team_id:
        raise HTTPException(status_code=400, detail="A team cannot play itself.")

    model, feature_columns, category_dtypes = _load_model()
    match_date = req.match_date or dt.date.today()

    db = SessionLocal()
    try:
        home = db.query(Team).filter_by(id=req.home_team_id).one_or_none()
        away = db.query(Team).filter_by(id=req.away_team_id).one_or_none()
        if home is None or away is None:
            raise HTTPException(status_code=404, detail="One or both team IDs were not found.")

        feature_dict = build_prediction_features(db, home, away, match_date)

        # Apply manual overrides, if provided — e.g. for a brand-new team
        # with no real history, where the computed features are just
        # neutral defaults. Only fields the caller actually set are applied;
        # everything else keeps the computed value.
        _apply_overrides(feature_dict, "home", req.home_overrides)
        _apply_overrides(feature_dict, "away", req.away_overrides)
        if req.is_derby is not None:
            feature_dict["is_derby"] = req.is_derby
        if req.travel_distance_km is not None:
            feature_dict["travel_distance_km"] = req.travel_distance_km

        # Build a single-row DataFrame with columns in the EXACT order/set the
        # model was trained on — missing engineered columns (there shouldn't
        # be any, but this guards against a stale model file) default to NaN,
        # which XGBoost handles natively.
        row = {col: feature_dict.get(col) for col in feature_columns}
        X = pd.DataFrame([row])
        for col in X.columns:
            if col in CATEGORICAL_FEATURE_COLUMNS:
                # MUST use the exact category set saved from training, not
                # infer fresh from this single row — otherwise a model
                # trained with enable_categorical=True can raise or behave
                # inconsistently on a category it encoded differently (or
                # not at all) during training.
                dtype = category_dtypes.get(col) if category_dtypes else None
                X[col] = X[col].astype(dtype) if dtype is not None else X[col].astype("category")
            elif X[col].dtype == bool:
                X[col] = X[col].astype(int)
            else:
                # Coerces None -> NaN and ensures a proper numeric dtype.
                # Without this, a column that's None for this single row
                # (e.g. travel_distance_km when a team has no geocoded
                # stadium yet) comes back as pandas dtype 'object', which
                # XGBoost rejects outright rather than treating as missing.
                X[col] = pd.to_numeric(X[col], errors="coerce")

        proba = model.predict_proba(X)[0]
        probabilities = dict(zip(OUTCOME_LABELS, [float(p) for p in proba]))
        predicted_outcome = max(probabilities, key=probabilities.get)

        key_factors = _explain_prediction(feature_dict, home.name, away.name)

        return PredictResponse(
            home_team=home.name,
            away_team=away.name,
            match_date=match_date,
            probabilities=probabilities,
            predicted_outcome=predicted_outcome,
            key_factors=key_factors,
        )
    finally:
        db.close()


def _apply_overrides(feature_dict: dict, side: str, overrides: Optional[ManualOverrides]) -> None:
    """Applies non-None fields from a ManualOverrides object onto feature_dict, prefixed for the given side ('home'/'away')."""
    if overrides is None:
        return
    field_map = {
        "elo": f"{side}_elo",
        "rest_days": f"{side}_rest_days",
        "win_rate_last5": f"{side}_win_rate_last5",
        "avg_goals_scored_last5": f"{side}_avg_goals_scored_last5",
        "avg_goals_conceded_last5": f"{side}_avg_goals_conceded_last5",
        "injury_count": f"{side}_injury_count",
        "injury_importance_sum": f"{side}_injury_importance_sum",
    }
    for override_field, feature_key in field_map.items():
        value = getattr(overrides, override_field)
        if value is not None:
            feature_dict[feature_key] = value


def _explain_prediction(features: dict, home_name: str, away_name: str) -> List[PredictionFactor]:
    """
    Produces plain-language "key factors" by comparing this specific
    match's feature values against neutral/average expectations —
    NOT a SHAP explanation (that's a planned future upgrade; this is a
    simpler heuristic that's still grounded in the actual feature values
    for this specific matchup, not a canned generic message).
    """
    factors = []

    elo_diff = features["home_elo"] - features["away_elo"]
    if abs(elo_diff) > 30:
        stronger, weaker = (home_name, away_name) if elo_diff > 0 else (away_name, home_name)
        factors.append(PredictionFactor(
            label="Rating gap",
            detail=f"{stronger} rated stronger than {weaker} by {abs(elo_diff):.0f} Elo points.",
        ))

    if features["is_derby"]:
        factors.append(PredictionFactor(label="Derby match", detail="These two teams are known rivals — derbies are historically less predictable."))

    if features["home_injury_count"] > 0:
        factors.append(PredictionFactor(
            label=f"{home_name} injuries",
            detail=f"{features['home_injury_count']} player(s) currently out for {home_name}.",
        ))
    if features["away_injury_count"] > 0:
        factors.append(PredictionFactor(
            label=f"{away_name} injuries",
            detail=f"{features['away_injury_count']} player(s) currently out for {away_name}.",
        ))

    if features["travel_distance_km"] is not None and features["travel_distance_km"] > 300:
        factors.append(PredictionFactor(
            label="Travel distance",
            detail=f"{away_name} travelling {features['travel_distance_km']:.0f}km for this match.",
        ))

    form_diff = features["home_win_rate_last5"] - features["away_win_rate_last5"]
    if abs(form_diff) > 0.3:
        better = home_name if form_diff > 0 else away_name
        factors.append(PredictionFactor(label="Recent form", detail=f"{better} in noticeably better recent form."))

    if features["h2h_matches_played"] > 0:
        factors.append(PredictionFactor(
            label="Head-to-head",
            detail=f"{home_name} won {features['h2h_home_win_rate']*100:.0f}% of the last {features['h2h_matches_played']} meetings.",
        ))

    if not factors:
        factors.append(PredictionFactor(label="Close matchup", detail="No single factor stands out strongly — a closely-matched fixture."))

    return factors


# Serve the frontend, if the directory exists (keeps the API usable standalone during development).
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    # sw.js MUST be served from the root path (not /static/sw.js) — a service
    # worker's default control scope is the directory it's served from, so
    # serving it from /static/ would only let it control /static/* requests,
    # not the whole app. Root-scope is what makes the PWA installable and
    # able to cache/serve the whole app shell offline.
    @app.get("/sw.js")
    def serve_service_worker():
        return FileResponse(str(FRONTEND_DIR / "sw.js"), media_type="application/javascript")

    @app.get("/manifest.json")
    def serve_manifest():
        return FileResponse(str(FRONTEND_DIR / "manifest.json"), media_type="application/manifest+json")

    @app.get("/icon-192.png")
    def serve_icon_192():
        return FileResponse(str(FRONTEND_DIR / "icon-192.png"), media_type="image/png")

    @app.get("/icon-512.png")
    def serve_icon_512():
        return FileResponse(str(FRONTEND_DIR / "icon-512.png"), media_type="image/png")

    @app.get("/icon-512-maskable.png")
    def serve_icon_512_maskable():
        return FileResponse(str(FRONTEND_DIR / "icon-512-maskable.png"), media_type="image/png")
