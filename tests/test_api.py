"""
Tests for app.api using FastAPI's TestClient — exercises the full stack
(database -> feature building -> model -> response) in-process, without
needing a running server.

Includes a regression test for a real bug found during manual end-to-end
testing: a team with no geocoded stadium coordinates produced a
travel_distance_km of None, which pandas inferred as dtype 'object' for a
single-row DataFrame, which XGBoost rejected outright. Fixed in both
app.api.predict and app.train_model.prepare_features via pd.to_numeric
coercion — this test exists so that fix can't silently regress.
"""

import datetime as dt

import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import api as api_module
from app.database import Base
from app.data_split import time_based_split
from app.models import EloHistory, Team
from app.synthetic_data import generate_synthetic_feature_table
from app.train_model import prepare_features, train_xgboost


@pytest.fixture()
def test_db_session(tmp_path):
    # A FILE-based SQLite DB, not :memory: — FastAPI's TestClient runs
    # endpoint code in a worker thread, and an in-memory SQLite connection
    # is bound to the thread that created it, causing
    # "SQLite objects created in a thread can only be used in that same
    # thread" errors. A file-based DB lets each thread open its own
    # connection to the same underlying data.
    db_path = tmp_path / "test_api.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(test_db_session, tmp_path, monkeypatch):
    """
    Builds a small real team set (some WITH stadium coordinates, some
    WITHOUT — deliberately, to exercise the missing-coordinate dtype bug),
    trains a tiny real model, and wires the FastAPI app to use this test
    database and model instead of the real ones.
    """
    # Team WITH coordinates
    arsenal = Team(name="Arsenal", stadium_lat=51.5549, stadium_lon=-0.1084, league="Premier League")
    # Team WITHOUT coordinates — this is what triggered the dtype bug
    man_utd = Team(name="Manchester United", league="Premier League")
    chelsea = Team(name="Chelsea", stadium_lat=51.4816, stadium_lon=-0.1909, league="Premier League")
    test_db_session.add_all([arsenal, man_utd, chelsea])
    test_db_session.commit()

    test_db_session.add(EloHistory(team_id=arsenal.id, date=dt.date(2026, 1, 1), rating=1600))
    test_db_session.add(EloHistory(team_id=man_utd.id, date=dt.date(2026, 1, 1), rating=1550))
    test_db_session.commit()

    # Train a tiny real model on synthetic data (fast, and exercises the real training code path)
    synth_df = generate_synthetic_feature_table(n_matches=300, seed=99)
    train_df, val_df, _ = time_based_split(synth_df, train_frac=0.8, val_frac=0.15)
    X_train, feature_cols = prepare_features(train_df)
    X_val, _ = prepare_features(val_df)
    model = train_xgboost(X_train, train_df["result"], X_val=X_val, y_val=val_df["result"])

    model_path = tmp_path / "test_model.joblib"
    joblib.dump({"model": model, "feature_columns": feature_cols}, model_path)

    # Wire the app to our test database and model instead of the real ones.
    monkeypatch.setattr(api_module, "SessionLocal", lambda: test_db_session)
    monkeypatch.setattr(api_module, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(api_module, "_model_cache", None)
    monkeypatch.setattr(api_module, "_feature_columns_cache", None)
    # db.close() is called in finally blocks throughout the API — make it a no-op
    # so the shared test session stays usable across multiple requests in one test.
    monkeypatch.setattr(test_db_session, "close", lambda: None)

    test_client = TestClient(api_module.app)
    test_client.team_ids = {"arsenal": arsenal.id, "man_utd": man_utd.id, "chelsea": chelsea.id}
    return test_client


# ---------------------------------------------------------------------------
# /api/teams
# ---------------------------------------------------------------------------
def test_list_teams_returns_all_teams(client):
    response = client.get("/api/teams")
    assert response.status_code == 200
    names = {t["name"] for t in response.json()}
    assert names == {"Arsenal", "Manchester United", "Chelsea"}


# ---------------------------------------------------------------------------
# /api/predict — the core regression test
# ---------------------------------------------------------------------------
def test_predict_works_when_away_team_has_no_stadium_coordinates(client):
    """
    THE REGRESSION TEST: Manchester United has no stadium_lat/lon in this
    fixture. Before the fix, this raised a 500 error from XGBoost
    ('DataFrame.dtypes for data must be int, float, bool or category...
    Invalid columns: travel_distance_km: object'). Must now return 200.
    """
    response = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["man_utd"],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["home_team"] == "Arsenal"
    assert data["away_team"] == "Manchester United"


def test_predict_probabilities_sum_to_one(client):
    response = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["chelsea"],
    })
    assert response.status_code == 200
    probs = response.json()["probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_rejects_team_playing_itself(client):
    response = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["arsenal"],
    })
    assert response.status_code == 400


def test_predict_rejects_unknown_team_id(client):
    response = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": 999999,
    })
    assert response.status_code == 404


def test_predict_returns_key_factors_grounded_in_elo_gap(client):
    """Arsenal (1600) vs Man Utd (1550) — a real 50pt Elo gap should surface as a key factor."""
    response = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["man_utd"],
    })
    data = response.json()
    labels = [f["label"] for f in data["key_factors"]]
    assert "Rating gap" in labels


def test_predict_returns_valid_predicted_outcome(client):
    response = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["chelsea"],
    })
    data = response.json()
    assert data["predicted_outcome"] in ("H", "D", "A")
    # predicted_outcome must match the argmax of the returned probabilities
    probs = data["probabilities"]
    assert data["predicted_outcome"] == max(probs, key=probs.get)


# ---------------------------------------------------------------------------
# Missing model file
# ---------------------------------------------------------------------------
def test_predict_returns_clear_error_when_model_missing(test_db_session, monkeypatch):
    arsenal = Team(name="Arsenal")
    chelsea = Team(name="Chelsea")
    test_db_session.add_all([arsenal, chelsea])
    test_db_session.commit()

    monkeypatch.setattr(api_module, "SessionLocal", lambda: test_db_session)
    monkeypatch.setattr(api_module, "MODEL_PATH", "/nonexistent/path/model.joblib")
    monkeypatch.setattr(api_module, "_model_cache", None)
    monkeypatch.setattr(api_module, "_feature_columns_cache", None)
    monkeypatch.setattr(test_db_session, "close", lambda: None)

    test_client = TestClient(api_module.app)
    response = test_client.post("/api/predict", json={"home_team_id": arsenal.id, "away_team_id": chelsea.id})
    assert response.status_code == 503
    assert "No trained model found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# PWA static assets
# ---------------------------------------------------------------------------
def test_root_serves_frontend_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_manifest_json_served_with_correct_content_type(client):
    response = client.get("/manifest.json")
    assert response.status_code == 200
    assert response.json()["name"] == "Match Predictor"


def test_service_worker_served_at_root_scope_with_js_content_type(client):
    """Must be served at /sw.js (not /static/sw.js) for its default control
    scope to cover the whole app — see the comment in app/api.py."""
    response = client.get("/sw.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_icons_served_with_png_content_type(client):
    for path in ["/icon-192.png", "/icon-512.png", "/icon-512-maskable.png"]:
        response = client.get(path)
        assert response.status_code == 200, f"{path} failed"
        assert "image/png" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# League filtering and team creation
# ---------------------------------------------------------------------------
def test_list_leagues_returns_distinct_leagues(client):
    response = client.get("/api/leagues")
    assert response.status_code == 200
    assert set(response.json()) == {"Premier League"}  # all fixture teams share this league value


def test_list_teams_filters_by_league(client, test_db_session):
    # Add a team in a different league to prove filtering actually excludes it.
    kpl_team = Team(name="Gor Mahia", league="KPL")
    test_db_session.add(kpl_team)
    test_db_session.commit()

    response = client.get("/api/teams", params={"league": "KPL"})
    assert response.status_code == 200
    names = {t["name"] for t in response.json()}
    assert names == {"Gor Mahia"}


def test_create_team_returns_new_team_with_id(client):
    response = client.post("/api/teams", json={"name": "Brand New FC", "league": "KPL"})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Brand New FC"
    assert data["league"] == "KPL"
    assert isinstance(data["id"], int)


def test_create_team_rejects_duplicate_name(client):
    client.post("/api/teams", json={"name": "Duplicate FC", "league": "KPL"})
    response = client.post("/api/teams", json={"name": "Duplicate FC", "league": "KPL"})
    assert response.status_code == 409


def test_newly_created_team_appears_in_teams_list(client):
    client.post("/api/teams", json={"name": "Fresh Arrivals FC", "league": "KPL"})
    response = client.get("/api/teams", params={"league": "KPL"})
    names = {t["name"] for t in response.json()}
    assert "Fresh Arrivals FC" in names


# ---------------------------------------------------------------------------
# Manual overrides (for brand-new teams with no real history)
# ---------------------------------------------------------------------------
def test_predict_with_home_overrides_changes_the_prediction(client):
    """A very strong manual Elo override for the home team should push the
    prediction meaningfully toward a home win compared to no override."""
    baseline = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["chelsea"],
    }).json()

    boosted = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["chelsea"],
        "home_overrides": {"elo": 2200},  # a huge, unmistakable Elo boost
    }).json()

    assert boosted["probabilities"]["H"] > baseline["probabilities"]["H"]


def test_predict_with_derby_override(client):
    response = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["chelsea"],
        "is_derby": True,
    })
    assert response.status_code == 200
    labels = [f["label"] for f in response.json()["key_factors"]]
    assert "Derby match" in labels


def test_predict_with_travel_distance_override(client):
    response = client.post("/api/predict", json={
        "home_team_id": client.team_ids["arsenal"],
        "away_team_id": client.team_ids["chelsea"],
        "travel_distance_km": 5000,
    })
    assert response.status_code == 200
    labels = [f["label"] for f in response.json()["key_factors"]]
    assert "Travel distance" in labels


def test_predict_manual_override_workflow_for_brand_new_team(client):
    """
    The realistic end-to-end flow: create a brand-new team, then predict
    for it using manual overrides since it has no real history yet.
    """
    create_response = client.post("/api/teams", json={"name": "Newly Promoted FC", "league": "Premier League"})
    new_team_id = create_response.json()["id"]

    response = client.post("/api/predict", json={
        "home_team_id": new_team_id,
        "away_team_id": client.team_ids["arsenal"],
        "home_overrides": {
            "elo": 1400,
            "win_rate_last5": 0.6,
            "rest_days": 7,
        },
    })
    assert response.status_code == 200
    data = response.json()
    assert data["home_team"] == "Newly Promoted FC"
    assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
