"""
Tests for app.predict.build_prediction_features.
"""

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import EloHistory, Injury, Match, Player, Team
from app.predict import build_prediction_features


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_build_prediction_features_returns_all_expected_keys(db):
    home = Team(name="Arsenal", stadium_lat=51.5549, stadium_lon=-0.1084)
    away = Team(name="Chelsea", stadium_lat=51.4816, stadium_lon=-0.1909)
    db.add_all([home, away])
    db.commit()

    features = build_prediction_features(db, home, away, dt.date(2026, 1, 1))

    expected_keys = {
        "home_elo", "away_elo", "league", "home_rest_days", "away_rest_days",
        "home_win_rate_last5", "home_avg_goals_scored_last5", "home_avg_goals_conceded_last5",
        "away_win_rate_last5", "away_avg_goals_scored_last5", "away_avg_goals_conceded_last5",
        "h2h_home_win_rate", "h2h_avg_goal_diff", "h2h_matches_played",
        "travel_distance_km", "is_derby",
        "home_injury_count", "home_injury_importance_sum",
        "away_injury_count", "away_injury_importance_sum",
    }
    assert set(features.keys()) == expected_keys


def test_build_prediction_features_uses_current_elo(db):
    home = Team(name="Arsenal")
    away = Team(name="Chelsea")
    db.add_all([home, away])
    db.commit()

    db.add(EloHistory(team_id=home.id, date=dt.date(2025, 12, 1), rating=1650.0))
    db.commit()

    features = build_prediction_features(db, home, away, dt.date(2026, 1, 1))
    assert features["home_elo"] == pytest.approx(1650.0)
    assert features["away_elo"] == pytest.approx(1500.0)  # no Elo history -> base rating fallback


def test_build_prediction_features_defaults_to_today_when_no_date_given(db):
    home = Team(name="Arsenal")
    away = Team(name="Chelsea")
    db.add_all([home, away])
    db.commit()

    features = build_prediction_features(db, home, away, match_date=None)
    # No specific assertion on values — just confirming it runs without error
    # and produces a valid dict when no date is passed.
    assert features["home_elo"] == pytest.approx(1500.0)


def test_build_prediction_features_uses_current_injuries_not_match_specific(db):
    home = Team(name="Arsenal")
    away = Team(name="Chelsea")
    db.add_all([home, away])
    db.commit()

    star = Player(name="Star Player", team_id=home.id, importance_weight=0.9)
    db.add(star)
    db.commit()
    # Injury with NO match_missed_id (since there's no match row for a future fixture) —
    # this must still be picked up via the date-window logic, not match_missed_id.
    db.add(Injury(player_id=star.id, date_out=dt.date(2025, 12, 20), date_back=dt.date(2026, 1, 15), match_missed_id=None))
    db.commit()

    features = build_prediction_features(db, home, away, dt.date(2026, 1, 1))
    assert features["home_injury_count"] == 1
    assert features["home_injury_importance_sum"] == pytest.approx(0.9)


def test_build_prediction_features_travel_distance_uses_home_stadium(db):
    home = Team(name="Arsenal", stadium_lat=51.5549, stadium_lon=-0.1084)
    away = Team(name="Man United", stadium_lat=53.4631, stadium_lon=-2.2913)
    db.add_all([home, away])
    db.commit()

    features = build_prediction_features(db, home, away, dt.date(2026, 1, 1))
    assert 255 < features["travel_distance_km"] < 270  # known ~262km London-Manchester distance


def test_build_prediction_features_derby_flag(db):
    home = Team(name="Arsenal")
    away = Team(name="Tottenham Hotspur")
    db.add_all([home, away])
    db.commit()

    features = build_prediction_features(db, home, away, dt.date(2026, 1, 1))
    assert features["is_derby"] is True
