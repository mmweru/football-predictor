"""
Tests for app.elo — the pure math functions plus a full build against an
in-memory database.

Run with: pytest -v
"""

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.elo import (
    BASE_RATING,
    build_elo_history,
    expected_score,
    regress_toward_mean,
    update_elo,
)
from app.models import EloHistory, Match, Team


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


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------
def test_expected_score_equal_ratings_is_fifty_fifty():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_favors_higher_rating():
    assert expected_score(1700, 1500) > 0.5
    assert expected_score(1500, 1700) < 0.5


def test_expected_score_symmetry():
    a = expected_score(1600, 1400)
    b = expected_score(1400, 1600)
    assert a + b == pytest.approx(1.0)


def test_update_elo_winner_gains_loser_loses():
    new_home, new_away = update_elo(1500, 1500, result_home=1.0, k=20, home_advantage=0)
    assert new_home > 1500
    assert new_away < 1500
    # Zero-sum when ratings start equal and there's no home advantage bias:
    assert (new_home - 1500) == pytest.approx(-(new_away - 1500))


def test_update_elo_draw_between_equal_teams_is_unchanged():
    new_home, new_away = update_elo(1500, 1500, result_home=0.5, k=20, home_advantage=0)
    assert new_home == pytest.approx(1500)
    assert new_away == pytest.approx(1500)


def test_update_elo_upset_moves_rating_more_than_expected_result():
    # Big underdog (1400) beating a big favorite (1700) should move ratings
    # by close to the full K, since the upset was highly unexpected.
    new_home, new_away = update_elo(1400, 1700, result_home=1.0, k=20, home_advantage=0)
    home_gain = new_home - 1400
    assert home_gain > 15  # expected score was low, so actual gain should be close to K=20


def test_home_advantage_increases_home_expected_score():
    home_elo, away_elo = 1500, 1500
    exp_no_bonus = expected_score(home_elo, away_elo)
    exp_with_bonus = expected_score(home_elo + 70, away_elo)
    assert exp_with_bonus > exp_no_bonus


def test_regress_toward_mean_pulls_rating_closer_to_base():
    high_rating = 1800.0
    regressed = regress_toward_mean(high_rating)
    assert BASE_RATING < regressed < high_rating

    low_rating = 1200.0
    regressed_low = regress_toward_mean(low_rating)
    assert low_rating < regressed_low < BASE_RATING


# ---------------------------------------------------------------------------
# Full build against a database
# ---------------------------------------------------------------------------
def _make_team(db, name):
    t = Team(name=name)
    db.add(t)
    db.commit()
    return t


def test_build_elo_history_new_teams_start_at_base_rating(db):
    home = _make_team(db, "Home FC")
    away = _make_team(db, "Away FC")
    db.add(Match(date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id, home_score=1, away_score=1))
    db.commit()

    build_elo_history(db, k=20, home_advantage=70)

    home_row = db.query(EloHistory).filter_by(team_id=home.id).one()
    away_row = db.query(EloHistory).filter_by(team_id=away.id).one()
    # First-ever match: pre-match rating must be the base rating for both.
    assert home_row.rating == pytest.approx(BASE_RATING)
    assert away_row.rating == pytest.approx(BASE_RATING)


def test_build_elo_history_processes_chronologically_not_insertion_order(db):
    home = _make_team(db, "Team A")
    away = _make_team(db, "Team B")
    # Insert the LATER match first, to prove the function sorts by date, not by row order.
    db.add(Match(date=dt.date(2024, 3, 1), home_team_id=home.id, away_team_id=away.id, home_score=1, away_score=0))
    db.add(Match(date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id, home_score=0, away_score=2))
    db.commit()

    build_elo_history(db, k=20, home_advantage=70)

    jan_row = db.query(EloHistory).filter_by(team_id=home.id, date=dt.date(2024, 1, 1)).one()
    mar_row = db.query(EloHistory).filter_by(team_id=home.id, date=dt.date(2024, 3, 1)).one()
    # Jan match must show the team at base rating (first ever match).
    assert jan_row.rating == pytest.approx(BASE_RATING)
    # March pre-match rating must reflect the Jan result (a loss), so it should be BELOW base.
    assert mar_row.rating < BASE_RATING


def test_build_elo_history_unscored_matches_are_ignored(db):
    home = _make_team(db, "Team X")
    away = _make_team(db, "Team Y")
    # Future/unplayed fixture with no score yet — must not be processed.
    db.add(Match(date=dt.date(2024, 6, 1), home_team_id=home.id, away_team_id=away.id, home_score=None, away_score=None))
    db.commit()

    rows_written = build_elo_history(db, k=20, home_advantage=70)
    assert rows_written == 0


def test_build_elo_history_is_idempotent(db):
    home = _make_team(db, "Repeat FC")
    away = _make_team(db, "Other FC")
    db.add(Match(date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id, home_score=2, away_score=0))
    db.commit()

    build_elo_history(db, k=20, home_advantage=70)
    first_count = db.query(EloHistory).count()

    build_elo_history(db, k=20, home_advantage=70)  # re-run without reset
    second_count = db.query(EloHistory).count()

    assert first_count == second_count == 2  # upserted in place, not duplicated
