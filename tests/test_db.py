"""
Verification tests for the database layer.

These run against an in-memory SQLite database by default (fast, no setup
required) so you can confirm the schema and relationships are correct
before ever touching real Postgres. See the README for how to point these
same tests at your actual Postgres database instead.

Run with:
    pytest -v
"""

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models import EloHistory, Injury, Match, Player, Team
from app.schemas import (
    EloHistoryCreate,
    InjuryCreate,
    MatchCreate,
    PlayerCreate,
    TeamCreate,
)


@pytest.fixture()
def db() -> Session:
    """Fresh in-memory SQLite database for every test — fully isolated, no cleanup needed."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, future=True)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
def test_create_team(db: Session):
    payload = TeamCreate(name="Arsenal", stadium_lat=51.5549, stadium_lon=-0.1084, league="Premier League")
    team = Team(**payload.model_dump())
    db.add(team)
    db.commit()

    fetched = db.query(Team).filter_by(name="Arsenal").one()
    assert fetched.id is not None
    assert fetched.league == "Premier League"
    assert fetched.stadium_lat == pytest.approx(51.5549)


def test_team_name_must_be_unique(db: Session):
    db.add(Team(name="Chelsea"))
    db.commit()

    db.add(Team(name="Chelsea"))
    with pytest.raises(Exception):  # IntegrityError in Postgres/SQLite
        db.commit()
    db.rollback()


def test_team_pydantic_rejects_bad_latitude():
    with pytest.raises(Exception):
        TeamCreate(name="Nowhere FC", stadium_lat=999.0, stadium_lon=0.0)


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------
def test_create_match_with_team_relationship(db: Session):
    home = Team(name="Liverpool", league="Premier League")
    away = Team(name="Everton", league="Premier League")
    db.add_all([home, away])
    db.commit()

    payload = MatchCreate(
        date=dt.date(2025, 3, 1),
        home_team_id=home.id,
        away_team_id=away.id,
        home_score=2,
        away_score=1,
        competition="Premier League",
    )
    match = Match(**payload.model_dump())
    db.add(match)
    db.commit()

    fetched = db.query(Match).one()
    assert fetched.home_team.name == "Liverpool"
    assert fetched.away_team.name == "Everton"
    assert fetched.home_score == 2
    # relationship works the other way too
    assert fetched in home.home_matches


def test_duplicate_fixture_same_date_rejected(db: Session):
    home = Team(name="Home FC")
    away = Team(name="Away FC")
    db.add_all([home, away])
    db.commit()

    m1 = Match(date=dt.date(2025, 1, 1), home_team_id=home.id, away_team_id=away.id)
    db.add(m1)
    db.commit()

    m2 = Match(date=dt.date(2025, 1, 1), home_team_id=home.id, away_team_id=away.id)
    db.add(m2)
    with pytest.raises(Exception):
        db.commit()
    db.rollback()


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------
def test_create_player_linked_to_team(db: Session):
    team = Team(name="Man City")
    db.add(team)
    db.commit()

    payload = PlayerCreate(name="Erling Someone", team_id=team.id, position="Forward", importance_weight=0.9)
    player = Player(**payload.model_dump())
    db.add(player)
    db.commit()

    fetched = db.query(Player).one()
    assert fetched.team.name == "Man City"
    assert fetched.importance_weight == pytest.approx(0.9)
    assert player in team.players


def test_importance_weight_bounds_enforced_by_pydantic():
    with pytest.raises(Exception):
        PlayerCreate(name="Too Important", team_id=1, importance_weight=1.5)


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------
def test_create_injury_linked_to_player_and_match(db: Session):
    team = Team(name="Spurs")
    db.add(team)
    db.commit()

    player = Player(name="Injured Player", team_id=team.id, importance_weight=0.7)
    db.add(player)
    db.commit()

    opponent = Team(name="Opponent FC")
    db.add(opponent)
    db.commit()
    match = Match(date=dt.date(2025, 2, 1), home_team_id=team.id, away_team_id=opponent.id)
    db.add(match)
    db.commit()

    payload = InjuryCreate(
        player_id=player.id,
        date_out=dt.date(2025, 1, 25),
        date_back=dt.date(2025, 2, 15),
        match_missed_id=match.id,
    )
    injury = Injury(**payload.model_dump())
    db.add(injury)
    db.commit()

    fetched = db.query(Injury).one()
    assert fetched.player.name == "Injured Player"
    assert fetched.match_missed.date == dt.date(2025, 2, 1)


def test_injury_without_match_missed_is_allowed(db: Session):
    team = Team(name="Newcastle")
    db.add(team)
    db.commit()
    player = Player(name="Long Term Injury", team_id=team.id)
    db.add(player)
    db.commit()

    injury = Injury(player_id=player.id, date_out=dt.date(2025, 1, 1), date_back=None, match_missed_id=None)
    db.add(injury)
    db.commit()

    fetched = db.query(Injury).one()
    assert fetched.match_missed is None
    assert fetched.date_back is None


# ---------------------------------------------------------------------------
# Elo history
# ---------------------------------------------------------------------------
def test_create_elo_history_entry(db: Session):
    team = Team(name="Aston Villa")
    db.add(team)
    db.commit()

    payload = EloHistoryCreate(team_id=team.id, date=dt.date(2025, 1, 1), rating=1523.4)
    entry = EloHistory(**payload.model_dump())
    db.add(entry)
    db.commit()

    fetched = db.query(EloHistory).one()
    assert fetched.team.name == "Aston Villa"
    assert fetched.rating == pytest.approx(1523.4)


def test_elo_rating_must_be_positive_pydantic():
    with pytest.raises(Exception):
        EloHistoryCreate(team_id=1, date=dt.date(2025, 1, 1), rating=-10)


def test_one_elo_entry_per_team_per_date(db: Session):
    team = Team(name="Brighton")
    db.add(team)
    db.commit()

    db.add(EloHistory(team_id=team.id, date=dt.date(2025, 1, 1), rating=1500))
    db.commit()

    db.add(EloHistory(team_id=team.id, date=dt.date(2025, 1, 1), rating=1510))
    with pytest.raises(Exception):
        db.commit()
    db.rollback()


# ---------------------------------------------------------------------------
# End-to-end sanity check across all five tables together
# ---------------------------------------------------------------------------
def test_full_workflow_across_all_tables(db: Session):
    home = Team(name="Fulham", stadium_lat=51.475, stadium_lon=-0.2216, league="Premier League")
    away = Team(name="Brentford", stadium_lat=51.4906, stadium_lon=-0.2887, league="Premier League")
    db.add_all([home, away])
    db.commit()

    striker = Player(name="Star Striker", team_id=home.id, position="Forward", importance_weight=0.85)
    db.add(striker)
    db.commit()

    match = Match(
        date=dt.date(2025, 4, 12),
        home_team_id=home.id,
        away_team_id=away.id,
        home_score=None,
        away_score=None,
        competition="Premier League",
    )
    db.add(match)
    db.commit()

    injury = Injury(player_id=striker.id, date_out=dt.date(2025, 4, 1), match_missed_id=match.id)
    db.add(injury)

    db.add(EloHistory(team_id=home.id, date=dt.date(2025, 4, 12), rating=1610.2))
    db.add(EloHistory(team_id=away.id, date=dt.date(2025, 4, 12), rating=1495.7))
    db.commit()

    # Everything should now be joinable starting from the match:
    fetched_match = db.query(Match).one()
    assert fetched_match.home_team.name == "Fulham"
    assert fetched_match.injuries_missed[0].player.name == "Star Striker"
    home_elo = db.query(EloHistory).filter_by(team_id=home.id).one()
    assert home_elo.rating > 1600
