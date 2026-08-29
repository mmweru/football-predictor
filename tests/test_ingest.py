"""
Tests for app.ingest_csv — team name normalization and full CSV ingestion.

Run with: pytest -v
"""

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.ingest_csv import get_or_create_team, parse_date
from app.models import Match, Team
from app.team_aliases import normalize_team_name


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
# Date parsing
# ---------------------------------------------------------------------------
def test_parse_date_two_digit_year():
    assert parse_date("11/08/23") == dt.date(2023, 8, 11)


def test_parse_date_four_digit_year():
    assert parse_date("11/08/2023") == dt.date(2023, 8, 11)


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date("not-a-date")


# ---------------------------------------------------------------------------
# Team name normalization
# ---------------------------------------------------------------------------
def test_normalize_known_alias():
    assert normalize_team_name("Man Utd") == "Manchester United"
    assert normalize_team_name("man utd") == "Manchester United"  # case-insensitive


def test_normalize_unmapped_name_passes_through():
    assert normalize_team_name("Some New Club") == "Some New Club"


def test_get_or_create_team_deduplicates_aliases(db):
    t1 = get_or_create_team(db, "Man Utd")
    db.commit()
    t2 = get_or_create_team(db, "Man United")  # different alias, same canonical target... 
    db.commit()

    # NOTE: "Man United" isn't in the alias table (only "man united" lowercase key is),
    # so this specifically tests the case-insensitive matching path.
    assert t1.id == t2.id
    assert db.query(Team).count() == 1


def test_get_or_create_team_creates_distinct_teams_for_distinct_names(db):
    t1 = get_or_create_team(db, "Arsenal")
    db.commit()
    t2 = get_or_create_team(db, "Chelsea")
    db.commit()

    assert t1.id != t2.id
    assert db.query(Team).count() == 2


# ---------------------------------------------------------------------------
# Full CSV ingestion (via a temp file, exercising the real function)
# ---------------------------------------------------------------------------
def test_ingest_csv_full_flow(tmp_path, monkeypatch):
    import app.database as db_module
    from app.ingest_csv import ingest_csv

    # Point the module-level engine/session at a fresh temp SQLite file for this test.
    test_db_path = tmp_path / "test.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}", future=True)
    TestSessionLocal = sessionmaker(bind=test_engine, future=True)

    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr("app.ingest_csv.engine", test_engine)
    monkeypatch.setattr("app.ingest_csv.SessionLocal", TestSessionLocal)

    csv_content = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,11/08/23,Arsenal,Man Utd,2,1,H\n"
        "E0,19/08/23,Man Utd,Arsenal,0,0,D\n"
    )
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(csv_content)

    ingest_csv(str(csv_file), competition="Premier League")

    session = TestSessionLocal()
    try:
        teams = session.query(Team).all()
        names = sorted(t.name for t in teams)
        assert names == ["Arsenal", "Manchester United"]  # alias correctly normalized

        matches = session.query(Match).all()
        assert len(matches) == 2
        assert all(m.competition == "Premier League" for m in matches)
    finally:
        session.close()


def test_ingest_csv_missing_required_column_raises(tmp_path):
    from app.ingest_csv import ingest_csv

    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("Date,HomeTeam,AwayTeam\n11/08/23,Arsenal,Chelsea\n")  # missing FTHG/FTAG

    with pytest.raises(ValueError, match="missing required columns"):
        ingest_csv(str(csv_file), competition=None)
