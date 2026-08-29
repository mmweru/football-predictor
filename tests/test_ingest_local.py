"""
Tests for app.ingest_local_csv — using a small synthetic CSV that mirrors
the exact column format of the real uploaded Kenyan football dataset
(different from football-data.co.uk's format: yyyy-mm-dd dates, different
column names, no venue data).
"""

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.ingest_local_csv import ingest_local_csv, parse_date
from app.models import Match, Team


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


def test_parse_date_yyyy_mm_dd():
    assert parse_date("2025-01-11") == dt.date(2025, 1, 11)


def test_parse_date_rejects_dd_mm_yyyy():
    # This format is football-data.co.uk's format, NOT this script's format —
    # confirms we don't accidentally silently accept the wrong format.
    with pytest.raises(ValueError):
        parse_date("11/01/2025")


def test_ingest_local_csv_full_flow(tmp_path, monkeypatch):
    import app.database as db_module
    import app.ingest_local_csv as ingest_module

    test_db_path = tmp_path / "test.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}", future=True)
    TestSessionLocal = sessionmaker(bind=test_engine, future=True)
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(ingest_module, "engine", test_engine)
    monkeypatch.setattr(ingest_module, "SessionLocal", TestSessionLocal)

    csv_content = (
        "Date,League,Season,Home Team,Away Team,Home Goals,Away Goals,Venue,Result\n"
        "2025-01-11,Kenyan Premier League,2025,Gor Mahia,AFC Leopards,2,1,Not listed,Home\n"
        "2025-01-18,FKF Division One,2025,Mombasa Stars,Nzoia Sugar,0,0,Not listed,Draw\n"
    )
    csv_file = tmp_path / "local.csv"
    csv_file.write_text(csv_content)

    ingest_local_csv(str(csv_file), league_group="KPL")

    session = TestSessionLocal()
    try:
        teams = session.query(Team).all()
        assert len(teams) == 4
        assert all(t.league == "KPL" for t in teams)  # all tagged with the UI bucket

        matches = session.query(Match).order_by(Match.date).all()
        assert len(matches) == 2
        # Specific competition name preserved per-match, NOT overwritten with "KPL":
        assert matches[0].competition == "Kenyan Premier League"
        assert matches[1].competition == "FKF Division One"
    finally:
        session.close()


def test_ingest_local_csv_missing_column_raises(tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("Date,Home Team,Away Team\n2025-01-01,A,B\n")  # missing League/goals columns

    with pytest.raises(ValueError, match="missing required columns"):
        ingest_local_csv(str(csv_file))


def test_ingest_local_csv_dry_run_writes_nothing(tmp_path, monkeypatch):
    import app.database as db_module
    import app.ingest_local_csv as ingest_module

    test_db_path = tmp_path / "test.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}", future=True)
    TestSessionLocal = sessionmaker(bind=test_engine, future=True)
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(ingest_module, "engine", test_engine)
    monkeypatch.setattr(ingest_module, "SessionLocal", TestSessionLocal)

    csv_file = tmp_path / "local.csv"
    csv_file.write_text("Date,League,Home Team,Away Team,Home Goals,Away Goals\n2025-01-11,KPL,A,B,1,0\n")

    ingest_local_csv(str(csv_file), dry_run=True)

    session = TestSessionLocal()
    try:
        assert session.query(Team).count() == 0
        assert session.query(Match).count() == 0
    finally:
        session.close()
