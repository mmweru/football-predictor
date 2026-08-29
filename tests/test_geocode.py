"""
Tests for app.geocode_teams — the LOGIC only, via a mocked geocode function.

This does NOT hit the real Nominatim service (not reachable from this
sandbox's network, and we shouldn't hammer a free public service during
automated tests anyway). It proves geocode_stadium() and geocode_all_teams()
correctly parse a geocoder's response and write it to the database — you
still need to run app.geocode_teams for real, once, against actual Nominatim.
"""

from collections import namedtuple

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.geocode_teams import geocode_stadium
from app.models import Team

FakeLocation = namedtuple("FakeLocation", ["latitude", "longitude"])


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


def test_geocode_stadium_returns_lat_lon_on_success():
    def fake_geocode_fn(query):
        assert query == "Arsenal stadium"  # confirms the query format sent to the geocoder
        return FakeLocation(latitude=51.5549, longitude=-0.1084)

    result = geocode_stadium(fake_geocode_fn, "Arsenal")
    assert result == (51.5549, -0.1084)


def test_geocode_stadium_returns_none_when_not_found():
    def fake_geocode_fn(query):
        return None

    result = geocode_stadium(fake_geocode_fn, "Some Obscure Local Club FC")
    assert result is None


def test_geocode_all_teams_writes_coordinates_to_db(db, monkeypatch):
    from app import geocode_teams

    team = Team(name="Arsenal")
    db.add(team)
    db.commit()

    # Patch SessionLocal used inside geocode_teams to return OUR test session.
    monkeypatch.setattr(geocode_teams, "SessionLocal", lambda: db)
    # Preflight check (build_geocoder) is a no-op for this test — its real
    # behavior (the placeholder-User-Agent check) is tested separately in
    # tests/test_venue_timeout.py.
    monkeypatch.setattr(geocode_teams, "build_geocoder", lambda: None)
    # geocode_all_teams calls geocode_stadium(None, team.name) for each team —
    # patch geocode_stadium itself (the actual per-team lookup call) rather
    # than trying to intercept the network layer underneath it.
    monkeypatch.setattr(
        geocode_teams, "geocode_stadium", lambda geocode_fn, name: (51.5549, -0.1084)
    )

    # db.close() is called inside geocode_all_teams's finally block; make it a no-op
    # so our fixture's session stays usable for the assertion below.
    monkeypatch.setattr(db, "close", lambda: None)

    geocode_teams.geocode_all_teams()

    refreshed = db.query(Team).filter_by(name="Arsenal").one()
    assert refreshed.stadium_lat == pytest.approx(51.5549)
    assert refreshed.stadium_lon == pytest.approx(-0.1084)


def test_geocode_all_teams_skips_teams_that_already_have_coordinates(db, monkeypatch):
    from app import geocode_teams

    team = Team(name="Chelsea", stadium_lat=51.4816, stadium_lon=-0.1909)
    db.add(team)
    db.commit()

    monkeypatch.setattr(geocode_teams, "SessionLocal", lambda: db)

    call_count = {"n": 0}

    def fake_build_geocoder():
        def fn(query):
            call_count["n"] += 1
            return FakeLocation(latitude=0, longitude=0)
        return fn

    monkeypatch.setattr(geocode_teams, "build_geocoder", fake_build_geocoder)
    monkeypatch.setattr(db, "close", lambda: None)

    geocode_teams.geocode_all_teams()

    assert call_count["n"] == 0  # never called — team already had coordinates


def test_geocode_all_teams_uses_league_fallback_when_exact_geocoding_fails(db, monkeypatch):
    from app import geocode_teams

    team = Team(name="Obscure Local Club", league="KPL")  # no coordinates yet
    db.add(team)
    db.commit()

    monkeypatch.setattr(geocode_teams, "SessionLocal", lambda: db)
    monkeypatch.setattr(geocode_teams, "build_geocoder", lambda: None)
    monkeypatch.setattr(geocode_teams, "geocode_stadium", lambda geocode_fn, name: None)  # exact lookup always fails
    monkeypatch.setattr(db, "close", lambda: None)

    geocode_teams.geocode_all_teams()

    refreshed = db.query(Team).filter_by(name="Obscure Local Club").one()
    # Should have been assigned the KPL fallback (Nairobi), not left as None.
    assert refreshed.stadium_lat is not None
    assert refreshed.stadium_lon is not None
    assert refreshed.stadium_lat == pytest.approx(-1.286389)


def test_geocode_all_teams_leaves_unmapped_league_teams_as_not_found(db, monkeypatch):
    from app import geocode_teams

    team = Team(name="Some Other Team", league="SomeUnmappedLeague")
    db.add(team)
    db.commit()

    monkeypatch.setattr(geocode_teams, "SessionLocal", lambda: db)
    monkeypatch.setattr(geocode_teams, "build_geocoder", lambda: None)
    monkeypatch.setattr(geocode_teams, "geocode_stadium", lambda geocode_fn, name: None)
    monkeypatch.setattr(db, "close", lambda: None)

    geocode_teams.geocode_all_teams()

    refreshed = db.query(Team).filter_by(name="Some Other Team").one()
    assert refreshed.stadium_lat is None  # no fallback defined for this league -> stays None
