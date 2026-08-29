"""
Tests for app.venue and app.set_match_venue.

Neither the real Nominatim service nor the real IP geolocation API are
reachable from this sandbox, so all network-dependent behavior is tested
against mocked functions — same pattern as tests/test_geocode.py. This
proves the LOGIC is correct; running the real thing against live services
is the next step on your machine.
"""

from collections import namedtuple

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Match, Team
from app.venue import geocode_address, get_ip_based_location

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


# ---------------------------------------------------------------------------
# geocode_address
# ---------------------------------------------------------------------------
def test_geocode_address_returns_coordinates():
    def fake_geocode_fn(query):
        assert query == "Kasarani Stadium, Nairobi"
        return FakeLocation(latitude=-1.2455, longitude=36.8989)

    result = geocode_address("Kasarani Stadium, Nairobi", geocode_fn=fake_geocode_fn)
    assert result == (-1.2455, 36.8989)


def test_geocode_address_returns_none_when_not_found():
    def fake_geocode_fn(query):
        return None

    result = geocode_address("Some Nonexistent Place XYZ", geocode_fn=fake_geocode_fn)
    assert result is None


def test_geocode_address_falls_back_to_photon_when_nominatim_fails(monkeypatch):
    """Simulates a Nominatim failure (e.g. the 403 seen in practice) and confirms
    the fallback geocoder is used instead of the whole lookup failing."""
    from app import venue as venue_module

    def failing_primary(address):
        raise Exception("Non-successful status code 403")

    def working_fallback(address):
        return FakeLocation(latitude=52.4068, longitude=-1.5197)  # Coventry

    monkeypatch.setattr(venue_module, "_get_geocode_fn", lambda: failing_primary)
    monkeypatch.setattr(venue_module, "_get_fallback_geocode_fn", lambda: working_fallback)

    result = geocode_address("Coventry stadium")
    assert result == (52.4068, -1.5197)


def test_geocode_address_returns_none_when_both_geocoders_fail(monkeypatch):
    from app import venue as venue_module

    def failing_primary(address):
        raise Exception("403")

    def failing_fallback(address):
        raise Exception("also failed")

    monkeypatch.setattr(venue_module, "_get_geocode_fn", lambda: failing_primary)
    monkeypatch.setattr(venue_module, "_get_fallback_geocode_fn", lambda: failing_fallback)

    result = geocode_address("Some Place")
    assert result is None


# ---------------------------------------------------------------------------
# get_ip_based_location
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, json_data, status_ok=True):
        self._json_data = json_data
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise Exception("HTTP error")

    def json(self):
        return self._json_data


def test_get_ip_based_location_success():
    def fake_http_get(url, timeout=5):
        return FakeResponse({"latitude": -1.286389, "longitude": 36.817223, "city": "Nairobi", "country_name": "Kenya"})

    result = get_ip_based_location(http_get=fake_http_get)
    assert result is not None
    lat, lon, description = result
    assert lat == pytest.approx(-1.286389)
    assert lon == pytest.approx(36.817223)
    assert "Nairobi" in description
    assert "Kenya" in description


def test_get_ip_based_location_handles_missing_fields():
    def fake_http_get(url, timeout=5):
        return FakeResponse({"latitude": None, "longitude": None})

    result = get_ip_based_location(http_get=fake_http_get)
    assert result is None


def test_get_ip_based_location_handles_request_failure():
    def fake_http_get(url, timeout=5):
        raise ConnectionError("network unreachable")

    result = get_ip_based_location(http_get=fake_http_get)
    assert result is None


# ---------------------------------------------------------------------------
# set_match_venue (the DB-writing functions)
# ---------------------------------------------------------------------------
def test_set_venue_from_coordinates_updates_match(db, monkeypatch):
    from app import set_match_venue

    home = Team(name="Home FC")
    away = Team(name="Away FC")
    db.add_all([home, away])
    db.commit()

    import datetime as dt
    match = Match(date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id)
    db.add(match)
    db.commit()

    monkeypatch.setattr(set_match_venue, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)

    set_match_venue.set_venue_from_coordinates(match.id, -1.2921, 36.8219, name="Test Field")

    refreshed = db.query(Match).filter_by(id=match.id).one()
    assert refreshed.venue_lat == pytest.approx(-1.2921)
    assert refreshed.venue_lon == pytest.approx(36.8219)
    assert refreshed.venue_name == "Test Field"


def test_set_venue_from_address_uses_geocoding(db, monkeypatch):
    from app import set_match_venue

    home = Team(name="Home FC")
    away = Team(name="Away FC")
    db.add_all([home, away])
    db.commit()

    import datetime as dt
    match = Match(date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id)
    db.add(match)
    db.commit()

    monkeypatch.setattr(set_match_venue, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(set_match_venue, "geocode_address", lambda address: (-1.2455, 36.8989))

    set_match_venue.set_venue_from_address(match.id, "Kasarani Stadium, Nairobi")

    refreshed = db.query(Match).filter_by(id=match.id).one()
    assert refreshed.venue_lat == pytest.approx(-1.2455)
    assert refreshed.venue_name == "Kasarani Stadium, Nairobi"


def test_set_venue_from_address_handles_geocoding_failure(db, monkeypatch, capsys):
    from app import set_match_venue

    home = Team(name="Home FC")
    away = Team(name="Away FC")
    db.add_all([home, away])
    db.commit()

    import datetime as dt
    match = Match(date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id)
    db.add(match)
    db.commit()

    monkeypatch.setattr(set_match_venue, "geocode_address", lambda address: None)

    set_match_venue.set_venue_from_address(match.id, "Nonexistent Place")

    refreshed = db.query(Match).filter_by(id=match.id).one()
    assert refreshed.venue_lat is None  # nothing should have been written
    captured = capsys.readouterr()
    assert "Could not geocode" in captured.out


def test_set_venue_from_ip_autodetect_requires_confirmation(db, monkeypatch, capsys):
    from app import set_match_venue

    home = Team(name="Home FC")
    away = Team(name="Away FC")
    db.add_all([home, away])
    db.commit()

    import datetime as dt
    match = Match(date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id)
    db.add(match)
    db.commit()

    monkeypatch.setattr(set_match_venue, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(
        set_match_venue, "get_ip_based_location", lambda: (-1.286389, 36.817223, "Nairobi, Kenya")
    )

    # Simulate the user declining confirmation
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    set_match_venue.set_venue_from_ip_autodetect(match.id)

    refreshed = db.query(Match).filter_by(id=match.id).one()
    assert refreshed.venue_lat is None  # declined, nothing written

    # Now simulate the user confirming
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    set_match_venue.set_venue_from_ip_autodetect(match.id)

    refreshed = db.query(Match).filter_by(id=match.id).one()
    assert refreshed.venue_lat == pytest.approx(-1.286389)


def test_set_venue_missing_match_id_reports_cleanly(db, monkeypatch, capsys):
    from app import set_match_venue

    monkeypatch.setattr(set_match_venue, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)

    set_match_venue.set_venue_from_coordinates(99999, -1.29, 36.82)

    captured = capsys.readouterr()
    assert "No match found" in captured.out
