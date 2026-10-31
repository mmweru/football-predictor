from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app import geocode_teams
from app.geocode_teams import geocode_stadium
from app.models import Team
from app.venue import is_uk_coordinate


def test_uk_coordinate_bounds_accept_uk_and_reject_wrong_country_results():
    assert is_uk_coordinate(51.5, -0.1)
    assert not is_uk_coordinate(39.0, -94.0)
    assert not is_uk_coordinate(-26.1, 28.0)


def test_stadium_lookup_qualifies_query_and_rejects_foreign_result():
    captured = []

    def geocode(query):
        captured.append(query)
        return SimpleNamespace(latitude=39.05, longitude=-94.48)

    assert geocode_stadium(geocode, "Notts County") is None
    assert "United Kingdom" in captured[0]


def test_stoke_lookup_uses_specific_stadium_and_city():
    captured = []

    def geocode(query):
        captured.append(query)
        return SimpleNamespace(latitude=52.9884, longitude=-2.1755)

    assert geocode_stadium(geocode, "Stoke") == (52.9884, -2.1755)
    assert captured == ["bet365 Stadium, Stoke-on-Trent, United Kingdom"]


def test_stadium_lookup_accepts_uk_result():
    result = geocode_stadium(lambda query: SimpleNamespace(latitude=53.0, longitude=-2.0), "Notts County")
    assert result == (53.0, -2.0)


def test_invalid_saved_coordinates_are_regeocoded(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, future=True)()
    try:
        team = Team(name="Wolverhampton Wanderers", league="Premier League", stadium_lat=-26.1,
                    stadium_lon=28.0, is_active=True)
        db.add(team)
        db.commit()
        monkeypatch.setattr(geocode_teams, "SessionLocal", lambda: db)
        monkeypatch.setattr(geocode_teams, "build_geocoder", lambda: None)
        monkeypatch.setattr(geocode_teams, "geocode_stadium", lambda geocode_fn, name: (52.6, -2.1))

        geocode_teams.geocode_all_teams()

        check_db = sessionmaker(bind=engine, future=True)()
        saved = check_db.query(Team).one()
        assert (saved.stadium_lat, saved.stadium_lon) == (52.6, -2.1)
        check_db.close()
    finally:
        db.close()
        engine.dispose()


def test_explicit_team_selection_regeocodes_valid_but_suspicious_coordinates(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, future=True)()
    try:
        team = Team(name="Stoke", league="EFL Championship", stadium_lat=52.055,
                    stadium_lon=1.1453, is_active=True)
        db.add(team)
        db.commit()
        monkeypatch.setattr(geocode_teams, "SessionLocal", lambda: db)
        monkeypatch.setattr(geocode_teams, "build_geocoder", lambda: None)
        monkeypatch.setattr(geocode_teams, "geocode_stadium", lambda geocode_fn, name: (52.9884, -2.1755))

        geocode_teams.geocode_all_teams(only_team="Stoke")

        check_db = sessionmaker(bind=engine, future=True)()
        saved = check_db.query(Team).one()
        assert (saved.stadium_lat, saved.stadium_lon) == (52.9884, -2.1755)
        check_db.close()
    finally:
        db.close()
        engine.dispose()