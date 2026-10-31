import datetime as dt

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.data_sources import weather
from app.models import Match, Team


def test_weather_bulk_updates_avoid_expired_match_refresh_queries(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, future=True)()
    today = dt.date(2026, 10, 31)
    match_selects = 0

    @event.listens_for(engine, "before_cursor_execute")
    def count_match_selects(connection, cursor, statement, parameters, context, executemany):
        nonlocal match_selects
        normalized = statement.casefold()
        if normalized.lstrip().startswith("select") and "from matches" in normalized:
            match_selects += 1

    try:
        home_a = Team(name="Home A", league="Premier League", is_active=True, stadium_lat=51.0, stadium_lon=-0.1)
        home_b = Team(name="Home B", league="Premier League", is_active=True, stadium_lat=52.0, stadium_lon=-0.2)
        away = Team(name="Away", league="Premier League", is_active=True)
        db.add_all([home_a, home_b, away])
        db.flush()
        dates = [today - dt.timedelta(days=2), today + dt.timedelta(days=2)]
        db.add_all([
            Match(date=dates[0], home_team_id=home_a.id, away_team_id=away.id, competition="Premier League", season=2026),
            Match(date=dates[1], home_team_id=home_b.id, away_team_id=away.id, competition="Premier League", season=2026),
        ])
        db.commit()
        monkeypatch.setattr(
            weather, "fetch_daily",
            lambda lat, lon, start, end, session, current_day: {
                start: (12.0, 1.0, 10.0) if start < current_day else (13.0, 0.0, 8.0),
            },
        )

        assert weather.fill_weather(db, today=today) == 2
        assert match_selects == 1
        rows = db.query(Match).order_by(Match.date).all()
        assert [row.temp_c for row in rows] == [12.0, 13.0]
    finally:
        db.close()
        engine.dispose()