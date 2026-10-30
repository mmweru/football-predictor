import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.merge_duplicates import merge_duplicate_teams
from app.models import Match, Prediction, Team


def test_duplicate_team_merge_collapses_rows_by_database_fixture_key():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, future=True)()
    try:
        canonical = Team(name="Crystal Palace", league="Premier League")
        duplicate = Team(name="Crystal Palace FC", league="Premier League")
        away = Team(name="Arsenal", league="Premier League")
        db.add_all([canonical, duplicate, away])
        db.flush()

        played = Match(
            date=dt.date(2026, 10, 31), home_team_id=canonical.id, away_team_id=away.id,
            home_score=1, away_score=0, competition="Premier League", season=2026,
        )
        fixture = Match(
            date=dt.date(2026, 10, 31), home_team_id=duplicate.id, away_team_id=away.id,
            competition=None, season=None,
        )
        db.add_all([played, fixture])
        db.flush()
        db.add(Prediction(
            match_id=fixture.id, model_version="test", p_home=0.5, p_draw=0.25, p_away=0.25, predicted="H",
        ))
        db.commit()

        result = merge_duplicate_teams(db)

        matches = db.query(Match).all()
        assert len(matches) == 1
        assert (matches[0].home_team_id, matches[0].away_team_id) == (canonical.id, away.id)
        assert (matches[0].home_score, matches[0].away_score) == (1, 0)
        assert db.query(Prediction).count() == 0
        assert result["matches_merged"] == 1
    finally:
        db.close()
        engine.dispose()