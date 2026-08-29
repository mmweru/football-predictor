"""
Inserts a handful of sample rows into every table so you can SEE data in
pgAdmin and manually confirm the schema/relationships look right — separate
from the automated pytest suite, which uses throwaway in-memory data.

Run:
    python -m app.seed_sample_data

Safe to run multiple times against a fresh database; if you re-run it after
data already exists you'll hit the unique constraints (that's expected —
it's proof the constraints work). Wipe the tables first if you want a clean
re-seed (see README "Resetting the database").
"""

import datetime as dt

from app.database import Base, SessionLocal, engine
from app.models import EloHistory, Injury, Match, Player, Team


def seed() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        arsenal = Team(name="Arsenal", stadium_lat=51.5549, stadium_lon=-0.1084, league="Premier League")
        chelsea = Team(name="Chelsea", stadium_lat=51.4816, stadium_lon=-0.1909, league="Premier League")
        db.add_all([arsenal, chelsea])
        db.commit()
        print(f"Inserted teams: {arsenal}, {chelsea}")

        match = Match(
            date=dt.date(2025, 10, 4),
            home_team_id=arsenal.id,
            away_team_id=chelsea.id,
            home_score=2,
            away_score=2,
            competition="Premier League",
        )
        db.add(match)
        db.commit()
        print(f"Inserted match: {match}")

        striker = Player(name="Sample Striker", team_id=arsenal.id, position="Forward", importance_weight=0.82)
        db.add(striker)
        db.commit()
        print(f"Inserted player: {striker}")

        injury = Injury(
            player_id=striker.id,
            date_out=dt.date(2025, 9, 20),
            date_back=dt.date(2025, 10, 10),
            match_missed_id=match.id,
        )
        db.add(injury)
        db.commit()
        print(f"Inserted injury: {injury}")

        db.add(EloHistory(team_id=arsenal.id, date=dt.date(2025, 10, 4), rating=1618.3))
        db.add(EloHistory(team_id=chelsea.id, date=dt.date(2025, 10, 4), rating=1587.9))
        db.commit()
        print("Inserted elo_history rows for both teams")

        print("\nSeed complete. Open pgAdmin and check the tables now.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
