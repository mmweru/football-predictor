"""
Ingests historical match results from a football-data.co.uk-format CSV
into the database, validating every row through Pydantic before it
touches SQLAlchemy.

Expected CSV columns (standard football-data.co.uk format):
    Date, HomeTeam, AwayTeam, FTHG, FTAG   [Div is optional/ignored — see --competition]

Date is accepted in either dd/mm/yy or dd/mm/yyyy form.

Usage:
    python -m app.ingest_csv path/to/E0.csv --competition "Premier League"

Source for this format: https://football-data.co.uk/notes.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Optional

import pandas as pd
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import Match, Team
from app.schemas import MatchCreate, TeamCreate
from app.team_aliases import normalize_team_name

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def parse_date(raw: str) -> dt.date:
    """football-data.co.uk uses dd/mm/yy in older files and dd/mm/yyyy in newer ones."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return dt.datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {raw!r}")


def get_or_create_team(db: Session, raw_name: str, league: Optional[str] = None) -> Team:
    """
    Looks up a team by its canonical (normalized) name; creates it if it
    doesn't exist yet. This is the single choke point where every match's
    team names funnel through normalize_team_name, so duplicates like
    "Man Utd" vs "Manchester United" collapse into one row here rather
    than propagating into `matches`.
    """
    canonical_name = normalize_team_name(raw_name)

    team = db.query(Team).filter_by(name=canonical_name).one_or_none()
    if team is not None:
        return team

    # Validate through Pydantic before creating — this is where you'd catch
    # e.g. a name that's empty or absurdly long, before it hits the DB.
    payload = TeamCreate(name=canonical_name, league=league)
    team = Team(**payload.model_dump())
    db.add(team)
    db.flush()  # assigns team.id without committing the whole transaction yet
    print(f"  [new team] {canonical_name!r}" + (f" (raw source name: {raw_name!r})" if raw_name != canonical_name else ""))
    return team


def ingest_csv(csv_path: str, competition: Optional[str], dry_run: bool = False) -> None:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    inserted, skipped_duplicate, skipped_invalid, teams_touched = 0, 0, 0, set()

    try:
        for row_num, row in df.iterrows():
            try:
                match_date = parse_date(str(row["Date"]))
            except ValueError as e:
                print(f"  [row {row_num}] skipping — {e}")
                skipped_invalid += 1
                continue

            home_raw, away_raw = str(row["HomeTeam"]), str(row["AwayTeam"])

            # FTHG/FTAG can be blank for postponed/unplayed fixtures in some files.
            home_score = int(row["FTHG"]) if pd.notna(row["FTHG"]) else None
            away_score = int(row["FTAG"]) if pd.notna(row["FTAG"]) else None

            home_team = get_or_create_team(db, home_raw, league=competition)
            away_team = get_or_create_team(db, away_raw, league=competition)
            teams_touched.update([home_team.name, away_team.name])

            try:
                payload = MatchCreate(
                    date=match_date,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    home_score=home_score,
                    away_score=away_score,
                    competition=competition,
                )
            except ValidationError as e:
                print(f"  [row {row_num}] validation failed — {e}")
                skipped_invalid += 1
                continue

            if dry_run:
                inserted += 1
                continue

            match = Match(**payload.model_dump())
            db.add(match)
            try:
                db.flush()
                inserted += 1
            except IntegrityError:
                db.rollback()
                skipped_duplicate += 1

        if not dry_run:
            db.commit()
        else:
            db.rollback()  # dry run: never persist anything

    finally:
        db.close()

    print("\n--- Ingestion summary ---")
    print(f"Source file:          {csv_path}")
    print(f"Mode:                 {'DRY RUN (nothing written)' if dry_run else 'LIVE'}")
    print(f"Teams touched:        {len(teams_touched)}")
    print(f"Matches inserted:     {inserted}")
    print(f"Duplicates skipped:   {skipped_duplicate}")
    print(f"Invalid rows skipped: {skipped_invalid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a football-data.co.uk-format results CSV.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument(
        "--competition", default=None, help="Competition/league label to tag every match with (e.g. 'Premier League')"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without writing anything to the database"
    )
    args = parser.parse_args()

    ingest_csv(args.csv_path, args.competition, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
