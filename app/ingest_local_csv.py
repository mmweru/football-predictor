"""
Ingests local/Kenyan-style match CSVs — a different format from
football-data.co.uk (different column names, yyyy-mm-dd dates, and
typically no venue data at all).

Expected columns: Date, League, Home Team, Away Team, Home Goals, Away Goals
(Season, Venue, and the derived columns like Result/Home Win/etc. are
ignored — they're redundant with what's computed elsewhere in this project.)

Every team ingested here is tagged with league="KPL" (a broad UI bucket —
see app.models.Team.league) regardless of which of the specific
sub-competitions it played in (Kenyan Premier League, county leagues,
cups, etc.) — those specific competition names are preserved per-match in
Match.competition, just not used for the top-level EPL/KPL UI split. See LOCAL_MATCH_DATA_GUIDE.md for why: this dataset spans 10
different Kenyan competitions, and grouping them under one UI bucket is a
deliberate simplification, not a data loss — nothing is discarded.

Usage:
    python -m app.ingest_local_csv path/to/kenyan_matches.csv
    python -m app.ingest_local_csv path/to/kenyan_matches.csv --league-group KPL --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Optional

import pandas as pd
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import Match, Team
from app.schemas import MatchCreate, TeamCreate
from app.team_aliases import normalize_team_name

REQUIRED_COLUMNS = ["Date", "League", "Home Team", "Away Team", "Home Goals", "Away Goals"]


def parse_date(raw: str) -> dt.date:
    return dt.datetime.strptime(raw.strip(), "%Y-%m-%d").date()


def get_or_create_team(db: Session, raw_name: str, league_group: str) -> Team:
    canonical_name = normalize_team_name(raw_name)
    team = db.query(Team).filter_by(name=canonical_name).one_or_none()
    if team is not None:
        return team

    payload = TeamCreate(name=canonical_name, league=league_group)
    team = Team(**payload.model_dump())
    db.add(team)
    db.flush()
    print(f"  [new team] {canonical_name!r} (league_group={league_group})")
    return team


def ingest_local_csv(csv_path: str, league_group: str = "KPL", dry_run: bool = False) -> None:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    inserted, skipped_duplicate, skipped_invalid, teams_touched = 0, 0, 0, set()

    try:
        for row_num, row in df.iterrows():
            try:
                match_date = parse_date(str(row["Date"]))
            except ValueError as e:
                print(f"  [row {row_num}] skipping — bad date: {e}")
                skipped_invalid += 1
                continue

            competition = str(row["League"])  # the SPECIFIC competition, preserved per-match
            home_raw, away_raw = str(row["Home Team"]), str(row["Away Team"])
            home_score, away_score = int(row["Home Goals"]), int(row["Away Goals"])

            home_team = get_or_create_team(db, home_raw, league_group)
            away_team = get_or_create_team(db, away_raw, league_group)
            teams_touched.update([home_team.name, away_team.name])

            try:
                payload = MatchCreate(
                    date=match_date, home_team_id=home_team.id, away_team_id=away_team.id,
                    home_score=home_score, away_score=away_score, competition=competition,
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
            db.rollback()

    finally:
        db.close()

    print("\n--- Local ingestion summary ---")
    print(f"Source file:          {csv_path}")
    print(f"League group:         {league_group}")
    print(f"Mode:                 {'DRY RUN (nothing written)' if dry_run else 'LIVE'}")
    print(f"Teams touched:        {len(teams_touched)}")
    print(f"Matches inserted:     {inserted}")
    print(f"Duplicates skipped:   {skipped_duplicate}")
    print(f"Invalid rows skipped: {skipped_invalid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a local/Kenyan-format match results CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--league-group", default="KPL", help="UI league bucket to tag every team with (default: KPL)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ingest_local_csv(args.csv_path, league_group=args.league_group, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
