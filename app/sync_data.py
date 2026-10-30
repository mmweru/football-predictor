"""
One command to bring the whole database up to date.

    python -m app.migrate                       # once, after upgrading
    python -m app.sync_data --seasons 8         # first run: 8 seasons x 4 leagues, fixtures, membership, Elo
    python -m app.sync_data                     # later runs: current season + fixtures (fast)
    python -m app.sync_data --all               # + football-data.org schedule, FPL injuries/players, weather
    python -m app.sync_data --local-csv "E0*.csv" --local-div E0     # load CSVs you downloaded by hand

Every step is independent: if one source is down the others still run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
from typing import List, Optional

import pandas as pd

from app.config import settings
from app.data_sources import footballdata_uk as fduk
from app.data_sources.http import DataSourceError
from app.database import Base, SessionLocal, engine
from app.leagues import BY_DIV, LEAGUES, season_start_year
from app.sync_core import TeamResolver, purge_stale_fixtures, possible_duplicate_teams, refresh_membership, upsert_matches


def _step(name: str, fn):
    try:
        return fn()
    except (DataSourceError, ValueError) as e:
        print(f"  [{name}] FAILED, continuing: {e}")
    except Exception as e:  # noqa: BLE001 - one broken source must never abort the whole sync
        print(f"  [{name}] unexpected error ({type(e).__name__}), continuing: {e}")


def sync(seasons: int = 1, use_fdorg: bool = False, use_fpl: bool = False, use_weather: bool = False,
         use_bigballs: bool = False, local_csv: Optional[str] = None, local_div: str = "E0", today: Optional[dt.date] = None,
         use_api_football: bool = False, api_football_seasons: Optional[List[int]] = None) -> None:
    from app.migrate import migrate
    migrate()
    today = today or dt.date.today()
    active = season_start_year(today)
    db = SessionLocal()
    official_rosters = {}
    try:
        # 1. results + stats + odds
        if local_csv:
            frames = [fduk.load_local_csv(f, BY_DIV[local_div]) for f in sorted(glob.glob(local_csv))]
            hist = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            hist = _step("history", lambda: fduk.fetch_history(range(active - seasons + 1, active + 1))) 
        if hist is not None and len(hist):
            print("history:", upsert_matches(db, hist))

        # 2. upcoming fixtures (+ their pre-match odds)
        fx = None if local_csv else _step("fixtures.csv", lambda: fduk.fetch_fixtures(today=today))
        if fx is not None and len(fx):
            print("fixtures:", upsert_matches(db, fx), "| stale removed:", purge_stale_fixtures(db, fx, today))

        # 3. optional richer sources
        if use_fdorg:
            official_rosters = _step("football-data.org", lambda: _sync_fdorg(db, active)) or {}
        if use_fpl:
            from app.data_sources.fpl import sync_fpl
            print("fpl:", _step("fpl", lambda: sync_fpl(db, today=today)))
        if use_bigballs and settings.bigballs_api_key:
            from app.data_sources.bigballs import sync_epl_injuries
            print("BigBalls EPL injuries:", _step("BigBalls injuries", lambda: sync_epl_injuries(db, today=today)))

        if use_api_football:
            if not settings.football_api_key:
                print("[skip] API-Football: FOOTBALL_API_KEY is not set in .env")
            else:
                from app.data_sources.api_football import APIFootball, sync_api_football
                target_seasons = api_football_seasons or [2024, 2025]
                api = APIFootball(settings.football_api_key)
                print("API-Football:", _step("API-Football", lambda: sync_api_football(db, api, target_seasons)))

        # 4. normalize duplicate source names before assigning current league membership
        from app.merge_duplicates import merge_duplicate_teams
        merged = merge_duplicate_teams(db)
        if merged["teams_merged"] or merged["teams_renamed"]:
            print("team-name repairs:", merged)
        rep = refresh_membership(db, active, official_rosters=official_rosters)
        print(f"\nActive clubs for {active}/{(active + 1) % 100:02d}: {rep['per_league']}")
        for w in rep["warnings"]:
            print("  WARNING:", w)
        dups = possible_duplicate_teams(db)
        if dups:
            print("  Possible duplicate team names (add to app/team_aliases.py):")
            for a, b, r in dups[:15]:
                print(f"    {a!r} ~ {b!r} ({r})")

        if use_weather:
            from app.data_sources.weather import fill_weather
            print("weather rows filled:", _step("weather", lambda: fill_weather(db, today=today)))

        # Older deployments used a separate EloHistory builder; the current feature engine derives Elo from matches.
        try:
            from app.elo import build_elo_history
        except ModuleNotFoundError as error:
            if error.name != "app.elo":
                raise
            print("[skip] legacy Elo history: app.elo is not installed; live predictions compute Elo from match history")
        else:
            build_elo_history(db, k=20.0, home_advantage=70.0)
    finally:
        db.close()


def _sync_fdorg(db, active: int) -> dict:
    from app.data_sources.footballdata_org import FootballDataOrg
    fd = FootballDataOrg(settings.football_data_org_token)
    resolver = TeamResolver(db)
    rosters = {}
    for lg in fd.free_leagues():
        df = fd.matches(lg, active)
        print(f"football-data.org {lg.name}:", upsert_matches(db, df, resolver=resolver))
        teams = {resolver.get(t["name"], lg.name).name: t["id"] for t in fd.teams(lg, active)}
        for name, fid in teams.items():
            resolver._cache[name].fdorg_id = fid
        db.commit()
        print(f"  official club count: {len(teams)} (expected {lg.expected_teams})")
        rosters[lg.name] = set(teams)
    return rosters


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync results, fixtures, odds, injuries and weather for EPL + EFL.")
    ap.add_argument("--seasons", type=int, default=1, help="how many seasons of history to (re)download, ending with the current one")
    ap.add_argument("--fdorg", action="store_true", help="football-data.org full schedule (PL + Championship; needs token)")
    ap.add_argument("--fpl", action="store_true", help="FPL injuries + player stats (EPL)")
    ap.add_argument("--weather", action="store_true", help="Open-Meteo weather (needs stadium coordinates)")
    ap.add_argument("--bigballs", action="store_true")
    ap.add_argument("--api-football", action="store_true", help="import API-Football fixtures and player stats (uses paginated API requests)")
    ap.add_argument("--api-football-seasons", type=int, nargs="+", default=[2024], help="season start years (free plan tested through 2024)")
    ap.add_argument("--all", action="store_true", help="everything configured")
    ap.add_argument("--local-csv", default=None, help='glob of football-data.co.uk CSV files you already have, e.g. "E0*.csv"')
    ap.add_argument("--local-div", default="E0", choices=list(BY_DIV))
    a = ap.parse_args()
    sync(a.seasons, use_fdorg=a.fdorg or (a.all and bool(settings.football_data_org_token)), use_fpl=a.fpl or a.all,
            use_weather=a.weather or a.all, use_bigballs=a.bigballs or a.all, local_csv=a.local_csv, local_div=a.local_div,
            use_api_football=a.api_football, api_football_seasons=a.api_football_seasons)


if __name__ == "__main__":
    main()
